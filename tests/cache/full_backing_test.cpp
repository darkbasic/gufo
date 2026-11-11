#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <latch>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/component_cache_resources.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/core/resource_error.hpp"
#include "src/families/qwen/vision/admission.hpp"
#include "src/models/qwen38_flash_next/cache_admission.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace {
std::map<void*, std::size_t> registered;
std::atomic<std::size_t> registered_bytes{};
std::function<void()> verify_segment_ownership;
bool fail_registration{}, fail_query{};
std::atomic<unsigned> native_creations{}, live_streams{}, live_events{};
std::atomic<unsigned> system_release_events{}, stream_synchronizations{};
std::atomic<unsigned> event_waits{};
int fail_native_creation{-1};
bool fail_event_record{};
std::atomic<bool> fail_copy_after_submission{}, fail_event_wait{};
std::atomic<bool> fail_stream_wait{}, fail_event_query{};

// HIP calls enqueue bounded work here. Waiting for an event advances only its
// recorded prefix, so a whole-stream drain cannot masquerade as a lease fence.
struct NativeGate {
  std::latch entered{1}, release{1};
};
struct NativeWork {
  std::size_t sequence;
  std::function<void()> run;
};
struct NativeStream {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<NativeWork> work;
  std::size_t submitted{}, completed{};
  bool running{};
};
struct NativeEvent {
  NativeStream* stream{};
  std::size_t sequence{};
};
// Configured only by the submitting thread before its next copy.
thread_local NativeGate* next_copy_gate{};

void RunThrough(NativeStream& stream, std::size_t sequence) {
  std::unique_lock lock(stream.mutex);
  while (stream.completed < sequence) {
    if (stream.running) {
      stream.changed.wait(lock, [&] {
        return !stream.running || stream.completed >= sequence;
      });
      continue;
    }
    assert(!stream.work.empty());
    auto work = std::move(stream.work.front());
    stream.work.pop_front();
    stream.running = true;
    lock.unlock();
    work.run();
    lock.lock();
    stream.completed = work.sequence;
    stream.running = false;
    stream.changed.notify_all();
  }
}
constexpr std::size_t kPayloadBytes = 4 * 4096;
constexpr std::size_t kMetadataBytes = 16384;
constexpr std::size_t kBudgetBytes = 1 << 20;

class Adapter final : public gufo::cache::testing::FakeAdapter {
public:
  std::vector<gufo::cache::BackingClass> BackingPlan(
      std::size_t) const override {
    return {
        {.block_bytes = 4096, .blocks = 4, .metadata_bytes = kMetadataBytes}};
  }
};

void Check(bool full, bool headroom, bool registration, bool query) {
  using namespace gufo;
  fail_registration = !registration;
  fail_query = !query;
  cache::ResourceLedger ledger(
      {kBudgetBytes, kBudgetBytes, kBudgetBytes, kBudgetBytes});
  std::string diagnostic;
  bool refused{};
  {
    hip::ComponentCacheHooks hooks;
    assert(!hooks.require_full_backing);
    hooks.require_full_backing = full;
    hooks.transfer_streams = 1;
    hooks.host_ceiling = [headroom] { return headroom ? kBudgetBytes : 0; };
    hooks.warn = [](std::string_view) {};
    hooks.info = [&](std::string_view text) { diagnostic = text; };
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(), ledger, std::move(hooks));
    try {
      resources->commit_backing(kBudgetBytes);
    } catch (const core::ResourceAdmissionError& error) {
      refused = true;
      assert(
          std::string_view(error.what()).find("required_payload_bytes=16384") !=
          std::string_view::npos);
    }
    const bool backed = headroom && registration && query;
    assert(refused == (full && !backed));
    const auto payload = ledger.Snapshot().bytes[static_cast<std::size_t>(
        cache::ResourceCategory::kBackingFree)];
    assert(payload == (backed ? kPayloadBytes : 0));
    if (!refused) {
      assert(diagnostic.find("granted_bytes=1048576") != std::string::npos);
      assert(diagnostic.find("planned_payload_bytes=16384") !=
             std::string::npos);
      assert(diagnostic.find("planned_metadata_bytes=16384") !=
             std::string::npos);
      assert(diagnostic.find("block_bytes=4096") != std::string::npos);
      assert(diagnostic.find("blocks=4") != std::string::npos);
      assert(diagnostic.find("segment_bytes=16384") != std::string::npos);
    }
    if (backed) {
      assert(diagnostic.find("committed_bytes=16384") != std::string::npos);
      // The logical cache budget includes nonpayload overhead. Requiring it
      // all as payload would incorrectly reject this valid backing plan.
      if (full)
        assert(diagnostic.find("required_payload_bytes=16384") !=
               std::string::npos);
      const auto registered_before = registered.size();
      auto buffer = resources->allocate({cache::testing::kRecurrent,
                                         cache::ResourceCategory::kPrivateState,
                                         0, 0, 32});
      assert(registered.size() == registered_before);
      assert(buffer.bytes.size() >= 32);
    } else if (!full) {
      assert(diagnostic.find("required_payload_bytes") == std::string::npos);
    }
  }
  assert(registered.empty() && ledger.Snapshot().total_bytes == 0);
}

void OnDemandLazyBacking(bool inject_registration_failure) {
  using namespace gufo;
  constexpr std::size_t block = 4096;
  constexpr std::size_t cache_bytes = kPayloadBytes + kMetadataBytes;
  constexpr std::size_t execution_bytes = block;
  constexpr std::size_t image_bytes = 2 * block;
  // Enough for the cache OR mixed images and partial backing. The configured
  // cache maximum is optional growth, not an upfront execution obligation.
  constexpr std::size_t available = execution_bytes + kPayloadBytes;
  std::atomic<std::size_t> image_committed{};
  models::qwen::vision::AdmissionBudget admission(
      [&] {
        return available - registered_bytes.load() - image_committed.load();
      },
      execution_bytes);
  cache::ResourceLedger cache_ledger({cache_bytes, cache_bytes, 0, 0});
  fail_registration = false;
  fail_query = false;
  std::barrier start(2);
  std::latch image_live(1), release_image(1);
  std::atomic<bool> synchronize_growth{false};
  const auto refused = [&](bool shared, std::size_t bytes) {
    bool exhausted = false;
    try {
      auto extra =
          shared ? admission.ReserveShared(bytes) : admission.Reserve(bytes);
    } catch (const core::ResourceAdmissionError&) {
      exhausted = true;
    }
    assert(exhausted);
  };
  {
    hip::CommittedBackingPool pool(
        cache_ledger,
        {.ram_bytes = cache_bytes,
         .block_bytes = block,
         .metadata_headroom_bytes = kMetadataBytes,
         .segment_bytes = block,
         .admit_segment =
             [&](std::size_t bytes) {
               assert(bytes == block);
               if (synchronize_growth.exchange(false)) {
                 start.arrive_and_wait();
                 image_live.wait();
               }
               return true;
             },
         .keep_segment =
             [](std::size_t, std::size_t before, std::size_t after) {
               return hip::RegistrationKeepsReserve(before, after);
             },
         .reserve_segment =
             [&](std::size_t bytes) {
               return admission.ReserveShared(bytes);
             }});
    assert(!pool.CommittedBytes());
    assert(admission.ProtectedBytes() == execution_bytes);
    assert(!admission.Outstanding());
    pool.GrowAhead(block);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == block);
    assert(admission.Outstanding() == block);
    assert(!admission.PendingBytes());

    synchronize_growth = true;
    fail_registration = inject_registration_failure;
    std::thread image([&] {
      start.arrive_and_wait();
      auto reservation = admission.Reserve(image_bytes);
      std::vector<std::byte> pixels(image_bytes);
      image_committed = image_bytes;
      auto charge = reservation.Convert();
      image_live.count_down();
      release_image.wait();
      std::vector<std::byte>().swap(pixels);
      image_committed = 0;
      charge = {};
    });
    pool.GrowAhead(kPayloadBytes);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() ==
           (inject_registration_failure ? block : 2 * block));
    assert(cache_ledger.Snapshot().total_bytes ==
           kMetadataBytes + pool.CommittedBytes());
    assert(registered_bytes == pool.CommittedBytes());
    assert(!admission.PendingBytes());
    if (inject_registration_failure) {
      // The failed physical allocation leaked neither a pending reservation
      // nor a committed charge. One explicit recovery follows fault removal.
      fail_registration = false;
      pool.GrowAhead(kPayloadBytes);
      pool.WaitForGrowth();
      assert(pool.CommittedBytes() == 2 * block);
    }
    assert(admission.Outstanding() == 2 * block + image_bytes);
    refused(false, 1);
    refused(true, 1);
    release_image.count_down();
    image.join();
    assert(admission.Outstanding() == 2 * block);
    // Fresh observation reflects committed backing exactly once. Growth can
    // reuse released image capacity, but cannot spend the execution allowance.
    pool.GrowAhead(kPayloadBytes);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == kPayloadBytes);
    assert(admission.Outstanding() == kPayloadBytes);
    assert(!admission.PendingBytes());
    refused(false, 1);
    refused(true, 1);
    auto checkpoint =
        pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
    assert(checkpoint);
    checkpoint->Bytes().front() = std::byte{7};
    checkpoint->Convert();
  }
  assert(registered.empty() && !registered_bytes);
  assert(!cache_ledger.Snapshot().total_bytes && !admission.Outstanding());
  // A previous refusal does not poison later work or the reservation ledger.
  auto image = admission.Reserve(image_bytes);
  assert(admission.PendingBytes() == image_bytes);
}

void OnDemandHooksKeepLazyReducedGrants(bool image_pressure) {
  using namespace gufo;
  using namespace gufo::models::qwen38_flash_next;
  constexpr std::size_t execution = 4096;
  const auto admission =
      std::make_shared<models::qwen::vision::AdmissionBudget>(
          [] { return execution + kPayloadBytes - registered_bytes.load(); },
          execution);
  cache::ResourceLedger ledger(
      {kBudgetBytes, kBudgetBytes, kBudgetBytes, kBudgetBytes});
  auto image =
      image_pressure ? admission->Reserve(4096) : cache::ResourceReservation{};
  std::string warnings, info;
  {
    hip::ComponentCacheHooks hooks;
    hooks.transfer_streams = 1;
    hooks.host_ceiling = [] { return kBudgetBytes; };
    hooks.warn = [&](std::string_view message) { warnings += message; };
    hooks.info = [&](std::string_view message) { info += message; };
    hooks = VisionCacheHooks(std::move(hooks), admission, [] { return 0; });
    assert(hooks.host_ceiling() == kBudgetBytes);
    assert(!hooks.require_full_backing);
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(), ledger, std::move(hooks));
    // Shared serving may grant less than the requested maximum after slots
    // allocate. The actual production hook accepts that grant unchanged.
    ValidateVisionCacheGrant(admission, kBudgetBytes, kBudgetBytes / 2);
    ValidateVisionCacheGrant(admission, kBudgetBytes, kBudgetBytes / 2,
                             kBudgetBytes);
    resources->commit_backing(kBudgetBytes / 2);
    assert(info.find("required_payload_bytes") == std::string::npos);
    assert(registered_bytes == (image_pressure ? 0 : kPayloadBytes));
    if (image_pressure) {
      assert(warnings.find("event=backing_limited reason=shared_reservation") !=
             std::string::npos);
      assert(admission->Outstanding() == 4096);
    } else {
      assert(warnings.empty());
      assert(admission->Outstanding() == kPayloadBytes);
      assert(!admission->PendingBytes());
    }
  }
  image = {};
  assert(!registered_bytes && !admission->Outstanding());
}

void CachePolicyCompatibility() {
  using namespace gufo;
  using namespace gufo::models::qwen38_flash_next;
  hip::ComponentCacheHooks upstream;
  upstream.host_ceiling = [] { return kBudgetBytes; };
  const auto ordinary = VisionCacheHooks(upstream, {}, [] { return 0; });
  assert(ordinary.host_ceiling() == kBudgetBytes);
  assert(!ordinary.require_full_backing && !ordinary.reserve_segment);
  ValidateVisionCacheGrant({}, kBudgetBytes, kBudgetBytes / 2, kBudgetBytes);

  const auto fixed =
      std::make_shared<models::qwen::vision::AdmissionBudget>(kBudgetBytes);
  const auto legacy =
      VisionCacheHooks(upstream, fixed, [] { return kPayloadBytes; });
  assert(legacy.host_ceiling() == kPayloadBytes);
  assert(legacy.require_full_backing && !legacy.reserve_segment);
  ValidateVisionCacheGrant(fixed, kBudgetBytes, kBudgetBytes, kBudgetBytes);
  for (const auto granted : {kBudgetBytes - 1, kBudgetBytes + 1}) {
    bool refused = false;
    try {
      ValidateVisionCacheGrant(fixed, kBudgetBytes, granted, kBudgetBytes);
    } catch (const core::ResourceAdmissionError& error) {
      refused = std::string_view(error.what()).find("reason=cache_grant") !=
                std::string_view::npos;
    }
    assert(refused);
  }
}

void SegmentTicketFailureAndLifetime(int failure) {
  using namespace gufo;
  constexpr std::size_t block = 4096;
  constexpr std::size_t cache_bytes = kMetadataBytes + kPayloadBytes;
  cache::ResourceLedger cache_ledger({cache_bytes, cache_bytes, 0, 0});
  const auto shared_capacity = failure == 1 ? 0 : kPayloadBytes;
  cache::ResourceLedger shared({shared_capacity, shared_capacity, 0, 0});
  if (failure == 4)
    shared.FailAfter(cache::LedgerStep::kConvert, 0);
  fail_registration = failure == 2;
  fail_query = false;
  verify_segment_ownership = [&] {
    // Both at physical allocation and at release, the corresponding shared
    // reservation/charge must still exist. No early accounting release.
    assert(shared.Snapshot().total_bytes >= registered_bytes.load());
  };
  std::optional<hip::BackingBlock> retained;
  {
    hip::CommittedBackingPool pool(
        cache_ledger,
        {.ram_bytes = cache_bytes,
         .block_bytes = block,
         .metadata_headroom_bytes = kMetadataBytes,
         .segment_bytes = block,
         .admit_segment = {},
         .keep_segment =
             [failure](std::size_t, std::size_t, std::size_t) {
               if (failure == 5)
                 throw std::bad_alloc();
               if (failure == 6)
                 throw std::runtime_error("injected post-registration failure");
               return failure != 3;
             },
         .reserve_segment =
             [&](std::size_t bytes) {
               return shared.Reserve(cache::ResourceCategory::kPrivateTail,
                                     bytes);
             }});
    pool.GrowAhead(block);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == (failure ? 0 : block));
    assert(shared.Snapshot().total_bytes == pool.CommittedBytes());
    if (!failure)
      retained = pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
  }
  // A capture owner can outlive the pool facade. Its registered segment and
  // shared accounting must survive together until the last physical user.
  assert(shared.Snapshot().total_bytes == (failure ? 0 : block));
  assert(registered_bytes == (failure ? 0 : block));
  retained.reset();
  assert(!shared.Snapshot().total_bytes && !registered_bytes);
  verify_segment_ownership = {};
  fail_registration = false;
}

void TransferConstructionFailures() {
  using gufo::hip::TransferPool;
  assert(!live_streams && !live_events);
  // Fail each stream/event creation, including failures after complete pairs
  // and after a stream whose matching event could not be created.
  for (int failure = 0; failure < 10; ++failure) {
    native_creations = 0;
    fail_native_creation = failure;
    bool refused = false;
    try {
      TransferPool pool(5);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused && native_creations == unsigned(failure + 1));
    assert(!live_streams && !live_events);
  }
  fail_native_creation = -1;
}

void TransferPreservationOwnership() {
  using namespace gufo;
  // A small synthetic pool exercises ownership, not a proposed production
  // minimum. Four session preservation leases remain held throughout three
  // concurrent transfers, as they do during serving capture/restore work.
  native_creations = 0;
  auto pool = std::make_unique<hip::TransferPool>(7);
  assert(native_creations == 14 && live_streams == 7 && live_events == 7);
  std::array<std::unique_ptr<hip::TransferStream>, 4> preservation;
  for (auto& stream : preservation) {
    stream = pool->TryAcquire();
    assert(stream);
  }
  std::barrier acquired(4), streams_released(4);
  std::latch release_streams(1), release_completions(1);
  std::array<std::thread, 3> workers;
  for (auto& worker : workers) {
    worker = std::thread([&] {
      auto stream = pool->TryAcquire();
      assert(stream);
      std::array<std::byte, 16> source{}, destination{};
      source.front() = std::byte{42};
      auto completion =
          stream->Copy(destination.data(), source.data(), source.size(),
                       hip::CopyDirection::kDeviceToHost, 4);
      acquired.arrive_and_wait();
      release_streams.wait();
      assert(completion.Wait() == cache::TransferResult::kSucceeded);
      assert(destination == source);
      stream.reset();
      streams_released.arrive_and_wait();
      release_completions.wait();
    });
  }
  acquired.arrive_and_wait();
  assert(!pool->TryAcquire());
  release_streams.count_down();
  streams_released.arrive_and_wait();
  // Synchronizing and releasing the stream do not release a pair while its
  // completion still owns it. No hidden native creation may cover exhaustion.
  assert(!pool->TryAcquire() && native_creations == 14);
  release_completions.count_down();
  for (auto& worker : workers)
    worker.join();
  {
    std::array<std::unique_ptr<hip::TransferStream>, 3> reused;
    for (auto& stream : reused) {
      stream = pool->TryAcquire();
      assert(stream);
    }
    assert(!pool->TryAcquire() && native_creations == 14);
  }
  pool.reset();
  assert(live_streams == 7 && live_events == 7);
  for (auto& stream : preservation)
    stream.reset();
  assert(!live_streams && !live_events && native_creations == 14);
}

void TransferQuarantine() {
  using namespace gufo;
  native_creations = 0;
  hip::TransferPool pool(5);
  std::array<std::unique_ptr<hip::TransferStream>, 4> preservation;
  for (auto& stream : preservation)
    stream = pool.TryAcquire();
  auto transfer = pool.TryAcquire();
  assert(transfer && !pool.TryAcquire());
  fail_event_record = true;
  bool failed = false;
  try {
    (void)transfer->Complete();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  fail_event_record = false;
  assert(failed);
  transfer.reset();
  // A failed pair cannot be reused or silently replaced by a native allocation.
  assert(!pool.TryAcquire() && native_creations == 10);
  preservation.front().reset();
  transfer = pool.TryAcquire();
  assert(transfer);
  auto completion = transfer->Complete();
  assert(completion.Wait() == cache::TransferResult::kSucceeded);
  assert(native_creations == 10);
}

void TransferCompletionOwnership() {
  using namespace gufo;
  native_creations = 0;
  std::optional<cache::Completion> completion;
  {
    hip::TransferPool pool(1);
    auto transfer = pool.TryAcquire();
    assert(transfer);
    completion.emplace(transfer->Complete());
  }
  // Even without the facade or stream, the completion owns the native pair.
  // No copies are queued in this case; queued-work tests below cover draining.
  assert(live_streams == 1 && live_events == 1 && native_creations == 2);
  assert(completion->Wait() == cache::TransferResult::kSucceeded);
  assert(live_streams == 1 && live_events == 1);
  completion.reset();
  assert(!live_streams && !live_events && native_creations == 2);
}

void SharedTransferConstructionAndCounts() {
  using namespace gufo;
  for (int failure = 0; failure < 7; ++failure) {
    native_creations = 0;
    fail_native_creation = failure;
    bool refused = false;
    try {
      hip::TransferPool pool(5, 2);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused && native_creations == unsigned(failure + 1));
    assert(!live_streams && !live_events);
  }
  fail_native_creation = -1;
  struct Case {
    std::size_t sessions, streams, leases;
  };
  for (const auto c :
       {Case{1, 2, 19}, Case{2, 3, 37}, Case{4, 5, 73}, Case{8, 9, 145}}) {
    const auto plan = hip::PlanComponentTransfers(c.sessions);
    assert(plan.native_streams == c.streams && plan.leases == c.leases);
    native_creations = system_release_events = 0;
    {
      hip::TransferPool pool(plan.leases, plan.native_streams);
      assert(live_streams == c.streams && live_events == c.leases);
      assert(system_release_events == c.leases);
      std::vector<std::unique_ptr<hip::TransferStream>> held;
      for (std::size_t i = 0; i < plan.leases; ++i) {
        held.push_back(pool.TryAcquire());
        assert(held.back());
      }
      assert(!pool.TryAcquire());
      assert(native_creations == c.streams + c.leases);
    }
    assert(!live_streams && !live_events);
  }
  for (const auto sessions :
       {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
    bool refused = false;
    try {
      (void)hip::PlanComponentTransfers(sessions);
    } catch (const std::exception&) {
      refused = true;
    }
    assert(refused);
  }
}

void ComponentTransferFactoryCounts() {
  using namespace gufo;
  // Zero selects the existing generic provider configuration. Serving passes
  // its actual session count to the production plan before creating resources.
  for (const std::size_t sessions : {0, 1, 4, 8}) {
    cache::ResourceLedger ledger(
        {kBudgetBytes, kBudgetBytes, kBudgetBytes, kBudgetBytes});
    hip::ComponentCacheHooks hooks;
    assert(hooks.transfer_streams == 64 && hooks.transfer_leases == 0);
    auto streams = hooks.transfer_streams;
    auto leases = hooks.transfer_streams;
    if (sessions) {
      const auto plan = hip::PlanComponentTransfers(sessions);
      streams = hooks.transfer_streams = plan.native_streams;
      leases = hooks.transfer_leases = plan.leases;
    }
    hooks.host_ceiling = [] { return kBudgetBytes; };
    hooks.warn = [](std::string_view) {};
    hooks.info = [](std::string_view) {};
    native_creations = system_release_events = 0;
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(), ledger, std::move(hooks));
    // Every native handle already exists at resource construction. No cache
    // payload has committed, and borrowing cannot create a later native cost.
    assert(live_streams == streams && live_events == leases);
    assert(system_release_events == (sessions ? leases : 0));
    assert(registered.empty() && registered_bytes == 0);
    std::vector<std::unique_ptr<cache::Stream>> held;
    for (std::size_t i = 0; i < leases; ++i) {
      held.push_back(resources->stream());
      assert(held.back());
    }
    assert(!resources->stream());
    resources.reset();
    assert(live_streams == streams && live_events == leases);
    held.clear();
    assert(!live_streams && !live_events);
    assert(native_creations == streams + leases);
  }
}

void SharedTransferFences() {
  using namespace gufo;
  hip::TransferPool pool(3, 1);
  auto first = pool.TryAcquire();
  auto later = pool.TryAcquire();
  const std::array<std::byte, 4> a{std::byte{11}}, b{std::byte{22}};
  std::array<std::byte, 4> out_a{}, out_b{};
  assert(first->Native() == later->Native());
  auto done_a = first->Copy(out_a.data(), a.data(), a.size(),
                            hip::CopyDirection::kDeviceToHost, a.size());
  auto done_b = later->Copy(out_b.data(), b.data(), b.size(),
                            hip::CopyDirection::kDeviceToHost, b.size());
  assert(!done_a.Ready() && !done_b.Ready());
  const auto synchronizations = stream_synchronizations.load();
  assert(done_a.Wait() == cache::TransferResult::kSucceeded);
  // A whole native-stream synchronization would also write out_b here.
  assert(out_a == a && out_b != b && !done_b.Ready());
  assert(stream_synchronizations == synchronizations);
  first.reset();
  assert(out_b != b && !done_b.Ready());
  assert(done_b.Wait() == cache::TransferResult::kSucceeded && out_b == b);

  auto unsealed = pool.TryAcquire();
  out_a = {};
  assert(hipMemcpyAsync(out_a.data(), a.data(), a.size(), hipMemcpyDeviceToHost,
                        unsealed->Native()) == hipSuccess);
  assert(unsealed->Synchronize() == cache::TransferResult::kSucceeded);
  assert(out_a == a);
  // Manual synchronization may be followed by more Native submissions. The
  // same preallocated event must be re-recorded after the new work.
  out_a = {};
  assert(hipMemcpyAsync(out_a.data(), a.data(), a.size(), hipMemcpyDeviceToHost,
                        unsealed->Native()) == hipSuccess);
  auto final = unsealed->Complete();
  assert(!final.Ready());
  assert(final.Wait() == cache::TransferResult::kSucceeded && out_a == a);
  assert(stream_synchronizations == synchronizations);
}

void SharedCachedNativeCleanup() {
  using namespace gufo;
  hip::TransferPool pool(3, 1);
  const auto creations = native_creations.load();
  const auto synchronizations = stream_synchronizations.load();
  const std::array<std::byte, 4> source{std::byte{33}};
  std::array<std::byte, 4> first{}, later{};
  bool interrupted = false;
  try {
    auto transfer = pool.TryAcquire();
    const auto native = transfer->Native();
    assert(hipMemcpyAsync(first.data(), source.data(), source.size(),
                          hipMemcpyDeviceToHost, native) == hipSuccess);
    assert(transfer->Synchronize() == cache::TransferResult::kSucceeded);
    assert(first == source);
    // Native() need not be called again: an unsealed saved handle remains
    // usable after an explicit synchronization.
    assert(hipMemcpyAsync(later.data(), source.data(), source.size(),
                          hipMemcpyDeviceToHost, native) == hipSuccess);
    assert(transfer->Synchronize() == cache::TransferResult::kSucceeded);
    assert(later == source);
    later = {};
    assert(hipMemcpyAsync(later.data(), source.data(), source.size(),
                          hipMemcpyDeviceToHost, native) == hipSuccess);
    throw std::runtime_error("interrupt before Complete");
  } catch (const std::runtime_error&) {
    interrupted = true;
  }
  // Check before pool destruction: its whole-stream shutdown must not hide
  // missing cleanup of the transfer's last submissions.
  assert(interrupted && later == source);
  assert(stream_synchronizations == synchronizations);
  assert(native_creations == creations);
}

void SharedTransferServingBound() {
  using namespace gufo;
  const auto plan = hip::PlanComponentTransfers(4);
  native_creations = 0;
  auto pool =
      std::make_unique<hip::TransferPool>(plan.leases, plan.native_streams);
  std::array<std::unique_ptr<hip::TransferStream>, 4> preservation;
  for (auto& stream : preservation)
    stream = pool->TryAcquire();
  std::array<std::array<std::byte, 4>, 5> background_source{},
      background_copy{};
  std::array<std::unique_ptr<hip::TransferStream>, 5> background;
  std::array<std::optional<cache::Completion>, 5> background_done;
  NativeGate delayed;
  for (std::size_t i = 0; i < background.size(); ++i) {
    background_source[i][0] = std::byte(i + 1);
    background[i] = pool->TryAcquire();
    if (i == 0)
      next_copy_gate = &delayed;
    background_done[i].emplace(background[i]->Copy(
        background_copy[i].data(), background_source[i].data(), 4,
        hip::CopyDirection::kDeviceToHost, 4));
  }
  // Conservative production bound: four permanent leases, four foreground
  // batches of sixteen, plus four already submitted idle pieces and disk work.
  // Logical batch width deliberately exceeds the five physical streams.
  std::barrier admitted(5);
  std::latch start_waiting(1);
  std::array<std::thread, 4> callers;
  for (auto& caller : callers) {
    caller = std::thread([&] {
      std::array<std::array<std::byte, 4>, 16> source{}, destination{};
      std::array<std::unique_ptr<hip::TransferStream>, 16> streams;
      std::array<std::optional<cache::Completion>, 16> completions;
      for (std::size_t i = 0; i < streams.size(); ++i) {
        streams[i] = pool->TryAcquire();
        assert(streams[i]);
        source[i][0] = std::byte(i + 10);
        completions[i].emplace(
            streams[i]->Copy(destination[i].data(), source[i].data(), 4,
                             hip::CopyDirection::kDeviceToHost, 4));
      }
      admitted.arrive_and_wait();
      start_waiting.wait();
      for (std::size_t i = 0; i < completions.size(); ++i) {
        assert(completions[i]->Wait() == cache::TransferResult::kSucceeded);
        assert(source[i] == destination[i]);
      }
    });
  }
  admitted.arrive_and_wait();
  assert(!pool->TryAcquire());
  assert(native_creations == plan.native_streams + plan.leases);
  start_waiting.count_down();
  delayed.entered.wait();
  assert(background_copy[0] != background_source[0]);
  assert(live_streams == plan.native_streams && live_events == plan.leases);
  delayed.release.count_down();
  for (auto& caller : callers)
    caller.join();
  for (auto& completion : background_done)
    assert(completion->Wait() == cache::TransferResult::kSucceeded);
  assert(background_source == background_copy);
  for (auto& stream : preservation)
    stream.reset();
  for (auto& stream : background)
    stream.reset();
  pool.reset();
  assert(live_streams == plan.native_streams && live_events == plan.leases);
  for (auto& completion : background_done)
    completion.reset();
  assert(!live_streams && !live_events);
  assert(native_creations == plan.native_streams + plan.leases);
}

void SharedTransferCancellation() {
  using namespace gufo;
  auto pool = std::make_unique<hip::TransferPool>(2, 1);
  NativeGate delayed;
  std::latch submitted(1);
  std::atomic<bool> unwound{};
  std::array<std::byte, 4> source{std::byte{42}}, destination{};
  std::jthread caller([&](std::stop_token stop) {
    {
      auto stream = pool->TryAcquire();
      next_copy_gate = &delayed;
      auto completion = stream->Copy(destination.data(), source.data(), 4,
                                     hip::CopyDirection::kDeviceToHost, 4);
      submitted.count_down();
      std::mutex mutex;
      std::condition_variable_any wake;
      std::unique_lock lock(mutex);
      wake.wait(lock, stop, [] { return false; });
      // Cancellation unwinds the owner; it cannot release submitted operands
      // or make the physical stream disappear before its completion settles.
    }
    unwound = true;
  });
  submitted.wait();
  caller.request_stop();
  delayed.entered.wait();
  pool.reset();
  assert(!unwound && live_streams == 1 && live_events == 2);
  delayed.release.count_down();
  caller.join();
  assert(unwound && source == destination && !live_streams && !live_events);
}

void SharedTransferConcurrentDrain() {
  using namespace gufo;
  hip::TransferPool pool(2, 1);
  auto stream = pool.TryAcquire();
  NativeGate delayed;
  std::array<std::byte, 4> source{std::byte{7}}, destination{};
  next_copy_gate = &delayed;
  auto completion = stream->Copy(destination.data(), source.data(), 4,
                                 hip::CopyDirection::kDeviceToHost, 4);
  const auto before = event_waits.load();
  auto waiter =
      std::async(std::launch::async, [&] { return completion.Wait(); });
  delayed.entered.wait();
  std::latch other_started(1);
  auto synchronizer = std::async(std::launch::async, [&] {
    other_started.count_down();
    return stream->Synchronize();
  });
  other_started.wait();
  delayed.release.count_down();
  assert(waiter.get() == cache::TransferResult::kSucceeded);
  assert(synchronizer.get() == cache::TransferResult::kSucceeded);
  assert(source == destination && event_waits == before + 1);
}

void SharedTransferFailures() {
  using namespace gufo;
  for (unsigned failure = 0; failure < 5; ++failure) {
    native_creations = 0;
    hip::TransferPool pool(4, 2);
    auto failing = pool.TryAcquire();
    auto healthy = pool.TryAcquire();
    auto peer = pool.TryAcquire();
    auto second_healthy = pool.TryAcquire();
    assert(failing->Native() == peer->Native());
    assert(failing->Native() != healthy->Native());
    std::array<std::byte, 4> source{std::byte{9}}, destination{}, peer_copy{};
    auto peer_done = peer->Copy(peer_copy.data(), source.data(), 4,
                                hip::CopyDirection::kDeviceToHost, 4);
    bool failed = false;
    fail_event_record = failure == 0 || failure == 3;
    fail_copy_after_submission = failure == 1;
    fail_event_wait = failure == 2;
    fail_stream_wait = failure == 3;
    fail_event_query = failure == 4;
    try {
      auto done = failing->Copy(destination.data(), source.data(), 4,
                                hip::CopyDirection::kDeviceToHost, 4);
      if (failure == 4)
        assert(!done.Ready());
      failed = done.Wait() == cache::TransferResult::kFailed;
    } catch (const std::runtime_error&) {
      failed = true;
    }
    fail_event_record = false;
    fail_copy_after_submission = fail_event_wait = false;
    fail_stream_wait = fail_event_query = false;
    assert(failed && destination == source && peer_copy == source);
    assert(peer_done.Wait() == cache::TransferResult::kFailed);
    failing.reset();
    peer.reset();
    bool exhausted = false;
    try {
      (void)pool.TryAcquire();
    } catch (const std::runtime_error&) {
      exhausted = true;
    }
    // Permanent lane loss is a terminating failure, never null-and-poll.
    assert(exhausted && native_creations == 6);
    healthy.reset();
    auto reused = pool.TryAcquire();
    assert(reused && reused->Native() == second_healthy->Native());
    auto done = reused->Complete();
    assert(done.Wait() == cache::TransferResult::kSucceeded);
  }
  assert(!live_streams && !live_events);
}

void SharedBoundFailure() {
  using namespace gufo;
  hip::TransferPool pool(3, 1);
  auto bound = pool.TryAcquire();
  auto independent = pool.TryAcquire();
  hipStream_t external{};
  assert(hipStreamCreateWithFlags(&external, hipStreamNonBlocking) ==
         hipSuccess);
  auto owner = std::shared_ptr<void>(external, [](void* stream) {
    assert(hipStreamSynchronize(stream) == hipSuccess);
    assert(hipStreamDestroy(stream) == hipSuccess);
  });
  bound->BindTo(external, owner);
  owner.reset();
  std::array<std::byte, 4> source{std::byte{3}}, destination{};
  fail_event_record = true;
  bool failed = false;
  try {
    (void)bound->Copy(destination.data(), source.data(), 4,
                      hip::CopyDirection::kDeviceToHost, 4);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  fail_event_record = false;
  assert(failed && source == destination && live_streams == 2);
  bound.reset();
  assert(live_streams == 1);
  // An external execution stream's failure loses its logical slot, not the
  // pool's unrelated native lane or the other event slots mapped to it.
  auto spare = pool.TryAcquire();
  assert(spare && spare->Native() == independent->Native());
  auto done = independent->Complete();
  assert(done.Wait() == cache::TransferResult::kSucceeded);
}

void SharedBoundCompletionOwnership() {
  using namespace gufo;
  auto pool = std::make_unique<hip::TransferPool>(2, 1);
  auto bound = pool->TryAcquire();
  hipStream_t external{};
  assert(hipStreamCreateWithFlags(&external, hipStreamNonBlocking) ==
         hipSuccess);
  bool destroyed = false;
  auto owner = std::shared_ptr<void>(external, [&](void* stream) {
    assert(hipStreamSynchronize(stream) == hipSuccess);
    assert(hipStreamDestroy(stream) == hipSuccess);
    destroyed = true;
  });
  bound->BindTo(external, owner);
  std::optional<cache::Completion> completion(bound->Complete());
  owner.reset();
  bound.reset();
  pool.reset();
  assert(!destroyed && live_streams == 2 && live_events == 2);
  assert(completion->Wait() == cache::TransferResult::kSucceeded);
  assert(!destroyed);
  completion.reset();
  assert(destroyed && !live_streams && !live_events);
}

}  // namespace

hipError_t hipHostMalloc(void** pointer, std::size_t bytes, unsigned) {
  *pointer = std::malloc(bytes);
  return *pointer ? hipSuccess : hipErrorOutOfMemory;
}
hipError_t hipHostFree(void* pointer) {
  std::free(pointer);
  return hipSuccess;
}
hipError_t hipHostRegister(void* pointer, std::size_t bytes, unsigned) {
  if (fail_registration)
    return hipErrorOutOfMemory;
  registered.emplace(pointer, bytes);
  registered_bytes.fetch_add(bytes);
  if (verify_segment_ownership)
    verify_segment_ownership();
  return hipSuccess;
}
hipError_t hipHostUnregister(void* pointer) {
  if (verify_segment_ownership)
    verify_segment_ownership();
  registered_bytes.fetch_sub(registered.at(pointer));
  assert(registered.erase(pointer) == 1);
  return hipSuccess;
}
hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  *free = *total = std::size_t{8} << 30;
  return fail_query ? hipErrorOutOfMemory : hipSuccess;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned) {
  if (int(native_creations.fetch_add(1)) == fail_native_creation)
    return hipErrorOutOfMemory;
  *stream = new NativeStream;
  ++live_streams;
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  const auto* native = static_cast<NativeStream*>(stream);
  assert(native->work.empty() && !native->running);
  delete native;
  assert(live_streams.fetch_sub(1));
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t stream) {
  ++stream_synchronizations;
  auto& native = *static_cast<NativeStream*>(stream);
  std::size_t submitted;
  {
    const std::lock_guard lock(native.mutex);
    submitted = native.submitted;
  }
  RunThrough(native, submitted);
  return fail_stream_wait.exchange(false) ? hipErrorOutOfMemory : hipSuccess;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  if (int(native_creations.fetch_add(1)) == fail_native_creation)
    return hipErrorOutOfMemory;
  *event = new NativeEvent;
  if (flags & hipEventReleaseToSystem)
    ++system_release_events;
  ++live_events;
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  delete static_cast<NativeEvent*>(event);
  assert(live_events.fetch_sub(1));
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  if (fail_event_record)
    return hipErrorOutOfMemory;
  auto& native = *static_cast<NativeStream*>(stream);
  const std::lock_guard lock(native.mutex);
  *static_cast<NativeEvent*>(event) = {&native, native.submitted};
  return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t event) {
  if (fail_event_query.exchange(false))
    return hipErrorOutOfMemory;
  const auto& native = *static_cast<NativeEvent*>(event);
  if (!native.stream)
    return hipSuccess;
  const std::lock_guard lock(native.stream->mutex);
  return native.stream->completed >= native.sequence ? hipSuccess
                                                     : hipErrorNotReady;
}
hipError_t hipEventSynchronize(hipEvent_t event) {
  ++event_waits;
  if (fail_event_wait.exchange(false))
    return hipErrorOutOfMemory;
  const auto& native = *static_cast<NativeEvent*>(event);
  if (native.stream)
    RunThrough(*native.stream, native.sequence);
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind,
                          hipStream_t stream) {
  auto& native = *static_cast<NativeStream*>(stream);
  const std::lock_guard lock(native.mutex);
  auto* gate = std::exchange(next_copy_gate, nullptr);
  native.work.push_back({++native.submitted, [=] {
                           if (gate) {
                             gate->entered.count_down();
                             gate->release.wait();
                           }
                           std::memcpy(destination, source, bytes);
                         }});
  return fail_copy_after_submission.exchange(false) ? hipErrorOutOfMemory
                                                    : hipSuccess;
}
const char* hipGetErrorString(hipError_t) {
  return "simulated HIP failure";
}

int main() {
  for (const bool full : {false, true}) {
    Check(full, true, true, true);
    Check(full, false, true, true);
    Check(full, true, false, true);
    Check(full, true, true, false);
  }
  OnDemandLazyBacking(false);
  OnDemandLazyBacking(true);
  OnDemandHooksKeepLazyReducedGrants(false);
  OnDemandHooksKeepLazyReducedGrants(true);
  CachePolicyCompatibility();
  for (int failure = 0; failure <= 6; ++failure)
    SegmentTicketFailureAndLifetime(failure);
  TransferConstructionFailures();
  TransferPreservationOwnership();
  TransferQuarantine();
  TransferCompletionOwnership();
  SharedTransferConstructionAndCounts();
  ComponentTransferFactoryCounts();
  SharedTransferFences();
  SharedCachedNativeCleanup();
  SharedTransferServingBound();
  SharedTransferCancellation();
  SharedTransferConcurrentDrain();
  SharedTransferFailures();
  SharedBoundFailure();
  SharedBoundCompletionOwnership();
  assert(!live_streams && !live_events);
}
