#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <latch>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
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
int fail_native_creation{-1};
bool fail_event_record{};
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
  // Synchronous stubs establish lifetime only, not device execution behavior.
  assert(live_streams == 1 && live_events == 1 && native_creations == 2);
  assert(completion->Wait() == cache::TransferResult::kSucceeded);
  assert(live_streams == 1 && live_events == 1);
  completion.reset();
  assert(!live_streams && !live_events && native_creations == 2);
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
  *stream = new unsigned{};
  ++live_streams;
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  delete static_cast<unsigned*>(stream);
  assert(live_streams.fetch_sub(1));
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  return hipSuccess;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  if (int(native_creations.fetch_add(1)) == fail_native_creation)
    return hipErrorOutOfMemory;
  *event = new unsigned{};
  ++live_events;
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  delete static_cast<unsigned*>(event);
  assert(live_events.fetch_sub(1));
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return fail_event_record ? hipErrorOutOfMemory : hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t) {
  return hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t) {
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind, hipStream_t) {
  std::memcpy(destination, source, bytes);
  return hipSuccess;
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
  assert(!live_streams && !live_events);
}
