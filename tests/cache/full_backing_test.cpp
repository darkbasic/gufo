#include <hip/hip_runtime.h>

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/component_cache_resources.hpp"
#include "src/core/resource_error.hpp"
#include "src/families/qwen/vision/admission.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace {
std::map<void*, std::size_t> registered;
bool fail_registration{}, fail_query{};
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
  constexpr std::size_t text_bytes = block;
  constexpr std::size_t scratch_bytes = block;
  constexpr std::size_t margin_bytes = block;
  constexpr std::size_t image_bytes = 2 * block;
  constexpr std::size_t protected_bytes =
      cache_bytes + text_bytes + scratch_bytes + margin_bytes;
  constexpr std::size_t capacity = protected_bytes + image_bytes;
  models::qwen::vision::AdmissionBudget admission(capacity, protected_bytes);
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
                 assert(admission.AccountedBytes() == capacity);
               }
               // Physical growth spends the cache allowance already protected
               // in the shared epoch. It never competes for the image
               // remainder.
               return true;
             },
         .keep_segment =
             [](std::size_t, std::size_t before, std::size_t after) {
               return hip::RegistrationKeepsReserve(before, after);
             }});
    assert(!pool.CommittedBytes());
    assert(admission.AccountedBytes() == protected_bytes);
    assert(!admission.Outstanding());
    pool.GrowAhead(block);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == block);
    assert(cache_ledger.Snapshot().total_bytes == kMetadataBytes + block);
    // Most payload is uncommitted, but its full configured allowance is not
    // available for images. The fake HIP reading stays optimistically high.
    refused(false, image_bytes + 1);
    assert(admission.AccountedBytes() == protected_bytes);

    synchronize_growth = true;
    fail_registration = inject_registration_failure;
    std::thread image([&] {
      start.arrive_and_wait();
      auto reservation = admission.Reserve(image_bytes);
      std::vector<std::byte> pixels(image_bytes);
      image_live.count_down();
      release_image.wait();
      // Pixels are destroyed before the reservation leaves this scope.
    });
    pool.GrowAhead(kPayloadBytes);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() ==
           (inject_registration_failure ? block : kPayloadBytes));
    assert(cache_ledger.Snapshot().total_bytes ==
           kMetadataBytes + pool.CommittedBytes());
    assert(registered.size() == pool.CommittedBytes() / block);
    assert(admission.AccountedBytes() == capacity);
    assert(admission.Outstanding() == image_bytes);
    refused(false, 1);
    refused(true, 1);
    if (inject_registration_failure) {
      // One explicit recovery after removing the injected fault, not a loop
      // that waits for a favorable capacity observation or hides a failure.
      fail_registration = false;
      pool.GrowAhead(kPayloadBytes);
      pool.WaitForGrowth();
      assert(pool.CommittedBytes() == kPayloadBytes);
      assert(cache_ledger.Snapshot().total_bytes == cache_bytes);
      assert(admission.AccountedBytes() == capacity);
      refused(false, 1);
    }
    release_image.count_down();
    image.join();
    assert(!admission.Outstanding());
    assert(admission.AccountedBytes() == protected_bytes);
    assert(pool.CommittedBytes() == kPayloadBytes);
    // Text/scratch use their protected allowance while cached backing stays
    // physically committed. Another image can reuse only the same remainder.
    std::vector<std::byte> text(text_bytes), scratch(scratch_bytes);
    auto checkpoint =
        pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
    assert(checkpoint);
    checkpoint->Bytes().front() = std::byte{7};
    checkpoint->Convert();
    {
      auto reservation = admission.Reserve(image_bytes);
      std::vector<std::byte> pixels(image_bytes);
      assert(admission.AccountedBytes() == capacity);
      refused(false, 1);
    }
    assert(!admission.Outstanding());
    assert(cache_ledger.Snapshot().total_bytes == cache_bytes);
  }
  assert(registered.empty() && !cache_ledger.Snapshot().total_bytes);
  assert(admission.AccountedBytes() == protected_bytes);
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
  return hipSuccess;
}
hipError_t hipHostUnregister(void* pointer) {
  assert(registered.erase(pointer) == 1);
  return hipSuccess;
}
hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  *free = *total = std::size_t{8} << 30;
  return fail_query ? hipErrorOutOfMemory : hipSuccess;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned) {
  *stream = new unsigned{};
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  delete static_cast<unsigned*>(stream);
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  return hipSuccess;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  return hipStreamCreateWithFlags(event, flags);
}
hipError_t hipEventDestroy(hipEvent_t event) {
  return hipStreamDestroy(event);
}
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return hipSuccess;
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
}
