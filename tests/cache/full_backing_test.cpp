#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/component_cache_resources.hpp"
#include "src/core/resource_error.hpp"
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
}
