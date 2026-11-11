#include "tools/diagnostics/hip_resource_probe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/ext/host_budget.hpp"

namespace {

struct Stream {
  bool pending{false};
};
struct Event {
  Stream* stream{nullptr};
};
struct State {
  std::map<std::string, std::size_t> calls;
  std::string fail_operation;
  std::size_t fail_call{0};
  std::vector<Stream*> streams;
  std::vector<Event*> events;
  Stream default_stream;
  void* device{nullptr};
  void* host{nullptr};
  bool initialized{false};
  bool warmed{false};
  bool host_observed{false};
  std::size_t host_observations{0};
  std::size_t maximum_allocation{0};
  std::size_t total_allocations{0};
  std::size_t transfers{0};
} state;

bool Fail(const char* operation) {
  const auto call = ++state.calls[operation];
  return operation == state.fail_operation && call == state.fail_call;
}

Stream* Native(hipStream_t stream) {
  return stream ? static_cast<Stream*>(stream) : &state.default_stream;
}

void Settled() {
  assert(!state.default_stream.pending);
  for (const auto* stream : state.streams)
    assert(!stream->pending);
}

void Reset(std::string failure = {}, std::size_t call = 0) {
  assert(state.streams.empty() && state.events.empty());
  assert(!state.host && !state.device);
  state = {};
  state.fail_operation = std::move(failure);
  state.fail_call = call;
}

struct Result {
  int code;
  std::string output;
  std::string errors;
};

Result Run(const std::vector<const char*>& arguments) {
  std::ostringstream output;
  std::ostringstream errors;
  const auto code = gufo::diagnostics::RunHipResourceProbe(
      static_cast<int>(arguments.size()), arguments.data(), output, errors);
  // The real probe's error paths must settle operands and release every
  // successfully created fake handle, including the real pool's shared state.
  Settled();
  assert(state.streams.empty() && state.events.empty());
  assert(!state.host && !state.device);
  assert(state.maximum_allocation <= 262144);
  assert(state.total_allocations <= 524288);
  return {code, output.str(), errors.str()};
}

std::vector<gufo::json::Value> Records(const Result& result) {
  std::vector<gufo::json::Value> records;
  std::istringstream lines(result.output);
  for (std::string line; std::getline(lines, line);)
    records.push_back(gufo::json::parse(line));
  assert(!records.empty());
  return records;
}

const gufo::json::Value& Phase(const std::vector<gufo::json::Value>& records,
                               std::string_view phase) {
  const auto found =
      std::find_if(records.begin(), records.end(), [phase](const auto& record) {
        return record.member_str("phase") == phase;
      });
  assert(found != records.end());
  return *found;
}

const gufo::json::Value& Field(const gufo::json::Value& record,
                               const char* name) {
  const auto* field = record.find(name);
  assert(field);
  return *field;
}

void SuccessMatrix() {
  for (const char* mode : {"streams", "events", "pairs", "pool"}) {
    for (const char* count : {"1", "2", "4", "8", "16", "64"}) {
      Reset();
      const auto result = Run({"probe", "--mode", mode, "--count", count});
      assert(result.code == 0 && result.errors.empty());
      const auto records = Records(result);
      const auto expected = static_cast<std::size_t>(std::stoul(count));
      assert(Field(Phase(records, "before_hip"), "hip_free_bytes").is_null());
      assert(Field(Phase(records, "warmed"), "live_streams").as_size() == 0);
      assert(Field(Phase(records, "warmed"), "live_events").as_size() == 0);
      assert(
          Field(Phase(records, "created"), "submitted_transfers").as_size() ==
          0);
      assert(
          Field(Phase(records, "submitted"), "submitted_transfers").as_size() ==
          2 * expected);
      assert(Field(Phase(records, "resources_destroyed"), "live_streams")
                 .as_size() == 0);
      assert(Field(Phase(records, "resources_destroyed"), "live_events")
                 .as_size() == 0);
      assert(Field(records.back(), "result").str() == "success");
      assert(state.transfers == 2 * expected + 2);  // Includes warm-up.
      assert(state.calls["stream_create"] ==
             (std::string_view(mode) == "events" ? 0 : expected));
      assert(state.calls["event_create"] ==
             (std::string_view(mode) == "streams" ? 0 : expected));
      if (std::string_view(mode) == "pool") {
        assert(Field(Phase(records, "pool_facade_destroyed"), "live_streams")
                   .as_size() == expected);
        assert(Field(Phase(records, "pool_leases_released"), "live_events")
                   .as_size() == expected);
      } else {
        // First-use records include a counter even when all resources were
        // created already, so first/subsequent costs can be distinguished.
        assert(
            Field(Phase(records, "transfer_submitted"), "submitted_transfers")
                .as_size() == 1);
        assert(Field(Phase(records, "reverse_transfer_submitted"),
                     "submitted_transfers")
                   .as_size() == 2);
      }
    }
  }
}

void Failures() {
  for (const char* mode : {"streams", "events", "pairs", "pool"}) {
    std::vector<std::pair<const char*, std::size_t>> failures{
        {"init", 1},        {"device", 1},   {"malloc", 1}, {"host_malloc", 1},
        {"copy", 1},        {"copy", 2},     {"copy", 3},   {"copy", 4},
        {"stream_sync", 1}, {"memory", 2},   {"memory", 5}, {"host_memory", 6},
        {"free", 1},        {"host_free", 1}};
    if (std::string_view(mode) != "events")
      failures.push_back({"stream_create", 2});
    if (std::string_view(mode) != "streams") {
      failures.push_back({"event_create", 2});
      failures.push_back({"event_record", 1});
      failures.push_back({"event_sync", 1});
    }
    // The production pool destructor intentionally does not expose destroy
    // status. Raw modes can check these calls; no fake test invents a stronger
    // destruction guarantee for the production pool API.
    if (std::string_view(mode) != "pool") {
      if (std::string_view(mode) != "events")
        failures.push_back({"stream_destroy", 1});
      if (std::string_view(mode) != "streams")
        failures.push_back({"event_destroy", 1});
    }
    for (const auto& [operation, call] : failures) {
      Reset(operation, call);
      const auto result = Run({"probe", "--mode", mode, "--count", "4"});
      assert(result.code == 1 && !result.errors.empty());
      assert(state.calls[operation] >= call);
      const auto records = Records(result);
      assert(Field(records.back(), "result").str() == "failed");
      if (std::string_view(operation) == "free" ||
          std::string_view(operation) == "host_free" ||
          std::string_view(operation) == "stream_destroy" ||
          std::string_view(operation) == "event_destroy")
        assert(!Field(records.back(), "checked_cleanup_succeeded").as_bool());
    }
  }
}

void Arguments() {
  for (const auto& arguments : std::vector<std::vector<const char*>>{
           {"probe"},
           {"probe", "--mode", "pairs"},
           {"probe", "--mode", "invalid", "--count", "4"},
           {"probe", "--mode", "pairs", "--count", "0"},
           {"probe", "--mode", "pairs", "--count", "65"},
           {"probe", "--mode", "pairs", "--count", "4", "--count", "8"}}) {
    Reset();
    assert(Run(arguments).code == 2);
    assert(state.calls.empty() && state.host_observations == 0);
  }
  Reset();
  assert(Run({"probe", "--help"}).code == 0);
  assert(state.calls.empty() && state.host_observations == 0);
}

}  // namespace

hipError_t hipInit(unsigned flags) {
  assert(flags == 0);
  if (Fail("init"))
    return hipErrorInvalidValue;
  state.initialized = true;
  return hipSuccess;
}
hipError_t hipSetDevice(int device) {
  assert(state.initialized && device == 0);
  return Fail("device") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipRuntimeGetVersion(int* version) {
  *version = 70000;
  return hipSuccess;
}
hipError_t hipDriverGetVersion(int* version) {
  *version = 70000;
  return hipSuccess;
}
hipError_t hipMalloc(void** pointer, std::size_t bytes) {
  if (Fail("malloc"))
    return hipErrorOutOfMemory;
  *pointer = state.device = std::malloc(bytes);
  assert(*pointer);
  state.maximum_allocation = std::max(state.maximum_allocation, bytes);
  state.total_allocations += bytes;
  return hipSuccess;
}
hipError_t hipHostMalloc(void** pointer, std::size_t bytes, unsigned flags) {
  assert(flags == hipHostMallocDefault);
  if (Fail("host_malloc"))
    return hipErrorOutOfMemory;
  *pointer = state.host = std::malloc(bytes);
  assert(*pointer);
  state.maximum_allocation = std::max(state.maximum_allocation, bytes);
  state.total_allocations += bytes;
  return hipSuccess;
}
hipError_t hipFree(void* pointer) {
  Settled();
  assert(pointer == state.device);
  std::free(pointer);
  state.device = nullptr;
  return Fail("free") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipHostFree(void* pointer) {
  Settled();
  assert(pointer == state.host);
  std::free(pointer);
  state.host = nullptr;
  return Fail("host_free") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  assert(state.initialized && state.host_observed);
  state.host_observed = false;
  if (Fail("memory"))
    return hipErrorInvalidValue;
  // Logical observations exercise reporting only, never a GPU footprint model.
  *free = 8192;
  *total = 16384;
  return hipSuccess;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* result, unsigned flags) {
  assert(state.warmed && flags == hipStreamNonBlocking);
  if (Fail("stream_create"))
    return hipErrorOutOfMemory;
  auto* stream = new Stream;
  state.streams.push_back(stream);
  *result = stream;
  return hipSuccess;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* result, unsigned flags) {
  assert(state.warmed && flags == hipEventDisableTiming);
  if (Fail("event_create"))
    return hipErrorOutOfMemory;
  auto* event = new Event;
  state.events.push_back(event);
  *result = event;
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t handle) {
  auto* stream = Native(handle);
  assert(!stream->pending);
  assert(std::erase(state.streams, stream) == 1);
  delete stream;
  return Fail("stream_destroy") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t handle) {
  auto* event = static_cast<Event*>(handle);
  assert(std::erase(state.events, event) == 1);
  delete event;
  return Fail("event_destroy") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t handle) {
  if (Fail("stream_sync"))
    return hipErrorInvalidValue;
  Native(handle)->pending = false;
  if (!handle && state.calls["copy"] >= 2)
    state.warmed = true;
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind direction,
                          hipStream_t handle) {
  assert(bytes <= 262144);
  assert(direction == hipMemcpyHostToDevice ||
         direction == hipMemcpyDeviceToHost);
  // Failure can occur after work was submitted. The caller still owes a drain.
  Native(handle)->pending = true;
  if (Fail("copy"))
    return hipErrorInvalidValue;
  std::memcpy(destination, source, bytes);
  ++state.transfers;
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t handle, hipStream_t stream) {
  static_cast<Event*>(handle)->stream = Native(stream);
  return Fail("event_record") ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t handle) {
  if (Fail("event_sync"))
    return hipErrorInvalidValue;
  if (const auto stream = static_cast<Event*>(handle)->stream)
    stream->pending = false;
  return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t handle) {
  const auto stream = static_cast<Event*>(handle)->stream;
  return stream && stream->pending ? hipErrorNotReady : hipSuccess;
}
const char* hipGetErrorString(hipError_t) {
  return "injected failure";
}

namespace gufo::ext {
HostSnapshotMemory QueryHostSnapshotMemory() {
  if (Fail("host_memory"))
    throw std::runtime_error("injected host observation failure");
  state.host_observed = true;
  ++state.host_observations;
  HostSnapshotMemory snapshot;
  snapshot.raw_available_bytes = 12288;
  snapshot.ceiling_bytes = 4096;
  snapshot.from_meminfo = true;
  return snapshot;
}
}  // namespace gufo::ext

int main() {
  Arguments();
  SuccessMatrix();
  Failures();
  std::cout << "hip resource probe: 24 mode/count cases, failure paths and CLI "
               "checks passed (fake HIP; no GPU footprint claim)\n";
}
