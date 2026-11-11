#include "tools/diagnostics/hip_resource_probe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/transfer_pool.hpp"
#include "src/ext/host_budget.hpp"

namespace gufo::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::array<std::size_t, 6> kCounts{1, 2, 4, 8, 16, 64};
constexpr std::size_t kTransferBytes = 4096;
// Same allocation for every mode/count: less than 1 MiB in total. Distinct
// slices prevent concurrent streams from racing on either buffer.
constexpr std::size_t kBufferBytes = kCounts.back() * kTransferBytes;

void Check(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

bool Milestone(std::size_t count) {
  return std::find(kCounts.begin(), kCounts.end(), count) != kCounts.end();
}

struct Options {
  std::string_view mode;
  std::size_t count{0};
};

Options Parse(int argc, const char* const* argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (i + 1 >= argc)
      throw std::invalid_argument("each option requires a value");
    const std::string_view value = argv[++i];
    if (argument == "--mode" && options.mode.empty()) {
      if (value != "streams" && value != "events" && value != "pairs" &&
          value != "pool")
        throw std::invalid_argument(
            "mode must be streams, events, pairs or pool");
      options.mode = value;
    } else if (argument == "--count" && options.count == 0) {
      for (const auto count : kCounts)
        if (value == std::to_string(count))
          options.count = count;
      if (options.count == 0)
        throw std::invalid_argument("count must be 1, 2, 4, 8, 16 or 64");
    } else {
      throw std::invalid_argument("unknown or duplicate option");
    }
  }
  if (options.mode.empty() || options.count == 0)
    throw std::invalid_argument("both --mode and --count are required");
  return options;
}

class Probe {
public:
  Probe(Options options, std::ostream& output, std::ostream& errors)
      : options_(options),
        output_(output),
        errors_(errors),
        streams_(options.count),
        events_(options.count) {
    leases_.reserve(options.count);
    completions_.reserve(options.count);
  }

  ~Probe() {
    // Error exits also drain work before freeing its operands. Never reset the
    // device: runtime caching after destruction is part of this measurement.
    (void)ReleaseResources();
    (void)ReleaseBuffers();
  }

  void Run() {
    Environment();
    Observe("before_hip", false);
    Check(hipInit(0), "hipInit");
    Check(hipSetDevice(0), "hipSetDevice");
    hip_initialized_ = true;
    Observe("initialized");

    Check(hipMalloc(&device_, kBufferBytes), "hipMalloc");
    Check(hipHostMalloc(&host_, kBufferBytes, hipHostMallocDefault),
          "hipHostMalloc");
    std::memset(host_, 0x5a, kBufferBytes);
    Observe("buffers_ready");
    // Warm allocation, transfer and synchronization on the default stream.
    // No explicit stream/event is created before measuring its first creation.
    Check(hipMemcpyAsync(device_, host_, kBufferBytes, hipMemcpyHostToDevice,
                         nullptr),
          "warm hipMemcpyAsync");
    Check(hipMemcpyAsync(host_, device_, kBufferBytes, hipMemcpyDeviceToHost,
                         nullptr),
          "warm reverse hipMemcpyAsync");
    Check(hipStreamSynchronize(nullptr), "warm hipStreamSynchronize");
    Observe("warmed");
    Runtime();

    if (options_.mode == "pool")
      RunPool();
    else
      RunRaw();

    if (!ReleaseResources())
      throw std::runtime_error("resource cleanup failed");
    Observe("resources_destroyed");
    if (!ReleaseBuffers())
      throw std::runtime_error("buffer cleanup failed");
    Observe("buffers_freed");
    Observe("final");
    output_ << "{\"record\":\"result\",\"result\":\"success\"}\n";
  }

  void Failed() {
    const bool resources_released = ReleaseResources();
    const bool buffers_released = ReleaseBuffers();
    // Preserve the original error if the observation also fails. A failed run
    // cannot establish a memory saving, even when cleanup appears successful.
    try {
      Observe("failure_cleanup", hip_initialized_);
    } catch (const std::exception& error) {
      errors_ << "cleanup observation: " << error.what() << '\n';
    }
    output_ << "{\"record\":\"result\",\"result\":\"failed\","
               "\"checked_cleanup_succeeded\":"
            << (resources_released && buffers_released && !cleanup_failed_
                    ? "true"
                    : "false")
            << "}\n";
  }

private:
  void Environment() {
    output_ << "{\"record\":\"configuration\",\"schema\":1,\"mode\":\""
            << options_.mode << "\",\"count\":" << options_.count
            << ",\"device\":0,\"host_buffer_bytes\":" << kBufferBytes
            << ",\"device_buffer_bytes\":" << kBufferBytes
            << ",\"transfer_bytes\":" << kTransferBytes << "}\n";
    // Only known resource/visibility knobs, never a full environment dump. An
    // unusual value is marked present, not copied into a diagnostic log.
    for (const char* name : {"GPU_MAX_HW_QUEUES", "HSA_MAX_QUEUES",
                             "HSA_ENABLE_SDMA", "HSA_ENABLE_PEER_SDMA",
                             "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}) {
      const char* value = std::getenv(name);
      output_ << "{\"record\":\"environment\",\"name\":\"" << name
              << "\",\"value\":";
      if (!value)
        output_ << "null";
      else if (std::string_view(value).find_first_not_of("0123456789,-") ==
               std::string_view::npos)
        output_ << '"' << value << '"';
      else
        output_ << "\"set_non_numeric\"";
      output_ << "}\n";
    }
  }

  void Runtime() {
    int runtime = 0;
    int driver = 0;
    Check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    Check(hipDriverGetVersion(&driver), "hipDriverGetVersion");
    output_ << "{\"record\":\"runtime\",\"runtime_version\":" << runtime
            << ",\"driver_version\":" << driver << "}\n";
  }

  void Observe(std::string_view phase, bool query_hip = true) {
    const auto begin = Clock::now();
    // The same host collector used for admission provides CMA/cgroup context.
    // Host then HIP are one sampling pass, not an atomic physical snapshot.
    const auto host = ext::QueryHostSnapshotMemory();
    std::size_t free = 0;
    std::size_t total = 0;
    if (query_hip)
      Check(hipMemGetInfo(&free, &total), "hipMemGetInfo");
    const auto end = Clock::now();
    output_
        << "{\"record\":\"memory\",\"phase\":\"" << phase << "\",\"mode\":\""
        << options_.mode << "\",\"count\":" << options_.count
        << ",\"live_streams\":" << live_streams_
        << ",\"live_events\":" << live_events_
        << ",\"submitted_transfers\":" << submitted_transfers_
        << ",\"elapsed_us\":"
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
               .count()
        << ",\"collection_us\":"
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()
        << ",\"hip_free_bytes\":";
    if (query_hip)
      output_ << free;
    else
      output_ << "null";
    output_ << ",\"hip_total_bytes\":";
    if (query_hip)
      output_ << total;
    else
      output_ << "null";
    output_ << ",\"host_available_bytes\":" << host.raw_available_bytes
            << ",\"cma_free_bytes\":" << host.cma_free_bytes
            << ",\"cma_excluded_bytes\":" << host.cma_excluded_bytes
            << ",\"cgroup_headroom_bytes\":";
    if (host.cgroup_headroom_bytes)
      output_ << *host.cgroup_headroom_bytes;
    else
      output_ << "null";
    output_ << ",\"host_ceiling_bytes\":" << host.ceiling_bytes
            << ",\"cgroup_limited\":"
            << (host.cgroup_limited ? "true" : "false")
            << ",\"host_from_meminfo\":"
            << (host.from_meminfo ? "true" : "false") << "}\n";
  }

  void RunRaw() {
    for (std::size_t i = 0; i < options_.count; ++i) {
      if (options_.mode != "events") {
        Check(hipStreamCreateWithFlags(&streams_[i], hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
        ++live_streams_;
        if (Milestone(i + 1))
          Observe("stream_created");
      }
      if (options_.mode != "streams") {
        Check(hipEventCreateWithFlags(&events_[i], hipEventDisableTiming),
              "hipEventCreateWithFlags");
        ++live_events_;
        if (Milestone(i + 1))
          Observe("event_created");
      }
    }
    Observe("created");
    for (std::size_t i = 0; i < options_.count; ++i) {
      auto* destination = static_cast<char*>(device_) + i * kTransferBytes;
      const auto* source = static_cast<char*>(host_) + i * kTransferBytes;
      Check(hipMemcpyAsync(destination, source, kTransferBytes,
                           hipMemcpyHostToDevice, streams_[i]),
            "hipMemcpyAsync");
      ++submitted_transfers_;
      if (Milestone(i + 1))
        Observe("transfer_submitted");
      Check(hipMemcpyAsync(static_cast<char*>(host_) + i * kTransferBytes,
                           destination, kTransferBytes, hipMemcpyDeviceToHost,
                           streams_[i]),
            "reverse hipMemcpyAsync");
      ++submitted_transfers_;
      if (Milestone(i + 1))
        Observe("reverse_transfer_submitted");
      if (events_[i]) {
        Check(hipEventRecord(events_[i], streams_[i]), "hipEventRecord");
        if (Milestone(i + 1))
          Observe("event_recorded");
      }
    }
    Observe("submitted");
    for (const auto event : events_)
      if (event)
        Check(hipEventSynchronize(event), "hipEventSynchronize");
    for (const auto stream : streams_)
      if (stream)
        Check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    if (options_.mode == "events")
      Check(hipStreamSynchronize(nullptr), "hipStreamSynchronize default");
    Observe("synchronized");
  }

  void RunPool() {
    pool_ = std::make_unique<hip::TransferPool>(options_.count);
    live_streams_ = live_events_ = options_.count;
    Observe("created");
    for (std::size_t i = 0; i < options_.count; ++i) {
      auto lease = pool_->TryAcquire();
      if (!lease)
        throw std::runtime_error("unexpected pool exhaustion");
      leases_.push_back(std::move(lease));
    }
    Observe("leases_acquired");
    for (std::size_t i = 0; i < options_.count; ++i) {
      auto* destination = static_cast<char*>(device_) + i * kTransferBytes;
      const auto* source = static_cast<char*>(host_) + i * kTransferBytes;
      Check(hipMemcpyAsync(destination, source, kTransferBytes,
                           hipMemcpyHostToDevice, leases_[i]->Native()),
            "pool native hipMemcpyAsync");
      ++submitted_transfers_;
      if (Milestone(i + 1))
        Observe("transfer_submitted");
      completions_.push_back(leases_[i]->Copy(
          static_cast<char*>(host_) + i * kTransferBytes, destination,
          kTransferBytes, hip::CopyDirection::kDeviceToHost, kTransferBytes));
      ++submitted_transfers_;
      if (Milestone(i + 1))
        Observe("copy_submitted");
    }
    Observe("submitted");
    for (auto& completion : completions_)
      if (completion.Wait() != cache::TransferResult::kSucceeded)
        throw std::runtime_error("pool completion failed");
    Observe("synchronized");
    // Use the actual completion ownership graph: destroying the facade or the
    // leases must not be mistaken for native resource reclamation.
    pool_.reset();
    Observe("pool_facade_destroyed");
    leases_.clear();
    Observe("pool_leases_released");
    completions_.clear();
    live_streams_ = live_events_ = 0;
  }

  bool Cleanup(hipError_t status, const char* operation) {
    if (status == hipSuccess)
      return true;
    cleanup_failed_ = true;
    errors_ << operation << ": " << hipGetErrorString(status) << '\n';
    return false;
  }

  bool ReleaseResources() {
    if (resources_released_)
      return !cleanup_failed_;
    resources_released_ = true;
    bool success = true;
    for (auto& completion : completions_)
      if (completion.Wait() != cache::TransferResult::kSucceeded)
        success = false;
    for (auto& lease : leases_)
      if (lease->Synchronize() != cache::TransferResult::kSucceeded)
        success = false;
    completions_.clear();
    leases_.clear();
    pool_.reset();
    for (const auto stream : streams_)
      if (stream)
        success = Cleanup(hipStreamSynchronize(stream),
                          "cleanup hipStreamSynchronize") &&
                  success;
    // Also settle a partially submitted warm-up copy on an error path.
    if (hip_initialized_)
      success =
          Cleanup(hipStreamSynchronize(nullptr), "cleanup default stream") &&
          success;
    for (auto& event : events_) {
      if (event)
        success = Cleanup(hipEventDestroy(event), "hipEventDestroy") && success;
      event = nullptr;
    }
    for (auto& stream : streams_) {
      if (stream)
        success =
            Cleanup(hipStreamDestroy(stream), "hipStreamDestroy") && success;
      stream = nullptr;
    }
    live_streams_ = live_events_ = 0;
    cleanup_failed_ = cleanup_failed_ || !success;
    return success;
  }

  bool ReleaseBuffers() {
    bool success = true;
    if (device_)
      success = Cleanup(hipFree(device_), "hipFree") && success;
    device_ = nullptr;
    if (host_)
      success = Cleanup(hipHostFree(host_), "hipHostFree") && success;
    host_ = nullptr;
    return success;
  }

  Options options_;
  std::ostream& output_;
  std::ostream& errors_;
  Clock::time_point start_{Clock::now()};
  bool hip_initialized_{false};
  bool resources_released_{false};
  bool cleanup_failed_{false};
  void* device_{nullptr};
  void* host_{nullptr};
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> events_;
  std::size_t live_streams_{0};
  std::size_t live_events_{0};
  std::size_t submitted_transfers_{0};
  std::unique_ptr<hip::TransferPool> pool_;
  std::vector<std::unique_ptr<hip::TransferStream>> leases_;
  std::vector<cache::Completion> completions_;
};

}  // namespace

int RunHipResourceProbe(int argc, const char* const* argv, std::ostream& output,
                        std::ostream& errors) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    output
        << "Usage: gufo-hip-resource-probe --mode streams|events|pairs|pool "
           "--count 1|2|4|8|16|64\n"
           "Run each mode/count in a fresh process with the tested HIP runtime "
           "and unchanged queue environment. No models or caches are read.\n";
    return 0;
  }
  try {
    Probe probe(Parse(argc, argv), output, errors);
    try {
      probe.Run();
      return 0;
    } catch (const std::exception& error) {
      errors << error.what() << '\n';
      probe.Failed();
      return 1;
    }
  } catch (const std::exception& error) {
    errors << error.what() << '\n';
    return 2;
  }
}

}  // namespace gufo::diagnostics

#ifndef GUFO_HIP_RESOURCE_PROBE_TEST
int main(int argc, char** argv) {
  return gufo::diagnostics::RunHipResourceProbe(argc, argv, std::cout,
                                                std::cerr);
}
#endif
