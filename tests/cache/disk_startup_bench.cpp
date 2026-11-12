#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>

#include "src/cache/disk.hpp"

// Requested ordinary C++ heap bytes, not allocator usable bytes, RSS or the
// ledger's logical charges. The optional validation hooks instrument an exact
// production function scope in matched profiling builds without a public API.
namespace {
struct Heap {
  std::atomic<bool> enabled{};
  std::atomic<std::size_t> live{}, peak{}, allocations{};
};
std::array<Heap, 2> heaps;
struct alignas(std::max_align_t) Allocation {
  std::size_t bytes;
  unsigned scopes;
};
using Clock = std::chrono::steady_clock;
Clock::time_point validation_begin;
double validation_ms{};
bool validation_measured{};
}  // namespace
[[gnu::noinline]] void* operator new(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(Allocation))
    throw std::bad_alloc();
  auto* allocation = static_cast<Allocation*>(
      std::malloc(sizeof(Allocation) + (bytes ? bytes : 1)));
  if (!allocation)
    throw std::bad_alloc();
  allocation->bytes = bytes;
  allocation->scopes = 0;
  for (unsigned i = 0; i < heaps.size(); ++i) {
    auto& heap = heaps[i];
    if (!heap.enabled)
      continue;
    allocation->scopes |= 1U << i;
    ++heap.allocations;
    auto live = heap.live.fetch_add(bytes) + bytes;
    auto peak = heap.peak.load();
    while (peak < live && !heap.peak.compare_exchange_weak(peak, live)) {
    }
  }
  return allocation + 1;
}
[[gnu::noinline]] void operator delete(void* memory) noexcept {
  if (!memory)
    return;
  auto* allocation = static_cast<Allocation*>(memory) - 1;
  for (unsigned i = 0; i < heaps.size(); ++i)
    if (allocation->scopes & (1U << i))
      heaps[i].live -= allocation->bytes;
  std::free(allocation);
}
[[gnu::noinline]] void operator delete(void* memory, std::size_t) noexcept {
  ::operator delete(memory);
}
[[gnu::noinline]] void* operator new[](std::size_t bytes) {
  return ::operator new(bytes);
}
[[gnu::noinline]] void operator delete[](void* memory) noexcept {
  ::operator delete(memory);
}
[[gnu::noinline]] void operator delete[](void* memory, std::size_t) noexcept {
  ::operator delete(memory);
}
extern "C" void GufoDiskValidationBegin() {
  heaps[1].enabled = true;
  validation_begin = Clock::now();
}
extern "C" void GufoDiskValidationEnd() {
  validation_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - validation_begin)
          .count();
  heaps[1].enabled = false;
  validation_measured = true;
}

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
DiskFileId Id(std::uint64_t value) {
  DiskFileId id{};
  for (unsigned i = 0; i < 8; ++i)
    id[i] = (value >> (8 * i)) & 255;
  return id;
}
void Write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file.good())
    throw std::runtime_error("fixture write failed");
}
std::size_t Number(const char* argument, std::size_t maximum) {
  std::size_t end{};
  const auto value = std::stoull(argument, &end);
  if (argument[end] || value > maximum)
    throw std::invalid_argument("benchmark count out of range");
  return value;
}
}  // namespace
int main(int argc, char** argv) try {
  if (argc != 1 && argc != 5) {
    std::cerr << "usage: cache_disk_startup_bench [manifests chunks private "
                 "shared_chunks]\n";
    return 2;
  }
  const auto count = argc == 5 ? Number(argv[1], 1000) : 8;
  const auto chunks = argc == 5 ? Number(argv[2], 60000) : 64;
  const auto private_count = argc == 5 ? Number(argv[3], 255) : 4;
  const auto shared = argc == 5 ? Number(argv[4], chunks) : 32;
  if (!count || chunks + private_count > 65536)
    throw std::invalid_argument("invalid benchmark geometry");
  // Bound aggregate work as well as each argument. The individual limits make
  // these products fit size_t on every supported platform. A large selectable
  // fixture may approximate file counts, never allocate arbitrary disk space.
  if (count * (chunks + private_count) > 8000000 ||
      shared + count * (chunks - shared + private_count) > 200000)
    throw std::invalid_argument("benchmark aggregate limit exceeded");
  ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
  char name[] = "/tmp/gufo-disk-startup-bench-XXXXXX";
  const auto* temporary = mkdtemp(name);
  if (!temporary)
    throw std::runtime_error("cannot create benchmark directory");
  const fs::path root(temporary);
  struct Cleanup {
    const fs::path& root;
    ~Cleanup() { fs::remove_all(root); }
  } cleanup{root};
  {
    DiskStore initialize(ledger, root, UINT64_MAX);
  }
  std::size_t expected_bytes{}, expected_references{};
  for (std::size_t i = 0; i < count; ++i) {
    DiskManifest manifest;
    manifest.checkpoint = {i + 1};
    manifest.lineage = {1};
    manifest.compatibility.fill(0x42);
    manifest.tokens.assign(chunks, 17);
    DiskComponent rows{{{1}, 1, ComponentKind::kAppendRows, 262144, 1, 0},
                       {{1}, chunks},
                       {},
                       {},
                       {}};
    for (std::size_t c = 0; c < chunks; ++c) {
      const auto id = Id(c < shared ? c + 1 : i * chunks + c + 1);
      rows.chunks.push_back({id, 262144, 1});
      const auto path = root / "v2/chunks" / DiskFileName(id);
      if (i == 0 || c >= shared) {
        Write(path, {});
        fs::resize_file(path, 262144);
      }
    }
    manifest.components.push_back(std::move(rows));
    for (std::size_t p = 0; p < private_count; ++p) {
      const auto id = Id(i * private_count + p + 1);
      manifest.components.push_back(
          {{{static_cast<std::uint32_t>(p + 2)},
            1,
            ComponentKind::kPrivateState,
            0,
            0,
            1048576},
           {{static_cast<std::uint32_t>(p + 2)}, chunks},
           {},
           {},
           DiskPayload{id, 1048576, 1}});
      const auto path = root / "v2/private" / DiskFileName(id);
      Write(path, {});
      fs::resize_file(path, 1048576);
    }
    const auto bytes = EncodeManifest(manifest);
    expected_bytes += bytes.size();
    expected_references += chunks + private_count;
    Write(root / "v2/manifests" / DiskFileName(Id(i + 1)), bytes);
  }
  (void)ledger.SnapshotAndResetPeaks();
  heaps[0].enabled = true;
  const auto begin = Clock::now();
  {
    DiskStore store(ledger, root, UINT64_MAX);
    const auto elapsed =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    heaps[0].enabled = false;
    assert(store.Entries().size() == count);
    assert(store.Stats().manifest_bytes_read == expected_bytes);
    assert(store.Stats().dependency_stats == expected_references);
    const auto snapshot = ledger.Snapshot();
    std::cout << "manifests=" << count << " references=" << expected_references
              << " unique_chunks=" << shared + count * (chunks - shared)
              << " private_files=" << count * private_count
              << " manifest_bytes_read=" << store.Stats().manifest_bytes_read
              << " startup_ms=" << elapsed
              << " logical_live_bytes=" << snapshot.ram_bytes
              << " logical_peak_bytes=" << snapshot.peak_ram_bytes
              << " cpp_heap_live_bytes=" << heaps[0].live.load()
              << " cpp_heap_peak_bytes=" << heaps[0].peak.load()
              << " cpp_allocations=" << heaps[0].allocations.load() << '\n';
    if (validation_measured)
      std::cout << "validation_ms=" << validation_ms
                << " validation_cpp_live_bytes=" << heaps[1].live.load()
                << " validation_cpp_peak_bytes=" << heaps[1].peak.load()
                << " validation_cpp_allocations=" << heaps[1].allocations.load()
                << '\n';
  }
  assert(!ledger.Snapshot().total_bytes && !heaps[0].live && !heaps[1].live);
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
