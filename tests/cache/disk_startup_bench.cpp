#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr unsigned kLegacyValidation = 5;
std::array<Heap, 6> heaps;
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
  heaps[kLegacyValidation].enabled = true;
  validation_begin = Clock::now();
}
extern "C" void GufoDiskValidationEnd() {
  validation_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - validation_begin)
          .count();
  heaps[kLegacyValidation].enabled = false;
  validation_measured = true;
}

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
// /proc observations are optional on non-Linux POSIX hosts. They are process
// and system observations, not ResourceLedger measurements. The reads
// themselves use libc buffers and add a small amount to process I/O/RSS;
// io_rchar/io_syscr include these sampler reads. RSS/high-water values are
// approximate kernel observations, sampled separately rather than atomically.
// No cache flushing, allocator trimming or payload reads are performed.
constexpr std::uint64_t kUnknown = UINT64_MAX;
constexpr std::array<const char*, 12> kObservationNames{
    "rss_bytes",
    "rss_high_water_bytes",
    "host_available_bytes",
    "host_cached_bytes",
    "host_slab_bytes",
    "host_reclaimable_slab_bytes",
    "io_rchar",
    "io_syscr",
    "io_read_bytes",
    "io_wchar",
    "io_syscw",
    "io_write_bytes"};
struct Observation {
  std::array<std::uint64_t, kObservationNames.size()> values;
  Observation() { values.fill(kUnknown); }
};
void ProcValues(const char* path, std::span<const char* const> keys,
                std::span<std::uint64_t> values, std::uint64_t scale) noexcept {
  auto* file = std::fopen(path, "r");
  if (!file)
    return;
  char line[512], key[128];
  unsigned long long value{};
  while (std::fgets(line, sizeof(line), file)) {
    if (std::sscanf(line, "%127[^:]: %llu", key, &value) != 2)
      continue;
    for (std::size_t i = 0; i < keys.size(); ++i)
      if (std::strcmp(key, keys[i]) == 0 && value <= UINT64_MAX / scale)
        values[i] = value * scale;
  }
  std::fclose(file);
}
Observation Observe() noexcept {
  Observation result;
  constexpr std::array status{"VmRSS", "VmHWM"};
  constexpr std::array memory{"MemAvailable", "Cached", "Slab", "SReclaimable"};
  constexpr std::array io{"rchar", "syscr", "read_bytes",
                          "wchar", "syscw", "write_bytes"};
  auto values = std::span(result.values);
  ProcValues("/proc/self/status", status, values.subspan(0, 2), 1024);
  ProcValues("/proc/meminfo", memory, values.subspan(2, 4), 1024);
  ProcValues("/proc/self/io", io, values.subspan(6, 6), 1);
  return result;
}
void PrintObservation(const char* name, const Observation& observation) {
  std::cout << "observation=" << name;
  for (std::size_t i = 0; i < observation.values.size(); ++i) {
    std::cout << ' ' << kObservationNames[i] << '=';
    if (observation.values[i] == kUnknown)
      std::cout << "unknown";
    else
      std::cout << observation.values[i];
  }
  std::cout << '\n';
}
constexpr std::array<const char*, 4> kPhaseNames{
    "directory_scan", "manifest_read_decode", "dependency_checks",
    "reference_validation"};
struct Phase {
  Clock::duration elapsed{};
  std::size_t calls{};
  Observation before, after;
};
std::array<Phase, kPhaseNames.size()> phases;
std::array<unsigned, 16> phase_stack;
unsigned phase_depth{};
Clock::time_point phase_begin;
// Times and allocation counts are exclusive of nested phases. Time excludes
// Observe(), while whole-constructor elapsed time includes sampling overhead.
// First entry and every exit are sampled; only the final exit is retained.
void StartupPhase(const char* name, bool entering) noexcept {
  const auto now = Clock::now();
  if (phase_depth) {
    const auto active = phase_stack[phase_depth - 1];
    phases[active].elapsed += now - phase_begin;
    heaps[active + 1].enabled = false;
  }
  unsigned index{};
  while (index != kPhaseNames.size() &&
         std::strcmp(name, kPhaseNames[index]) != 0)
    ++index;
  assert(index != kPhaseNames.size());
  auto& phase = phases[index];
  if (entering) {
    assert(phase_depth < phase_stack.size());
    if (!phase.calls++)
      phase.before = Observe();
    phase_stack[phase_depth++] = index;
  } else {
    assert(phase_depth && phase_stack[phase_depth - 1] == index);
    --phase_depth;
    phase.after = Observe();
  }
  if (phase_depth)
    heaps[phase_stack[phase_depth - 1] + 1].enabled = true;
  phase_begin = Clock::now();
}
void PrintPhases() {
  for (std::size_t i = 0; i < phases.size(); ++i) {
    const auto& phase = phases[i];
    if (!phase.calls)
      continue;
    std::cout
        << "phase=" << kPhaseNames[i] << " calls=" << phase.calls
        << " exclusive_ms="
        << std::chrono::duration<double, std::milli>(phase.elapsed).count()
        << " cpp_live_bytes=" << heaps[i + 1].live.load()
        << " cpp_peak_bytes=" << heaps[i + 1].peak.load()
        << " cpp_allocations=" << heaps[i + 1].allocations.load() << '\n';
    std::cout << "phase=" << kPhaseNames[i] << ' ';
    PrintObservation("first_enter", phase.before);
    std::cout << "phase=" << kPhaseNames[i] << ' ';
    PrintObservation("last_leave", phase.after);
  }
}
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
  enum class Mode { kTemporary, kCreate, kRecover };
  Mode mode = Mode::kTemporary;
  fs::path root;
  int argument = 1;
  if (argc >= 3 && std::strcmp(argv[1], "--create") == 0) {
    mode = Mode::kCreate;
    root = argv[2];
    argument = 3;
  } else if (argc == 3 && std::strcmp(argv[1], "--recover") == 0) {
    mode = Mode::kRecover;
    root = argv[2];
    argument = 3;
  }
  if (argc - argument != 0 && argc - argument != 4) {
    std::cerr
        << "usage: cache_disk_startup_bench [--create new-directory] "
           "[manifests chunks private shared_chunks]\n"
           "       cache_disk_startup_bench --recover fixture-directory\n";
    return 2;
  }
  const bool counts = argc - argument == 4;
  std::size_t count = counts ? Number(argv[argument], 1000) : 8;
  std::size_t chunks = counts ? Number(argv[argument + 1], 60000) : 64;
  std::size_t private_count = counts ? Number(argv[argument + 2], 255) : 4;
  std::size_t shared = counts ? Number(argv[argument + 3], chunks) : 32;
  std::size_t expected_bytes{}, expected_references{};
  constexpr auto marker_name = ".gufo-disk-startup-bench-v1";
  if (mode == Mode::kRecover) {
    if (!fs::is_directory(fs::symlink_status(root)) ||
        !fs::is_regular_file(fs::symlink_status(root / marker_name)) ||
        fs::hard_link_count(root / marker_name) != 1)
      throw std::invalid_argument("not a marked synthetic benchmark fixture");
    std::ifstream marker(root / marker_name);
    std::string magic;
    if (!(marker >> magic >> count >> chunks >> private_count >> shared >>
          expected_bytes) ||
        magic != "gufo-disk-startup-bench-v1")
      throw std::invalid_argument("invalid benchmark fixture marker");
    marker >> std::ws;
    if (!marker.eof())
      throw std::invalid_argument("invalid benchmark fixture marker suffix");
  }
  if (!count || count > 1000 || chunks > 60000 || private_count > 255 ||
      shared > chunks || chunks + private_count > 65536)
    throw std::invalid_argument("invalid benchmark geometry");
  // Bound aggregate work as well as each argument. The individual limits make
  // these products fit size_t on every supported platform. A large selectable
  // fixture may approximate file counts, never allocate arbitrary disk space.
  if (count * (chunks + private_count) > 8000000 ||
      shared + count * (chunks - shared + private_count) > 200000)
    throw std::invalid_argument("benchmark aggregate limit exceeded");
  ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
  if (mode == Mode::kTemporary) {
    char name[] = "/tmp/gufo-disk-startup-bench-XXXXXX";
    const auto* temporary = mkdtemp(name);
    if (!temporary)
      throw std::runtime_error("cannot create benchmark directory");
    root = temporary;
  } else if (mode == Mode::kCreate && !fs::create_directory(root)) {
    throw std::invalid_argument("benchmark fixture directory must not exist");
  }
  struct Cleanup {
    const fs::path& root;
    bool remove;
    ~Cleanup() {
      if (remove) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
      }
    }
  } cleanup{root, mode != Mode::kRecover};
  const auto before_fixture = Observe();
  if (mode != Mode::kRecover) {
    {
      DiskStore initialize(ledger, root, UINT64_MAX);
    }
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
    if (mode == Mode::kCreate) {
      std::ofstream marker(root / marker_name);
      marker << "gufo-disk-startup-bench-v1 " << count << ' ' << chunks << ' '
             << private_count << ' ' << shared << ' ' << expected_bytes << '\n';
      marker.close();
      if (!marker)
        throw std::runtime_error("cannot write benchmark fixture marker");
      cleanup.remove = false;
      PrintObservation("before_fixture", before_fixture);
      PrintObservation("after_fixture", Observe());
      std::cout << "fixture_created=" << root
                << " references=" << expected_references
                << " distinct_payloads="
                << shared + count * (chunks - shared + private_count) << '\n';
      return 0;
    }
  } else {
    expected_references = count * (chunks + private_count);
  }
  (void)ledger.SnapshotAndResetPeaks();
#ifdef GUFO_CACHE_TESTING
  DiskStore::SetStartupPhaseHook(StartupPhase);
#endif
  const auto before_startup = Observe();
  Observation after_startup;
  heaps[0].enabled = true;
  const auto begin = Clock::now();
  {
    DiskStore store(ledger, root, UINT64_MAX);
    const auto elapsed =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    heaps[0].enabled = false;
    after_startup = Observe();
#ifdef GUFO_CACHE_TESTING
    DiskStore::SetStartupPhaseHook(nullptr);
#endif
    assert(store.Entries().size() == count);
    assert(store.Stats().manifest_bytes_read == expected_bytes);
    // Baseline and candidate may perform different numbers of stat calls for
    // the same references. Geometry derives the expected distinct identities.
    const auto distinct = shared + count * (chunks - shared + private_count);
    const auto stats = store.Stats();
    assert(stats.dependency_stats == expected_references ||
           stats.dependency_stats == distinct);
    if (stats.dependency_references || stats.distinct_dependency_files) {
      assert(stats.dependency_references == expected_references);
      assert(stats.distinct_dependency_files == distinct);
    }
    const auto snapshot = ledger.Snapshot();
    std::cout << "manifests=" << count << " references=" << expected_references
              << " unique_chunks=" << shared + count * (chunks - shared)
              << " private_files=" << count * private_count
              << " distinct_payloads=" << distinct
              << " dependency_references=" << stats.dependency_references
              << " distinct_dependency_files="
              << stats.distinct_dependency_files
              << " dependency_stats=" << stats.dependency_stats
              << " manifest_bytes_read=" << store.Stats().manifest_bytes_read
              << " startup_ms=" << elapsed
              << " logical_live_bytes=" << snapshot.ram_bytes
              << " logical_peak_bytes=" << snapshot.peak_ram_bytes
              << " cpp_heap_live_bytes=" << heaps[0].live.load()
              << " cpp_heap_peak_bytes=" << heaps[0].peak.load()
              << " cpp_allocations=" << heaps[0].allocations.load() << '\n';
    if (validation_measured)
      std::cout << "validation_ms=" << validation_ms
                << " validation_cpp_live_bytes="
                << heaps[kLegacyValidation].live.load()
                << " validation_cpp_peak_bytes="
                << heaps[kLegacyValidation].peak.load()
                << " validation_cpp_allocations="
                << heaps[kLegacyValidation].allocations.load() << '\n';
    PrintPhases();
  }
  const auto after_destroy = Observe();
  std::cout << "cpp_heap_after_destroy_bytes=" << heaps[0].live.load()
            << " logical_after_destroy_bytes=" << ledger.Snapshot().total_bytes
            << '\n';
  // Only this pair brackets constructor I/O; after_destroy also includes the
  // metrics printed while the store was alive.
  PrintObservation("before_startup", before_startup);
  PrintObservation("after_startup", after_startup);
  PrintObservation("after_destroy", after_destroy);
  assert(!ledger.Snapshot().total_bytes);
  for (const auto& heap : heaps)
    assert(!heap.live);
  assert(!phase_depth);
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
