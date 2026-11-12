#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "src/core/platform/memory_observation_detail.hpp"
#include "tests/support/temporary_directory.hpp"

namespace {
void Require(bool condition, int line) {
  if (!condition) {
    std::fprintf(stderr, "memory observation check failed at line %d\n", line);
    std::exit(1);
  }
}
#define CHECK(condition) Require((condition), __LINE__)
using gufo::platform::detail::ParseMemoryObservation;
}  // namespace

int main() {
  const auto observed = ParseMemoryObservation(
      "Name:\tgufo\nVmRSS:\t123 kB\nVmHWM: 456 kB\n",
      "MemAvailable: 789 kB\nCached: 12 kB\nSlab: 34 kB\n"
      "SReclaimable: 56 kB\nSwapCached: 999 kB\n",
      "13 1 0x01 96 4608 0 0\nname type data\nsize 4 987654321\n"
      "metadata_size 4 12345\n");
  CHECK(observed.process_rss_bytes == 123 * 1024);
  CHECK(observed.process_peak_rss_bytes == 456 * 1024);
  CHECK(observed.host_available_bytes == 789 * 1024);
  CHECK(observed.host_cached_bytes == 12 * 1024);
  CHECK(observed.host_slab_bytes == 34 * 1024);
  CHECK(observed.host_reclaimable_slab_bytes == 56 * 1024);
  CHECK(observed.zfs_arc_size_bytes == 987654321);
  CHECK(observed.collection_us == 0);

  const auto missing = ParseMemoryObservation({}, {}, {});
  CHECK(!missing.process_rss_bytes && !missing.process_peak_rss_bytes);
  CHECK(!missing.host_available_bytes && !missing.host_cached_bytes);
  CHECK(!missing.host_slab_bytes && !missing.host_reclaimable_slab_bytes);
  CHECK(!missing.zfs_arc_size_bytes);
  const auto zero =
      ParseMemoryObservation("VmRSS: 0 kB", "Cached: 0 kB", "size 4 0");
  CHECK(zero.process_rss_bytes && *zero.process_rss_bytes == 0);
  CHECK(zero.host_cached_bytes && *zero.host_cached_bytes == 0);
  CHECK(zero.zfs_arc_size_bytes && *zero.zfs_arc_size_bytes == 0);

  for (const auto* value :
       {"-1 kB", "+1 kB", "1 MB", "1", "1 kB junk", "1.0 kB",
        "18446744073709551616 kB", "18014398509481984 kB", "not-a-number kB"}) {
    const auto bad = ParseMemoryObservation(std::string("VmRSS: ") + value,
                                            "MemAvailable: 1 kB", {});
    CHECK(!bad.process_rss_bytes);
    CHECK(bad.host_available_bytes == 1024);
  }
  const auto boundary = ParseMemoryObservation(
      "VmRSS: 18014398509481983 kB", {}, "size 4 18446744073709551615");
  CHECK(boundary.process_rss_bytes ==
        (std::numeric_limits<std::uint64_t>::max() / 1024) * 1024);
  CHECK(boundary.zfs_arc_size_bytes ==
        std::numeric_limits<std::uint64_t>::max());

  for (const auto* arc :
       {"size 3 1", "size 4 -1", "size 4 +1", "size 4 1 bytes", "size 4",
        "size 4 1.0", "size 4 18446744073709551616"})
    CHECK(!ParseMemoryObservation({}, {}, arc).zfs_arc_size_bytes);
  const auto duplicate = ParseMemoryObservation(
      "VmRSS: 1 kB\nVmRSS: 2 kB\nVmRSS: 3 kB\nVmHWM: 4 kB",
      "Cached: nope\nCached: 1 kB\nSlab: 2 kB", "size 4 1\nsize 4 2\nsize 4 3");
  CHECK(!duplicate.process_rss_bytes &&
        duplicate.process_peak_rss_bytes == 4096);
  CHECK(!duplicate.host_cached_bytes && duplicate.host_slab_bytes == 2048);
  CHECK(!duplicate.zfs_arc_size_bytes);

  const auto limit = gufo::platform::detail::kMemoryObservationFileLimit;
  const std::string oversized = "VmRSS: 1 kB\n" + std::string(limit, '\n');
  const auto bounded = ParseMemoryObservation(oversized, "Slab: 2 kB", {});
  CHECK(!bounded.process_rss_bytes && bounded.host_slab_bytes == 2048);
  const std::string exact =
      "VmRSS: 1 kB\n" +
      std::string(limit - std::string("VmRSS: 1 kB\n").size(), '\n');
  CHECK(ParseMemoryObservation(exact, {}, {}).process_rss_bytes == 1024);

  // Exercise the bounded reader used by the production query, without
  // depending on the host's /proc files, permissions or installed ZFS module.
  const TemporaryDirectory files("gufo-memory-observation-test");
  const auto status = (files.path() / "status").string();
  const auto meminfo = (files.path() / "meminfo").string();
  const auto arc = (files.path() / "arcstats").string();
  const auto write = [](const std::string& name, const std::string& contents) {
    std::ofstream output(name, std::ios::binary);
    output << contents;
    output.close();
    CHECK(bool(output));
  };
  const auto read = [&] {
    return gufo::platform::detail::ReadMemoryObservationFiles(
        status.c_str(), meminfo.c_str(), arc.c_str());
  };
  const auto absent_files = read();
  CHECK(!absent_files.process_rss_bytes && !absent_files.host_available_bytes &&
        !absent_files.zfs_arc_size_bytes);
  write(status, exact);
  write(meminfo, "MemAvailable: 0 kB\nCached: 3 kB\n");
  write(arc, "size 4 17\n");
  const auto exact_file = read();
  CHECK(exact_file.process_rss_bytes == 1024);
  CHECK(exact_file.host_available_bytes &&
        *exact_file.host_available_bytes == 0);
  CHECK(exact_file.host_cached_bytes == 3072 &&
        exact_file.zfs_arc_size_bytes == 17);
  write(status, exact + '\n');
  write(arc, "size 4 17\n" + std::string(limit, '\n'));
  const auto oversized_files = read();
  CHECK(!oversized_files.process_rss_bytes &&
        !oversized_files.zfs_arc_size_bytes);
  CHECK(oversized_files.host_cached_bytes == 3072);
  std::filesystem::remove(status);
  std::filesystem::create_directory(status);
  const auto unreadable = read();
  CHECK(!unreadable.process_rss_bytes && unreadable.host_cached_bytes == 3072);

  // A real query is best effort. No platform availability or physical-memory
  // amount is a test prerequisite; deterministic value checks are above.
  (void)gufo::platform::QueryMemoryObservation();
  std::puts("memory_observation_test passed");
}
