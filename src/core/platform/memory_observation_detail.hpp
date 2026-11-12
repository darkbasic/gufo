#ifndef GUFO_CORE_PLATFORM_MEMORY_OBSERVATION_DETAIL_HPP_
#define GUFO_CORE_PLATFORM_MEMORY_OBSERVATION_DETAIL_HPP_

#include <cstddef>
#include <string_view>

#include "src/core/platform/os_info.hpp"

namespace gufo::platform::detail {

// Private parsing boundary shared by the production reader and CPU fixtures.
// Each input is a complete OS text file. An oversized input contributes no
// values; unavailable inputs are empty. Parsing performs no heap allocation.
inline constexpr std::size_t kMemoryObservationFileLimit = 64 * 1024;
MemoryObservation ParseMemoryObservation(
    std::string_view process_status, std::string_view host_meminfo,
    std::string_view zfs_arcstats) noexcept;

// Reads the same bounded complete files as the Linux query. Private paths are
// injectable for CPU fixtures; missing/unreadable inputs contribute no values.
MemoryObservation ReadMemoryObservationFiles(const char* process_status,
                                             const char* host_meminfo,
                                             const char* zfs_arcstats) noexcept;

}  // namespace gufo::platform::detail

#endif
