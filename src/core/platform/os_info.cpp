#include "src/core/platform/os_info.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "src/core/platform/memory_observation_detail.hpp"

#if defined(_WIN32)
// clang-format off: psapi.h and bcrypt.h need the windows.h types above them,
// and the console and CRT isatty calls need the io and stdio headers.
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <io.h>
#include <stdio.h>
// clang-format on
#else
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace gufo::platform {

namespace {
std::string_view Trim(std::string_view text) noexcept {
  constexpr std::string_view whitespace = " \t\r";
  const auto begin = text.find_first_not_of(whitespace);
  if (begin == std::string_view::npos)
    return {};
  return text.substr(begin, text.find_last_not_of(whitespace) - begin + 1);
}

std::string_view Token(std::string_view& text) noexcept {
  text = Trim(text);
  const auto end = text.find_first_of(" \t\r");
  const auto token = text.substr(0, end);
  text = end == std::string_view::npos ? std::string_view{} : text.substr(end);
  return token;
}

std::optional<std::uint64_t> Unsigned(std::string_view text) noexcept {
  if (text.empty() || text.front() < '0' || text.front() > '9')
    return {};
  std::uint64_t value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size())
    return {};
  return value;
}

template<class F>
void Lines(std::string_view text, F consume) noexcept {
  if (text.size() > detail::kMemoryObservationFileLimit)
    return;
  while (!text.empty()) {
    const auto end = text.find('\n');
    consume(Trim(text.substr(0, end)));
    text = end == std::string_view::npos ? std::string_view{}
                                         : text.substr(end + 1);
  }
}

std::optional<std::uint64_t> Kilobytes(std::string_view text,
                                       std::string_view key) noexcept {
  bool found{};
  std::optional<std::uint64_t> result;
  Lines(text, [&](std::string_view line) {
    const auto colon = line.find(':');
    if (colon == std::string_view::npos || Trim(line.substr(0, colon)) != key)
      return;
    if (found) {
      result.reset();
      return;
    }
    found = true;
    auto rest = line.substr(colon + 1);
    const auto value = Unsigned(Token(rest));
    if (Token(rest) == "kB" && Trim(rest).empty() && value &&
        *value <= std::numeric_limits<std::uint64_t>::max() / 1024)
      result = *value * 1024;
  });
  return result;
}

std::optional<std::uint64_t> ArcSize(std::string_view text) noexcept {
  bool found{};
  std::optional<std::uint64_t> result;
  Lines(text, [&](std::string_view line) {
    if (Token(line) != "size")
      return;
    if (found) {
      result.reset();
      return;
    }
    found = true;
    // KSTAT_DATA_UINT64 records contain name, type (4), and byte value.
    if (Token(line) != "4")
      return;
    const auto value = Unsigned(Token(line));
    if (Trim(line).empty())
      result = value;
  });
  return result;
}

// Use one fixed buffer for all three files. A read failure or overlong file is
// wholly unavailable; never silently report a successfully parsed prefix.
std::string_view ReadObservationFile(
    const char* path,
    std::array<char, detail::kMemoryObservationFileLimit>& buffer) noexcept {
  auto* file = std::fopen(path, "r");
  if (!file)
    return {};
  const auto size = std::fread(buffer.data(), 1, buffer.size(), file);
  const bool complete =
      !std::ferror(file) && std::fgetc(file) == EOF && !std::ferror(file);
  std::fclose(file);
  return complete ? std::string_view(buffer.data(), size) : std::string_view{};
}
}  // namespace

namespace detail {
MemoryObservation ParseMemoryObservation(
    std::string_view process_status, std::string_view host_meminfo,
    std::string_view zfs_arcstats) noexcept {
  MemoryObservation result;
  result.process_rss_bytes = Kilobytes(process_status, "VmRSS");
  result.process_peak_rss_bytes = Kilobytes(process_status, "VmHWM");
  result.host_available_bytes = Kilobytes(host_meminfo, "MemAvailable");
  result.host_cached_bytes = Kilobytes(host_meminfo, "Cached");
  result.host_slab_bytes = Kilobytes(host_meminfo, "Slab");
  result.host_reclaimable_slab_bytes = Kilobytes(host_meminfo, "SReclaimable");
  result.zfs_arc_size_bytes = ArcSize(zfs_arcstats);
  return result;
}

MemoryObservation ReadMemoryObservationFiles(
    const char* process_status, const char* host_meminfo,
    const char* zfs_arcstats) noexcept {
  std::array<char, kMemoryObservationFileLimit> buffer;
  auto result = ParseMemoryObservation(
      ReadObservationFile(process_status, buffer), {}, {});
  const auto host =
      ParseMemoryObservation({}, ReadObservationFile(host_meminfo, buffer), {});
  result.host_available_bytes = host.host_available_bytes;
  result.host_cached_bytes = host.host_cached_bytes;
  result.host_slab_bytes = host.host_slab_bytes;
  result.host_reclaimable_slab_bytes = host.host_reclaimable_slab_bytes;
  result.zfs_arc_size_bytes =
      ParseMemoryObservation({}, {}, ReadObservationFile(zfs_arcstats, buffer))
          .zfs_arc_size_bytes;
  return result;
}
}  // namespace detail

MemoryObservation QueryMemoryObservation() noexcept {
  const auto begin = std::chrono::steady_clock::now();
  MemoryObservation result;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    result.process_rss_bytes = counters.WorkingSetSize;
    result.process_peak_rss_bytes = counters.PeakWorkingSetSize;
  }
  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  if (GlobalMemoryStatusEx(&memory))
    result.host_available_bytes = memory.ullAvailPhys;
#else
  result = detail::ReadMemoryObservationFiles(
      "/proc/self/status", "/proc/meminfo", "/proc/spl/kstat/zfs/arcstats");
#endif
  result.collection_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
  return result;
}

std::size_t PageSizeBytes() {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                              : static_cast<std::size_t>(4096);
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::size_t>(page_size)
                       : static_cast<std::size_t>(4096);
#endif
}

std::uint64_t AvailableMemoryBytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return 0;
  return status.ullAvailPhys;
#else
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return (pages > 0 && page_size > 0)
             ? static_cast<std::uint64_t>(pages) * page_size
             : 0;
#endif
}

int ProcessId() noexcept {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentProcessId());
#else
  return ::getpid();
#endif
}

std::optional<HostIdentity> QueryHostIdentity() {
#if defined(_WIN32)
  // RtlGetVersion reports the real NT version; GetVersionEx is capped by the
  // application manifest.
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  const auto get_version = ntdll == nullptr
                               ? nullptr
                               : reinterpret_cast<RtlGetVersionFn>(
                                     GetProcAddress(ntdll, "RtlGetVersion"));
  OSVERSIONINFOW version{};
  version.dwOSVersionInfoSize = sizeof(version);
  if (get_version == nullptr || get_version(&version) != 0)
    return std::nullopt;
  HostIdentity identity;
  identity.sysname = "Windows";
  char name[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD length = sizeof(name);
  if (GetComputerNameA(name, &length))
    identity.nodename.assign(name, length);
  identity.release = std::to_string(version.dwMajorVersion) + "." +
                     std::to_string(version.dwMinorVersion) + "." +
                     std::to_string(version.dwBuildNumber);
  identity.version = "build " + std::to_string(version.dwBuildNumber);
  identity.machine = "x86_64";
  return identity;
#else
  struct utsname uts{};
  if (::uname(&uts) != 0)
    return std::nullopt;
  return HostIdentity{uts.sysname, uts.nodename, uts.release, uts.version,
                      uts.machine};
#endif
}

const char* HostOsName() noexcept {
#if defined(_WIN32)
  return "Windows";
#else
  return "Linux";
#endif
}

std::optional<HostCpu> QueryHostCpu() {
#if defined(_WIN32)
  HostCpu cpu;
  cpu.logical_cores = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes != 0) {
    std::vector<std::byte> buffer(bytes);
    auto* const first =
        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffer.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, first,
                                         &bytes)) {
      for (DWORD offset = 0; offset < bytes;) {
        const auto* const entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                buffer.data() + offset);
        ++cpu.physical_cores;
        offset += entry->Size;
      }
    }
  }
  constexpr const char* kProcessorKey =
      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
  const auto read_string = [&](const char* value) {
    char text[256]{};
    DWORD size = sizeof(text);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, kProcessorKey, value, RRF_RT_REG_SZ,
                     nullptr, text, &size) != ERROR_SUCCESS)
      return std::string();
    std::string result(text);
    const auto end = result.find_last_not_of(' ');
    result.erase(end == std::string::npos ? 0 : end + 1);
    return result;
  };
  cpu.model_name = read_string("ProcessorNameString");
  cpu.vendor = read_string("VendorIdentifier");
  if (cpu.logical_cores == 0 && cpu.physical_cores == 0)
    return std::nullopt;
  return cpu;
#else
  const long online = ::sysconf(_SC_NPROCESSORS_ONLN);
  if (online <= 0)
    return std::nullopt;
  HostCpu cpu;
  cpu.logical_cores = static_cast<std::uint32_t>(online);
  return cpu;
#endif
}

std::optional<HostMemory> QueryHostMemory() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return std::nullopt;
  return HostMemory{status.ullTotalPhys, status.ullAvailPhys};
#else
  const long pages = ::sysconf(_SC_PHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return std::nullopt;
  return HostMemory{static_cast<std::uint64_t>(pages) * page_size,
                    AvailableMemoryBytes()};
#endif
}

bool LocalTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::localtime_s(out, &seconds) == 0;
#else
  return ::localtime_r(&seconds, out) != nullptr;
#endif
}

bool UtcTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::gmtime_s(out, &seconds) == 0;
#else
  return ::gmtime_r(&seconds, out) != nullptr;
#endif
}

bool StderrIsTerminal() noexcept {
#if defined(_WIN32)
  return ::_isatty(::_fileno(stderr)) != 0;
#else
  return ::isatty(STDERR_FILENO) != 0;
#endif
}

bool StdinIsTerminal() noexcept {
#if defined(_WIN32)
  return ::_isatty(::_fileno(stdin)) != 0;
#else
  return ::isatty(STDIN_FILENO) != 0;
#endif
}

std::filesystem::path ExecutablePath() {
#if defined(_WIN32)
  std::wstring path(MAX_PATH, L'\0');
  while (true) {
    const DWORD length = GetModuleFileNameW(nullptr, path.data(),
                                            static_cast<DWORD>(path.size()));
    if (length == 0)
      throw std::runtime_error("cannot determine the executable path");
    if (length < path.size()) {
      path.resize(length);
      return std::filesystem::path(path);
    }
    path.resize(path.size() * 2);
  }
#else
  std::error_code ec;
  auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec)
    throw std::runtime_error("cannot determine the executable path: " +
                             ec.message());
  return path;
#endif
}

PageFaults ReadPageFaults() noexcept {
#if defined(_WIN32)
  // The Windows counter totals soft and hard faults; the split is not
  // reported, so the count lands in the minor field.
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return {};
  return {counters.PageFaultCount, 0};
#else
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    return {};
  return {static_cast<std::uint64_t>(usage.ru_minflt),
          static_cast<std::uint64_t>(usage.ru_majflt)};
#endif
}

bool FillRandomBytes(std::span<std::uint8_t> out) noexcept {
  if (out.empty())
    return true;
#if defined(_WIN32)
  return BCryptGenRandom(nullptr, out.data(), static_cast<ULONG>(out.size()),
                         BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
  return ::getrandom(out.data(), out.size(), 0) ==
         static_cast<ssize_t>(out.size());
#endif
}

}  // namespace gufo::platform
