#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {
class Model;

/// Explicit on-demand admission cannot be combined with the legacy fixed
/// pool option, including its conservative zero spelling. Validate here so
/// either serving option helper rejects ambiguous native or typed requests.
inline bool VisionMemoryOnDemand(const ext::TextRunnerOptions& options) {
  const auto* policy = ext::FindOption<std::string>(options.provider_options,
                                                    "--vision-memory-policy");
  const auto* bytes = ext::FindOption<std::int64_t>(options.provider_options,
                                                    "--vision-memory-bytes");
  if (policy && *policy != "conservative" && *policy != "on-demand")
    throw std::invalid_argument(
        "--vision-memory-policy must be conservative or on-demand");
  const bool on_demand = policy && *policy == "on-demand";
  if (on_demand && bytes)
    throw std::invalid_argument(
        "--vision-memory-policy on-demand cannot be combined with "
        "--vision-memory-bytes");
  if (bytes && (*bytes < 0 || static_cast<std::uint64_t>(*bytes) >
                                  std::numeric_limits<std::size_t>::max()))
    throw std::invalid_argument(
        "--vision-memory-bytes must be a nonnegative byte count");
  if (policy && !on_demand && bytes && *bytes > 0)
    throw std::invalid_argument(
        "--vision-memory-policy conservative cannot be combined with a "
        "positive --vision-memory-bytes");
  if (options.vision_model_path.empty()) {
    if (on_demand)
      throw std::invalid_argument(
          "--vision-memory-policy on-demand requires --mmproj");
    if (bytes && *bytes > 0)
      throw std::invalid_argument("--vision-memory-bytes requires --mmproj");
  }
  return on_demand;
}

/// Preserve the legacy spelling: absent/zero means conservative admission,
/// while a positive value reserves the fixed shared pool. On-demand selection
/// is independent and must be read with VisionMemoryOnDemand().
inline std::size_t VisionMemoryBudgetBytes(
    const ext::TextRunnerOptions& options) {
  (void)VisionMemoryOnDemand(options);
  const auto* bytes = ext::FindOption<std::int64_t>(options.provider_options,
                                                    "--vision-memory-bytes");
  return bytes ? static_cast<std::size_t>(*bytes) : 0;
}

inline constexpr std::uint32_t DecodeConcurrencyForSessions(
    std::size_t session_count) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(session_count, 1, 8));
}

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {});
#endif

}  // namespace gufo::models::qwen38_flash_next
