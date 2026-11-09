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

/// Zero preserves conservative admission. A positive budget opts into a
/// bounded pool shared by every prepared image request and execution session.
inline std::size_t VisionMemoryBudgetBytes(
    const ext::TextRunnerOptions& options) {
  const auto* bytes = ext::FindOption<std::int64_t>(options.provider_options,
                                                    "--vision-memory-bytes");
  if (!bytes || *bytes == 0)
    return 0;
  if (*bytes < 0 || static_cast<std::uint64_t>(*bytes) >
                        std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "--vision-memory-bytes must be a nonnegative byte count");
  if (options.vision_model_path.empty())
    throw std::invalid_argument("--vision-memory-bytes requires --mmproj");
  return static_cast<std::size_t>(*bytes);
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
