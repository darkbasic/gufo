#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gufo/ext/component_cache_resources.hpp"
#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

// Only the legacy fixed pool promises a particular cache allowance at startup.
// On-demand leaves initial sizing and post-slot trims to the ordinary serving
// cache. This helper is used at both grant boundaries, before any commitment.
inline void ValidateVisionCacheGrant(
    const std::shared_ptr<qwen::vision::AdmissionBudget>& budget,
    std::size_t required, std::size_t granted,
    std::optional<std::size_t> reserved = std::nullopt) {
  if (budget && !budget->OnDemand() &&
      (granted < required || (reserved && granted > *reserved)))
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=cache_grant requested_bytes=" +
        std::to_string(required) + " granted_bytes=" + std::to_string(granted));
}

// Preserve the upstream host ceiling, lazy backing and pressure checks in
// on-demand mode. Only an arriving segment competes with image/private-copy
// reservations. Fixed pools retain their previous stronger backing policy.
inline hip::ComponentCacheHooks VisionCacheHooks(
    hip::ComponentCacheHooks upstream,
    const std::shared_ptr<qwen::vision::AdmissionBudget>& budget,
    std::function<std::size_t()> fixed_ceiling) {
  if (!budget)
    return upstream;
  upstream.require_full_backing = !budget->OnDemand();
  if (budget->OnDemand()) {
    upstream.reserve_segment = [budget](std::size_t bytes) {
      return budget->ReserveShared(bytes);
    };
  } else {
    upstream.host_ceiling = std::move(fixed_ceiling);
  }
  return upstream;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
