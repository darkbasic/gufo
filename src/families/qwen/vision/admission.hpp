#ifndef GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_
#define GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/resource_error.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::size_t kEncoderHidden = 1152;
inline constexpr std::size_t kEncoderFfn = 4304;
inline constexpr std::size_t kEncoderHeads = 16;
inline constexpr std::size_t kEncoderPatchDim = 1536;
inline constexpr std::size_t kPinnedUploadBytes = 16ULL << 20;
[[nodiscard]] constexpr std::size_t EncoderTileRows(std::size_t patches) {
  const auto limit = patches == 4096 ? 64U : 128U;
  return patches < limit ? patches : limit;
}

[[nodiscard]] std::size_t CheckedAdd(std::size_t a, std::size_t b);
[[nodiscard]] std::size_t CheckedMultiply(std::size_t a, std::size_t b);

struct ImageDimensions {
  std::uint32_t width;
  std::uint32_t height;
};

/// Geometry has already passed resizing: both sides are positive multiples
/// of 32 and at most 16,777,216 pixels. The encoder uses the same checks.
[[nodiscard]] std::size_t ImageScratchBytes(ImageDimensions image);
[[nodiscard]] std::size_t ImageEmbeddingBytes(ImageDimensions image,
                                              std::uint32_t output_width);
/// All history embeddings and prepared RGB pixels, the largest sequential
/// encoder working set, and both host and device positional buffers. The stable
/// descriptor belongs to the base text-session claim. Callers add old buffers
/// that coexist during replacement. No image or prompt content enters the
/// budget or its errors.
[[nodiscard]] std::size_t RequestMemoryBytes(
    std::size_t prefix_rows, std::span<const ImageDimensions> images,
    std::uint32_t output_width);

/// A fixed, model-wide budget carved out before text-session/cache admission.
/// Physical free-memory readings never replenish it: outstanding reservations
/// are always included by ResourceLedger's atomic capacity check.
class AdmissionBudget {
public:
  explicit AdmissionBudget(std::size_t capacity);
  [[nodiscard]] cache::ResourceReservation Reserve(std::size_t bytes);
  [[nodiscard]] std::size_t Capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t Outstanding() const;

private:
  const std::size_t capacity_;
  cache::ResourceLedger ledger_;
};

/// Decode/resize processes one image at a time. The working allowance covers
/// the 32 Mi-pixel decoder limit, orientation copy, RGB resize intermediate,
/// filter banks and decoder workspaces. Each prepared image retains at most
/// 16 Mi-pixels of RGB8. These are logical bounds, never eager allocations.
inline constexpr std::size_t kPreparationScratchBytes = std::size_t{1} << 30;
inline constexpr std::size_t kPreparedImageBytes = std::size_t{48} << 20;

/// Reserve before preparing or counting an image prompt. Finish releases only
/// transient work after decode/resize returns. Keep this owner alive until all
/// prepared pixels are freed, including queued/cancelled requests and image
/// history retained by a session. Construction failure releases both claims.
/// A null budget or no images preserves the conservative/text-only path.
class PreparationReservation {
public:
  PreparationReservation(std::shared_ptr<AdmissionBudget> budget,
                         std::size_t image_count);
  PreparationReservation(const PreparationReservation&) = delete;
  PreparationReservation& operator=(const PreparationReservation&) = delete;
  void Finish() noexcept;

private:
  cache::ResourceReservation scratch_;
  cache::ResourceReservation pixels_;
};

/// One session's high-water claim. Grow precedes any image allocation; failed
/// growth leaves it unchanged. A physical allocation failure after Grow keeps
/// the claim, since some buffers may already exist. Clear is legal only after
/// all corresponding device/host work has finished and buffers are freed.
/// Destruction has the same ordering requirement. A null budget is the
/// conservative policy and performs no additional admission.
class AdmissionLease {
public:
  AdmissionLease() = default;
  explicit AdmissionLease(std::shared_ptr<AdmissionBudget> budget);
  AdmissionLease(AdmissionLease&&) noexcept = default;
  AdmissionLease& operator=(AdmissionLease&&) noexcept = default;
  AdmissionLease(const AdmissionLease&) = delete;
  AdmissionLease& operator=(const AdmissionLease&) = delete;
  void Grow(std::size_t bytes);
  void Clear() noexcept;
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }

private:
  std::shared_ptr<AdmissionBudget> budget_;
  std::vector<cache::ResourceReservation> reservations_;
  std::size_t bytes_{0};
};

struct StartupMemoryPlan {
  std::size_t state_capacity_bytes;
  std::size_t per_session_bytes;
  std::size_t shared_reservation_bytes;
  [[nodiscard]] bool Fits(std::size_t sessions) const noexcept;
};

/// Preserve legacy admission when vision_budget is zero. Request-aware mode
/// subtracts every shared allowance once, then tests the unchanged text/MTP
/// state per session. The cache grant is reserved in full, never trimmed here.
[[nodiscard]] StartupMemoryPlan PlanStartupMemory(
    std::size_t available_bytes, std::size_t deferred_scratch,
    std::size_t base_per_session, std::size_t conservative_per_session,
    std::size_t vision_budget, std::size_t cache_budget,
    std::size_t runtime_margin);

}  // namespace gufo::models::qwen::vision

#endif
