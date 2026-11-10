#ifndef GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_
#define GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_

#include <atomic>
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

/// One model-lifetime capacity observation. Fixed-pool mode reserves capacity
/// at startup. On-demand mode protects text/cache/scratch obligations instead;
/// images and optional device snapshot copies acquire only actual demand from
/// the remainder. Lazy backing spends its protected allowance, never this
/// remainder. Later free readings cannot replenish either ledger or spend
/// committed storage twice. External pressure can still cause allocation
/// failure.
class AdmissionBudget {
public:
  explicit AdmissionBudget(std::size_t capacity);
  AdmissionBudget(std::size_t capacity, std::size_t protected_bytes);
  [[nodiscard]] cache::ResourceReservation Reserve(std::size_t bytes);
  [[nodiscard]] cache::ResourceReservation ReserveShared(std::size_t bytes);
  [[nodiscard]] std::size_t Capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool OnDemand() const noexcept { return on_demand_; }
  [[nodiscard]] std::size_t ProtectedBytes() const noexcept {
    return protected_bytes_;
  }
  [[nodiscard]] std::size_t AccountedBytes() const;
  /// One startup epoch funds one execution/cache owner. Reusing its runner
  /// for another pool would promise the same text/cache allowance twice.
  /// Failed initialization requires a new runner and capacity observation.
  void ClaimExecutionOwner();
  /// Additional obligations, including both retained allocations and work
  /// admitted before allocation. They are not subtracted from a newer reading.
  [[nodiscard]] std::size_t Outstanding() const;

private:
  [[nodiscard]] cache::ResourceReservation Reserve(
      cache::ResourceCategory category, std::size_t bytes);
  const std::size_t capacity_;
  const bool on_demand_{false};
  const std::size_t protected_bytes_{0};
  cache::ResourceLedger ledger_;
  cache::ResourceReservation protected_;
  std::atomic_flag execution_owner_{};
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
  /// On-demand prepared owners keep only actual RGB capacity after resizing.
  void Finish(std::size_t pixel_bytes);

private:
  cache::ResourceReservation scratch_;
  cache::ResourceReservation pixels_;
  bool on_demand_{false};
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
  /// Call after transient allocations/work end; retained and future required
  /// buffers must remain covered. Never allocates and cannot grow the claim.
  void Shrink(std::size_t bytes) noexcept;
  void Clear() noexcept;
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }

private:
  std::shared_ptr<AdmissionBudget> budget_;
  struct Part {
    cache::ResourceReservation reservation;
    std::size_t bytes;
  };
  std::vector<Part> reservations_;
  std::size_t bytes_{0};
};

struct StartupCapacity {
  std::size_t device_admission_bytes;
  std::size_t host_ceiling_bytes;
  std::size_t available_bytes;
  [[nodiscard]] const char* LimiterName() const noexcept;
};

/// Both inputs describe remaining capacity at the same loading phase. Loaded
/// weights and committed shared buffers are already reflected in these values;
/// their accounting totals must never be added back to either observation.
[[nodiscard]] StartupCapacity SelectStartupCapacity(
    std::size_t device_admission_bytes, std::size_t host_ceiling_bytes);

struct StartupMemoryPlan {
  std::size_t state_capacity_bytes;
  std::size_t per_session_bytes;
  std::size_t shared_reservation_bytes;
  [[nodiscard]] bool Fits(std::size_t sessions) const noexcept;
  [[nodiscard]] std::size_t RequiredBytes(std::size_t sessions) const;
  [[nodiscard]] std::size_t DeficitBytes(std::size_t sessions,
                                         std::size_t available) const;
};

/// Preserve legacy admission when vision_budget is zero and on_demand is false.
/// On-demand startup has no image deduction and preserves lazy cache backing.
/// Fixed-pool mode reserves vision_budget as before. Both opt-in modes
/// subtract every UNCOMMITTED shared allowance once, then test text/MTP
/// state per session. The cache grant is reserved in full, never trimmed here.
/// Already committed allocations lower the capacity snapshot instead. For
/// example startup DeferredScratchBytes omits existing logits/checkpoint/wide
/// buffers. Its snapshot-spare bound is deliberately retained even at runtime;
/// this planner does not reinterpret that owner's safety policy.
[[nodiscard]] StartupMemoryPlan PlanStartupMemory(
    std::size_t available_bytes, std::size_t deferred_scratch,
    std::size_t base_per_session, std::size_t conservative_per_session,
    std::size_t vision_budget, std::size_t cache_budget,
    std::size_t runtime_margin, bool on_demand = false);

}  // namespace gufo::models::qwen::vision

#endif
