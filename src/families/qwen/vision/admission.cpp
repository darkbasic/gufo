#include "src/families/qwen/vision/admission.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "src/core/image.hpp"

namespace gufo::models::qwen::vision {
namespace {
std::size_t PatchCount(ImageDimensions image) {
  if (!image.width || !image.height || image.width % 32 || image.height % 32 ||
      std::uint64_t{image.width} * image.height > 16777216) {
    throw std::invalid_argument("vision encoder requires a resized RGB32 grid");
  }
  return CheckedMultiply(image.width / 16, image.height / 16);
}
}  // namespace

std::size_t CheckedAdd(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=arithmetic_overflow");
  return a + b;
}

std::size_t CheckedMultiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=arithmetic_overflow");
  return a * b;
}

std::size_t ImageScratchBytes(ImageDimensions image) {
  const auto patches = PatchCount(image);
  const auto tile = EncoderTileRows(patches);
  // encoder.hip: pixels, hidden, work, normalized, QKV and attention are
  // 768 + 4608 + 17216 + 8608 + 6912 + 2304 bytes per patch. Scores and
  // probabilities are 16 heads * tile * patches * (4 + 2) bytes.
  const auto per_patch =
      16 * 16 * 3 + kEncoderHidden * sizeof(float) +
      std::max(kEncoderFfn, 3 * kEncoderHidden) * sizeof(float) +
      std::max(kEncoderFfn, kEncoderPatchDim) * sizeof(std::uint16_t) +
      3 * kEncoderHidden * sizeof(std::uint16_t) +
      kEncoderHidden * sizeof(std::uint16_t) +
      kEncoderHeads * tile * (sizeof(float) + sizeof(std::uint16_t));
  auto bytes = CheckedMultiply(patches, per_patch);
  bytes = CheckedAdd(bytes, kEncoderHidden * tile * sizeof(float));
  // Upload stages through explicit pinned host memory on the unified device.
  return CheckedAdd(bytes,
                    std::min<std::size_t>(768 * patches, kPinnedUploadBytes));
}

std::size_t ImageEmbeddingBytes(ImageDimensions image,
                                std::uint32_t output_width) {
  if (!output_width)
    throw std::invalid_argument("vision embedding width must be positive");
  // Four ViT patches merge to one FP32 output embedding.
  return CheckedMultiply(PatchCount(image), output_width);
}

std::size_t RequestMemoryBytes(std::size_t prefix_rows,
                               std::span<const ImageDimensions> images,
                               std::uint32_t output_width) {
  auto bytes = CheckedMultiply(prefix_rows, 6 * sizeof(std::int32_t));
  std::size_t scratch = 0;
  for (const auto image : images) {
    bytes = CheckedAdd(bytes, ImageEmbeddingBytes(image, output_width));
    bytes = CheckedAdd(bytes, CheckedMultiply(PatchCount(image), 16 * 16 * 3));
    scratch = std::max(scratch, ImageScratchBytes(image));
  }
  return CheckedAdd(bytes, scratch);
}

AdmissionBudget::AdmissionBudget(std::size_t capacity)
    : capacity_(capacity), ledger_({capacity, capacity, 0, 0}) {}

cache::ResourceReservation AdmissionBudget::Reserve(std::size_t bytes) {
  if (!bytes)
    return {};
  try {
    return ledger_.Reserve(cache::ResourceCategory::kPrivateState, bytes);
  } catch (const cache::ResourceExhausted&) {
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=capacity requested_bytes=" +
        std::to_string(bytes) +
        " outstanding_bytes=" + std::to_string(Outstanding()) +
        " capacity_bytes=" + std::to_string(capacity_));
  }
}

std::size_t AdmissionBudget::Outstanding() const {
  return ledger_.Snapshot().total_bytes;
}

PreparationReservation::PreparationReservation(
    std::shared_ptr<AdmissionBudget> budget, std::size_t image_count) {
  static_assert(core::kMaxImagePixels == std::size_t{32} * 1024 * 1024,
                "revisit the decode working bound if image limits change");
  if (!budget || !image_count)
    return;
  const auto pixels = CheckedMultiply(image_count, kPreparedImageBytes);
  (void)CheckedAdd(pixels, kPreparationScratchBytes);
  scratch_ = budget->Reserve(kPreparationScratchBytes);
  pixels_ = budget->Reserve(pixels);
}

void PreparationReservation::Finish() noexcept {
  scratch_ = {};
}

AdmissionLease::AdmissionLease(std::shared_ptr<AdmissionBudget> budget)
    : budget_(std::move(budget)) {}

void AdmissionLease::Grow(std::size_t bytes) {
  if (!budget_ || bytes <= bytes_)
    return;
  auto reservation = budget_->Reserve(bytes - bytes_);
  // vector growth may fail too; the uncommitted local reservation rolls back.
  reservations_.push_back(std::move(reservation));
  bytes_ = bytes;
}

void AdmissionLease::Clear() noexcept {
  reservations_.clear();
  bytes_ = 0;
}

bool StartupMemoryPlan::Fits(std::size_t sessions) const noexcept {
  return !per_session_bytes ||
         sessions <= state_capacity_bytes / per_session_bytes;
}

StartupMemoryPlan PlanStartupMemory(std::size_t available_bytes,
                                    std::size_t deferred_scratch,
                                    std::size_t base_per_session,
                                    std::size_t conservative_per_session,
                                    std::size_t vision_budget,
                                    std::size_t cache_budget,
                                    std::size_t runtime_margin) {
  if (!vision_budget) {
    return {available_bytes > deferred_scratch
                ? available_bytes - deferred_scratch
                : 0,
            conservative_per_session, deferred_scratch};
  }
  const auto shared = CheckedAdd(CheckedAdd(deferred_scratch, vision_budget),
                                 CheckedAdd(cache_budget, runtime_margin));
  if (shared > available_bytes) {
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=shared_capacity "
        "requested_bytes=" +
        std::to_string(shared) +
        " capacity_bytes=" + std::to_string(available_bytes));
  }
  return {available_bytes - shared, base_per_session, shared};
}

}  // namespace gufo::models::qwen::vision
