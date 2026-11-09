#include "src/families/qwen/vision/device_input.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "src/core/logging/logging.hpp"

namespace gufo::models::qwen::vision {
static_assert(sizeof(DeviceRope) <= 64);
namespace {
void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

}  // namespace

DeviceInput::~DeviceInput() {
  if (descriptor_ != nullptr)
    (void)hipFree(descriptor_);
  if (positions_ != nullptr)
    (void)hipFree(positions_);
}

void DeviceInput::SetMemoryAdmission(std::shared_ptr<AdmissionBudget> budget) {
  if (admission_budget_ || capacity_ || !embeddings_.empty())
    throw std::logic_error("vision admission must be set before requests");
  admission_budget_ = std::move(budget);
  admission_ = AdmissionLease(admission_budget_);
}

void DeviceInput::ReserveRequest(const Prompt* prompt, const Encoder* encoder,
                                 std::size_t prefix_rows, bool replacement) {
  if (!admission_budget_)
    return;
  if (prefix_rows == 0 && (!prompt || prompt->images.empty()))
    return;
  std::vector<ImageDimensions> images;
  if (prompt) {
    images.reserve(prompt->images.size());
    for (const auto& image : prompt->images) {
      if (!image.pixels || !encoder)
        throw std::invalid_argument("image input has no pixels or encoder");
      images.push_back({image.pixels->width, image.pixels->height});
    }
  }
  const auto required =
      RequestMemoryBytes(std::max(capacity_, prefix_rows), images,
                         encoder ? encoder->OutputWidth() : 0);
  // Count the old physical buffers as well as the complete replacement.
  // This deliberately overestimates the free-before-grow position path and
  // makes partial allocation/copy failures safe without transactional HIP
  // allocation. Keep this high-water claim until image storage is released.
  auto retained = Bytes() - (descriptor_ ? sizeof(DeviceRope) : 0);
  if (replacement && prompt_) {
    // This input keeps the old attachment until Configure replaces it. Its
    // resized host pixels can coexist with the new request.
    for (const auto& image : prompt_->images)
      if (image.pixels)
        retained = CheckedAdd(retained, image.pixels->pixels.size());
  }
  const auto requested =
      replacement ? CheckedAdd(required, retained) : required;
  try {
    admission_.Grow(requested);
  } catch (const core::ResourceAdmissionError& error) {
    server::Logger::Warn(
        "vision", "event=vision_admission_refused policy=request-aware " +
                      std::string(error.what()));
    throw;
  }
  server::Logger::Debug(
      "vision",
      "event=vision_admitted policy=request-aware claim_bytes=" +
          std::to_string(admission_.Bytes()) +
          " request_bytes=" + std::to_string(required) + " retained_bytes=" +
          std::to_string(retained) + " outstanding_bytes=" +
          std::to_string(admission_budget_->Outstanding()) +
          " capacity_bytes=" + std::to_string(admission_budget_->Capacity()));
}

void DeviceInput::ReleaseImageStorage(hipStream_t stream) {
  // The stable descriptor has already been rewritten to an empty layout.
  // Drain users before dropping embeddings, then positions, then the claim.
  Check(hipStreamSynchronize(stream));
  embeddings_.clear();
  if (positions_ != nullptr)
    Check(hipFree(positions_));
  positions_ = nullptr;
  capacity_ = 0;
  admission_.Clear();
  server::Logger::Debug(
      "vision",
      "event=vision_released policy=request-aware outstanding_bytes=" +
          std::to_string(admission_budget_->Outstanding()) +
          " capacity_bytes=" + std::to_string(admission_budget_->Capacity()));
}

void DeviceInput::PrepareRope(hipStream_t stream) {
  if (descriptor_ != nullptr)
    return;
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(nullptr, &layout_);
  Check(hipMalloc(&descriptor_, sizeof(DeviceRope)));
  const DeviceRope descriptor{};
  Check(hipMemcpyAsync(descriptor_, &descriptor, sizeof(descriptor),
                       hipMemcpyHostToDevice, stream));
  Check(hipStreamSynchronize(stream));
}

void DeviceInput::Configure(std::shared_ptr<const Prompt> prompt,
                            std::shared_ptr<Encoder> encoder,
                            hipStream_t stream) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(prompt.get(), nullptr);
  if (prompt && !prompt->images.empty() && !encoder) {
    throw std::invalid_argument("image input has no vision encoder");
  }
  const bool reuse = prompt_ && prompt && !prompt->cache_identity.empty() &&
                     encoder_ == encoder &&
                     prompt_->cache_identity == prompt->cache_identity;
  if (prompt)
    prompt->rope.Validate(std::numeric_limits<std::int32_t>::max());
  if (admission_budget_ && is_cancelled_ && is_cancelled_())
    throw std::runtime_error("vision encoding cancelled");
  ReserveRequest(prompt.get(), encoder.get(),
                 prompt ? prompt->rope.PrefixLength() : 0, !reuse);
  prompt_ = std::move(prompt);
  encoder_ = std::move(encoder);
  if (!reuse)
    embeddings_.clear();
  if (prompt_)
    embeddings_.resize(prompt_->images.size());
  ApplyLayout(prompt_ ? prompt_->rope : RopeLayout{}, stream);
  if (admission_budget_ && (!prompt_ || prompt_->images.empty()) &&
      layout_.images.empty())
    ReleaseImageStorage(stream);
}

void DeviceInput::RestoreLayout(const RopeLayout& layout, hipStream_t stream) {
  layout.Validate(std::numeric_limits<std::int32_t>::max());
  if (!layout_ready_ || layout != layout_ ||
      (descriptor_ == nullptr && !layout.images.empty())) {
    // RAM and disk restores require the attached history before rebuilding
    // positions. Reserve before any layout-dependent allocation or mutation.
    ReserveRequest(prompt_.get(), encoder_.get(), layout.PrefixLength(), true);
  }
  ApplyLayout(layout, stream);
}

void DeviceInput::ApplyLayout(const RopeLayout& layout, hipStream_t stream) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(nullptr, &layout);
  layout.Validate(std::numeric_limits<std::int32_t>::max());
  if (layout_ready_ && layout == layout_ &&
      (descriptor_ != nullptr || layout.images.empty()))
    return;
  // Allocation/copy failure may leave the device descriptor ahead of the
  // host layout, or pointing at a freed growth buffer. A later reset to text
  // must rewrite it even if the host layout already appears empty.
  layout_ready_ = false;
  const auto length = layout.PrefixLength();
  if (descriptor_ == nullptr)
    Check(hipMalloc(&descriptor_, sizeof(DeviceRope)));
  if (length > capacity_) {
    Check(hipStreamSynchronize(stream));
    if (positions_ != nullptr)
      Check(hipFree(positions_));
    positions_ = nullptr;
    capacity_ = 0;
    Check(
        hipMalloc(&positions_, std::size_t{length} * 3 * sizeof(std::int32_t)));
    capacity_ = length;
  }
  std::vector<std::array<std::int32_t, 3>> positions(length);
  for (std::uint32_t token = 0; token < length; ++token)
    positions[token] = layout.Position(token);
  if (length != 0) {
    Check(hipMemcpyAsync(positions_, positions.data(),
                         positions.size() * sizeof(positions[0]),
                         hipMemcpyHostToDevice, stream));
  }
  const DeviceRope descriptor{positions_, length, layout.Delta()};
  Check(hipMemcpyAsync(descriptor_, &descriptor, sizeof(descriptor),
                       hipMemcpyHostToDevice, stream));
  Check(hipStreamSynchronize(stream));
  layout_ = layout;
  layout_ready_ = true;
}

void DeviceInput::Inject(float* hidden, std::uint32_t position,
                         std::uint32_t count, std::uint32_t width,
                         std::uint32_t hc, hipStream_t stream) {
  if (!prompt_)
    return;
  for (std::size_t index = 0; index < prompt_->images.size(); ++index) {
    const auto& image = prompt_->images[index];
    const auto image_end =
        image.grid.offset + image.grid.height * image.grid.width;
    const auto begin = std::max(position, image.grid.offset);
    const auto end = std::min(position + count, image_end);
    if (begin >= end)
      continue;
    if (!embeddings_[index])
      embeddings_[index] = encoder_->Encode(*image.pixels, {}, is_cancelled_);
    const auto& embedding = *embeddings_[index];
    if (embedding.width() != width ||
        embedding.rows() != image.grid.height * image.grid.width) {
      throw std::logic_error("vision embedding does not match language input");
    }
    const auto rows = end - begin;
    LaunchRows(
        hidden + std::size_t{begin - position} * width * hc,
        embedding.data() + std::size_t{begin - image.grid.offset} * width, rows,
        width, hc, stream);
  }
}
}  // namespace gufo::models::qwen::vision
