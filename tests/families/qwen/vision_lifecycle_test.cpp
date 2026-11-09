#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/core/logging/logging.hpp"
#include "src/families/qwen/vision/device_input.hpp"

namespace vision = gufo::models::qwen::vision;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class Error = std::runtime_error, class Function>
void Refused(Function&& function) {
  try {
    function();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error("expected refusal");
}
}  // namespace

namespace fake {
std::mutex mutex;
std::unordered_map<void*, std::size_t> allocations;
std::weak_ptr<vision::AdmissionBudget> budget;
std::atomic<std::size_t> encodes{0};
// Fail one selected API call after n successful calls of the same kind.
thread_local std::optional<std::size_t> fail_malloc, fail_copy, fail_sync;
thread_local bool fail_encode_after_allocation{false};
bool Fail(std::optional<std::size_t>& remaining) {
  if (!remaining)
    return false;
  if (*remaining) {
    --*remaining;
    return false;
  }
  remaining.reset();
  return true;
}
std::size_t Bytes() {
  const std::lock_guard lock(mutex);
  std::size_t bytes = 0;
  for (const auto& [pointer, count] : allocations)
    bytes += count;
  return bytes;
}
void CheckClaim(std::size_t bytes) {
  if (bytes == sizeof(vision::DeviceRope))
    return;  // Stable descriptor belongs to the base text-session claim.
  if (const auto owner = budget.lock())
    Require(owner->Outstanding() >= bytes,
            "physical allocation/release must remain covered by its claim");
}
}  // namespace fake

hipError_t hipMalloc(void** pointer, std::size_t bytes) {
  if (fake::Fail(fake::fail_malloc))
    return hipErrorMemoryAllocation;
  fake::CheckClaim(bytes);
  auto storage = std::make_unique<std::byte[]>(bytes);
  const std::lock_guard lock(fake::mutex);
  fake::allocations.emplace(storage.get(), bytes);
  *pointer = storage.release();
  return hipSuccess;
}
hipError_t hipFree(void* pointer) {
  if (!pointer)
    return hipSuccess;
  const std::lock_guard lock(fake::mutex);
  const auto found = fake::allocations.find(pointer);
  Require(found != fake::allocations.end(), "double or unowned device free");
  fake::CheckClaim(found->second);
  delete[] static_cast<std::byte*>(pointer);
  fake::allocations.erase(found);
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind, hipStream_t) {
  // Model the difficult case: the descriptor may have changed even when the
  // copy reports failure. Recovery cannot trust only the previous host layout.
  std::memcpy(destination, source, bytes);
  return fake::Fail(fake::fail_copy) ? hipErrorUnknown : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  return fake::Fail(fake::fail_sync) ? hipErrorUnknown : hipSuccess;
}
const char* hipGetErrorString(hipError_t error) {
  return error == hipErrorMemoryAllocation ? "injected allocation failure"
                                           : "injected copy/sync failure";
}

namespace gufo::models::qwen::vision {
// The real DeviceInput .cpp, including Inject's ownership loop, is compiled
// unchanged. Only the weight-dependent encoder and row-copy kernel are fake.
struct Encoder::Impl {
  std::uint32_t width;
};
Encoder::Encoder(const std::filesystem::path&, std::uint32_t width)
    : impl_(std::make_unique<Impl>(Impl{width})) {}
Encoder::~Encoder() = default;
std::uint32_t Encoder::OutputWidth() const noexcept {
  return impl_->width;
}
Encoder::Embedding::~Embedding() {
  (void)hipFree(data_);
}
std::shared_ptr<const Encoder::Embedding> Encoder::Encode(
    const core::Image& image, const Observer&,
    const CancellationCheck& cancelled) {
  ++fake::encodes;
  if (cancelled && cancelled())
    throw std::runtime_error("injected encoder cancellation");
  auto result = std::shared_ptr<Embedding>(new Embedding);
  result->rows_ = image.width / 32 * (image.height / 32);
  result->width_ = impl_->width;
  const auto bytes =
      std::size_t{result->rows_} * result->width_ * sizeof(float);
  if (hipMalloc(&result->data_, bytes) != hipSuccess)
    throw std::bad_alloc();
  std::fill_n(result->data_, bytes / sizeof(float), 7.0F);
  if (fake::fail_encode_after_allocation) {
    fake::fail_encode_after_allocation = false;
    throw std::bad_alloc();
  }
  return result;
}
void DeviceInput::LaunchRows(float* hidden, const float* embedding,
                             std::uint32_t rows, std::uint32_t width,
                             std::uint32_t hc, hipStream_t) {
  for (std::size_t i = 0; i < std::size_t{rows} * width * hc; ++i)
    hidden[i] = embedding[(i / (width * hc)) * width + i % width];
}
}  // namespace gufo::models::qwen::vision

namespace {
constexpr std::uint32_t kWidth = 2560;
std::shared_ptr<vision::Prompt> Prompt(std::uint32_t offset = 2,
                                       std::uint32_t side = 32) {
  auto image = std::make_shared<gufo::core::Image>();
  image->width = side;
  image->height = side;
  image->pixels.resize(std::size_t{side} * side * 3);
  auto prompt = std::make_shared<vision::Prompt>();
  const vision::ImageGrid grid{offset, side / 32, side / 32};
  prompt->images.push_back({std::move(image), grid, {}});
  prompt->rope.images.push_back(grid);
  prompt->cache_identity = {static_cast<std::uint8_t>(offset),
                            static_cast<std::uint8_t>(side)};
  return prompt;
}
std::size_t Peak(const vision::Prompt& prompt) {
  std::vector<vision::ImageDimensions> images;
  for (const auto& image : prompt.images)
    images.push_back({image.pixels->width, image.pixels->height});
  return vision::RequestMemoryBytes(prompt.rope.PrefixLength(), images, kWidth);
}
void Inject(vision::DeviceInput& input, std::uint32_t offset,
            std::uint32_t rows = 1) {
  std::vector<float> hidden(std::size_t{rows} * kWidth);
  input.Inject(hidden.data(), offset, rows, kWidth, 1, nullptr);
  Require(hidden.front() == 7.0F, "fake row launcher received encoded image");
}
void EmptyDescriptor(const vision::DeviceInput& input) {
  Require(input.rope() && input.rope()->prefix_length == 0 &&
              input.rope()->delta == 0,
          "text transition repairs and empties stable device descriptor");
  Require(input.Bytes() == sizeof(vision::DeviceRope),
          "text transition releases every dynamic device buffer");
}

void RetentionAndTextTransitions() {
  const auto prompt = Prompt();
  auto budget = std::make_shared<vision::AdmissionBudget>(Peak(*prompt));
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.PrepareRope(nullptr);
    const auto* descriptor = input.rope();
    input.Configure(prompt, encoder, nullptr);
    Require(budget->Outstanding() == Peak(*prompt),
            "complete prepared request reserved before position allocation");
    Inject(input, 2);
    const auto encodes = fake::encodes.load();
    const auto bytes = input.Bytes();
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
    Require(input.Bytes() == bytes && fake::encodes == encodes,
            "identical image history reuses its retained embedding");
    Require(input.rope() == descriptor,
            "image requests preserve graph descriptor address");
    Refused<gufo::core::ResourceAdmissionError>(
        [&] { input.Configure(Prompt(20, 64), encoder, nullptr); });
    Require(input.PromptAttachment() == prompt && input.Bytes() == bytes,
            "oversized replacement leaves existing image state intact");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "text frees image memory and reservation");
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
  }
  Require(!budget->Outstanding() && !fake::Bytes(),
          "session destruction frees buffers before reservations");
  fake::budget.reset();
  // Default mode retains its historical high-water positions on text reuse.
  {
    vision::DeviceInput conservative;
    conservative.Configure(prompt, encoder, nullptr);
    conservative.Configure(nullptr, encoder, nullptr);
    Require(conservative.Bytes() > sizeof(vision::DeviceRope),
            "conservative default allocation retention is unchanged");
  }
  Require(!fake::Bytes(), "default session destruction releases buffers");
}

void GrowthAndRestoredHistory() {
  auto budget = std::make_shared<vision::AdmissionBudget>(2U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto small = Prompt();
  const auto large = Prompt(2, 64);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(small, encoder, nullptr);
    Inject(input, 2);
    const auto initial = budget->Outstanding();
    input.Configure(large, encoder, nullptr);
    Inject(input, 2, 4);
    Require(budget->Outstanding() > initial,
            "growth charges replacement peak before releasing old buffers");
    const auto* rope = input.rope();
    Require(
        rope->prefix_length == 6 && rope->delta == -2 &&
            rope->positions[12] == 2 && rope->positions[13] == 3 &&
            rope->positions[14] == 2 && rope->positions[15] == 2 &&
            rope->positions[16] == 3 && rope->positions[17] == 3,
        "production position upload preserves independent 2x2 mRoPE oracle");
    const auto retained = budget->Outstanding();
    input.RestoreLayout(large->rope, nullptr);
    Require(budget->Outstanding() == retained,
            "RAM reuse of the same layout retains the complete image claim");
    input.RestoreLayout({}, nullptr);
    Require(budget->Outstanding() >= retained && input.Bytes() > sizeof(*rope),
            "restoring an earlier frontier does not uncharge retained images");
    input.RestoreLayout(large->rope, nullptr);
    const auto encodes = fake::encodes.load();
    Inject(input, 2, 4);
    Require(fake::encodes == encodes,
            "restored history preserves already encoded image ownership");
    input.Configure(nullptr, encoder, nullptr);
  }
  // A fresh slot pays the complete attached history before a cache restores
  // positions. RAM/disk component restoration both call this production path.
  {
    vision::DeviceInput restored;
    restored.SetMemoryAdmission(budget);
    restored.Configure(large, encoder, nullptr);
    restored.RestoreLayout(large->rope, nullptr);
    Require(budget->Outstanding() == Peak(*large),
            "fresh checkpoint attachment charges all future image work");
    Inject(restored, 2, 4);
  }
  Require(!fake::Bytes() && !budget->Outstanding(),
          "restored sessions leave neither allocations nor reservations");
}

void AllocationAndCopyFailures() {
  auto budget = std::make_shared<vision::AdmissionBudget>(2U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto prompt = Prompt();
  for (unsigned failure = 0; failure != 6; ++failure) {
    {
      vision::DeviceInput input;
      input.SetMemoryAdmission(budget);
      if (failure < 2)
        fake::fail_malloc = failure;
      else if (failure < 4)
        fake::fail_copy = failure - 2;
      else
        fake::fail_sync = failure - 4;
      Refused([&] { input.Configure(prompt, encoder, nullptr); });
      Require(budget->Outstanding() == Peak(*prompt),
              "partial allocation/copy failure keeps complete reservation");
      input.Configure(nullptr, encoder, nullptr);
      EmptyDescriptor(input);
      Require(
          !budget->Outstanding(),
          "failed descriptor/position stages recover to text without leaks");
      input.Configure(prompt, encoder, nullptr);
      Inject(input, 2);
    }
    Require(!fake::Bytes() && !budget->Outstanding(),
            "failed stage does not corrupt the following image request");
  }
  for (const bool after_allocation : {false, true}) {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(prompt, encoder, nullptr);
    if (after_allocation)
      fake::fail_encode_after_allocation = true;
    else
      fake::fail_malloc = 0;
    Refused<std::bad_alloc>([&] { Inject(input, 2); });
    Require(budget->Outstanding() == Peak(*prompt),
            "encoder failure cannot release a live position claim");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "encoder failures recover to text");
  }
  Require(!fake::Bytes(), "encoder failure destroys temporary embeddings");
}

void Cancellation() {
  auto budget = std::make_shared<vision::AdmissionBudget>(1U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto prompt = Prompt();
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.SetCancellationCheck([] { return true; });
    Refused([&] { input.Configure(prompt, encoder, nullptr); });
    Require(!budget->Outstanding() && !fake::Bytes(),
            "cancelled admission allocates no device state");
    input.SetCancellationCheck({});
    input.Configure(prompt, encoder, nullptr);
    input.SetCancellationCheck([] { return true; });
    Refused([&] { Inject(input, 2); });
    Require(budget->Outstanding() == Peak(*prompt),
            "cancellation during encoding retains existing position claim");
    input.SetCancellationCheck({});
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
  }
  Require(!budget->Outstanding() && !fake::Bytes(),
          "cancelled request cleanup");
}

void ConcurrentInputs() {
  const auto prompt = Prompt();
  auto budget = std::make_shared<vision::AdmissionBudget>(Peak(*prompt));
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  std::barrier start(3);
  std::latch attempted(2), release(1);
  std::atomic<unsigned> admitted{0}, refused{0};
  auto request = [&] {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    start.arrive_and_wait();
    try {
      input.Configure(prompt, encoder, nullptr);
      ++admitted;
    } catch (const gufo::core::ResourceAdmissionError&) {
      ++refused;
    }
    attempted.count_down();
    release.wait();
  };
  std::thread first(request), second(request);
  start.arrive_and_wait();
  attempted.wait();
  Require(
      admitted == 1 && refused == 1 && budget->Outstanding() == Peak(*prompt),
      "actual concurrent Configure paths cannot oversubscribe the pool");
  release.count_down();
  first.join();
  second.join();
  Require(!fake::Bytes() && !budget->Outstanding(),
          "concurrent input lifetimes release exactly once");
}
}  // namespace

int main() {
  gufo::server::Logger::SetLevel(gufo::server::LogLevel::kError);
  RetentionAndTextTransitions();
  GrowthAndRestoredHistory();
  AllocationAndCopyFailures();
  Cancellation();
  ConcurrentInputs();
  std::cout << "vision production lifecycle checks passed (simulated HIP)\n";
}
