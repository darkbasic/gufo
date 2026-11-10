#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/families/qwen/vision/admission.hpp"

namespace vision = gufo::models::qwen::vision;
using gufo::core::ResourceAdmissionError;

namespace {
void Require(bool condition, const char* description) {
  if (!condition)
    throw std::runtime_error(description);
}

template<class Error = ResourceAdmissionError, class Function>
std::string Refused(Function&& function) {
  try {
    function();
  } catch (const Error& error) {
    return error.what();
  }
  throw std::runtime_error("expected admission refusal");
}

void StartupPolicies() {
  // Historical arithmetic fixture from #920, not a host/device snapshot or
  // evidence of physical fit. The later recorded admission is tested below.
  constexpr std::size_t available = 40088076288ULL;
  // Keep the stable image-capable graph descriptor's 64-byte allowance.
  constexpr std::size_t base = 7527108948ULL;
  constexpr std::size_t worst = 10214609236ULL;
  constexpr std::size_t four_gib = 4294967296ULL;
  constexpr std::size_t margin = 512ULL << 20;
  // The issue's available state capacity already excludes deferred scratch.
  const auto conservative = vision::PlanStartupMemory(
      available + 1024, 1024, base, worst, 0, four_gib, margin);
  Require(conservative.state_capacity_bytes == available &&
              conservative.per_session_bytes == worst,
          "default reproduces the legacy full-vision state claim");
  Require(conservative.Fits(3) && !conservative.Fits(4),
          "three conservative sessions fit but four do not");
  const auto aware = vision::PlanStartupMemory(
      available + 1024, 1024, base, worst, four_gib, four_gib, margin);
  Require(aware.Fits(4) && !aware.Fits(5),
          "historical synthetic capacity admits four text/MTP states");
  Require(aware.shared_reservation_bytes == 1024 + 2 * four_gib + margin,
          "vision, full cache grant, deferred scratch and margin are distinct");
  Require(
      aware.state_capacity_bytes == 30961270784ULL,
      "issue configuration retains 4 GiB vision and 4 GiB checkpoint cache");
  const auto insufficient = vision::PlanStartupMemory(
      4 * base + four_gib - 1, 0, base, worst, four_gib, 0, 0);
  Require(!insufficient.Fits(4), "text claims must still fit");
  const auto exact = vision::PlanStartupMemory(4 * base + four_gib, 0, base,
                                               worst, four_gib, 0, 0);
  Require(exact.Fits(4), "exact startup boundary admits");
  Require(!vision::PlanStartupMemory(10, 11, base, worst, 0, 0, 0).Fits(1),
          "legacy capacity subtraction saturates");
  Refused([] { (void)vision::PlanStartupMemory(10, 0, 1, 2, 11, 0, 0); });
  const auto maximum = std::numeric_limits<std::size_t>::max();
  Refused([&] {
    (void)vision::PlanStartupMemory(maximum, 1, 1, 2, maximum, 0, 0);
  });
  Refused([&] {
    (void)vision::PlanStartupMemory(maximum, 0, 1, 2, 1, maximum, 1);
  });
  Require(!exact.Fits(maximum), "session aggregation cannot wrap");
  Refused([&] { (void)exact.RequiredBytes(maximum); });
  Refused([&] { (void)exact.DeficitBytes(maximum, maximum); });
  const vision::StartupMemoryPlan sum_overflow{maximum, maximum - 2, 3};
  Refused([&] { (void)sum_overflow.RequiredBytes(1); });
}

void RecordedStartupFailure() {
  constexpr std::size_t selected = 34672033792ULL;
  constexpr std::size_t base = 7527108948ULL;
  constexpr std::size_t worst = 10214609236ULL;
  constexpr std::size_t scratch = 418013184ULL;
  constexpr std::size_t four_gib = 4294967296ULL;
  constexpr std::size_t margin = 536870912ULL;

  // The deployment logged only the selected minimum at final admission.
  // Equal inputs reproduce that known value without inventing which limit
  // won. The earlier vision-preparation readings are NOT this snapshot.
  const auto memory = vision::SelectStartupCapacity(selected, selected);
  const auto plan = vision::PlanStartupMemory(
      memory.available_bytes, scratch, base, worst, four_gib, four_gib, margin);
  Require(plan.state_capacity_bytes == 25127215104ULL &&
              plan.shared_reservation_bytes == 9544818688ULL,
          "recorded admission subtracts each distinct uncommitted allowance");
  Require(vision::CheckedMultiply(4, base) == 30108435792ULL &&
              plan.RequiredBytes(4) == 39653254480ULL,
          "recorded full-context text states and total requirement are exact");
  Require(
      plan.Fits(3) && !plan.Fits(4) &&
          plan.DeficitBytes(4, memory.available_bytes) == 4981220688ULL,
      "actual deployment remains refused with its unchanged 4.639 GiB deficit");
  Require(plan.DeficitBytes(3, memory.available_bytes) == 0,
          "a sufficient snapshot never reports a negative deficit");

  // This is an explicit hypothetical deduction, not a CLI mode change. Zero
  // vision-memory-bytes selects conservative admission and cannot model it.
  const auto without_vision = plan.RequiredBytes(4) - four_gib;
  Require(
      without_vision > selected && without_vision - selected == 686253392ULL,
      "making the entire vision pool lazy still leaves a text-state deficit");
  const auto conservative = vision::PlanStartupMemory(
      selected, scratch, base, worst, 0, four_gib, margin);
  Require(
      conservative.per_session_bytes == worst && !conservative.Fits(4),
      "zero retains conservative policy rather than removing vision charges");
}

void StartupCapacitySelection() {
  // These small limits are synthetic and independent of the recorded final
  // limiter, which was not observable in the deployment log.
  const auto host = vision::SelectStartupCapacity(260, 240);
  const auto device = vision::SelectStartupCapacity(240, 260);
  const auto equal = vision::SelectStartupCapacity(240, 240);
  Require(host.device_admission_bytes == 260 &&
              host.host_ceiling_bytes == 240 && host.available_bytes == 240 &&
              std::string(host.LimiterName()) == "host",
          "production selection respects the smaller host ceiling");
  Require(device.available_bytes == 240 &&
              std::string(device.LimiterName()) == "device",
          "production selection respects the smaller device budget");
  Require(equal.available_bytes == 240 &&
              std::string(equal.LimiterName()) == "equal",
          "equal limits are identified without choosing an arbitrary winner");

  const auto plan = [](std::size_t available, std::size_t scratch = 10,
                       std::size_t cache = 30) {
    return vision::PlanStartupMemory(available, scratch, 40, 60, 20, cache, 20);
  };
  for (const auto capacity : {host, device, equal}) {
    const auto exact = plan(capacity.available_bytes);
    Require(exact.state_capacity_bytes == 160 && exact.Fits(4) &&
                !exact.Fits(5) && exact.RequiredBytes(4) == 240 &&
                exact.DeficitBytes(4, capacity.available_bytes) == 0,
            "exact-fit selection funds text and every uncommitted reservation");
  }
  for (const auto capacity : {vision::SelectStartupCapacity(260, 239),
                              vision::SelectStartupCapacity(239, 260)}) {
    const auto insufficient = plan(capacity.available_bytes);
    Require(!insufficient.Fits(4) &&
                insufficient.DeficitBytes(4, capacity.available_bytes) == 1,
            "either limiter one byte below requirement refuses admission");
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  Require(vision::SelectStartupCapacity(maximum, 0).available_bytes == 0 &&
              vision::SelectStartupCapacity(0, maximum).available_bytes == 0,
          "zero capacity is not replaced by the other provider's larger limit");

  // The selected free-memory snapshot already excludes loaded allocations.
  // Committing the 30-byte cache and 10-byte scratch spends free memory, and
  // only those now-committed allowances disappear from the remaining claim.
  // Their bytes are never added back to the snapshot or charged twice.
  const auto before = plan(host.available_bytes);
  const auto committed = vision::SelectStartupCapacity(260 - 40, 240 - 40);
  const auto after = plan(committed.available_bytes, 0, 0);
  Require(
      after.state_capacity_bytes == before.state_capacity_bytes &&
          after.shared_reservation_bytes == 40 && after.Fits(4) &&
          after.RequiredBytes(4) == 200,
      "already-committed allocations remain consumed in both memory limits");
  Require(before.shared_reservation_bytes == 80 &&
              before.RequiredBytes(4) - after.RequiredBytes(4) == 40,
          "uncommitted reservations stay charged until physical commitment");

  // An unrelated allocation is also consumed in the snapshot, but cannot
  // reduce any of this model's remaining reservations.
  const auto less_free = vision::SelectStartupCapacity(210, 190);
  const auto occupied = plan(less_free.available_bytes, 0, 0);
  Require(
      !occupied.Fits(4) &&
          occupied.DeficitBytes(4, less_free.available_bytes) == 10,
      "already-consumed unrelated memory cannot be credited back to admission");
}

void GeometryAndHistory() {
  const vision::ImageDimensions tiny{32, 32};
  const vision::ImageDimensions medium{1024, 1024};
  const vision::ImageDimensions largest{4096, 4096};
  Require(vision::ImageScratchBytes(tiny) == 184704,
          "32px encoder allocation oracle includes pinned upload");
  Require(vision::ImageEmbeddingBytes(tiny, 2560) == 10240,
          "one merged image token is 2560 FP32 values");
  Require(vision::ImageScratchBytes(medium) == 194150400,
          "4096 patches use the special 64-row attention tile");
  Require(vision::ImageScratchBytes(largest) == 3471376384ULL,
          "maximum encoder working set includes the capped upload buffer");
  const std::array one{tiny};
  const std::array history{tiny, medium};
  Require(vision::RequestMemoryBytes(80, one, 2560) == 199936,
          "request accounts host and device positions and embeddings");
  Require(vision::RequestMemoryBytes(1200, history, 2560) == 207824000,
          "image history sums embeddings and takes peak sequential scratch");
  Require(vision::RequestMemoryBytes(80, {}, 2560) == 1920,
          "checkpoint layout restoration needs positional storage alone");
  Require(vision::RequestMemoryBytes(0, {}, 2560) == 0,
          "text requests need no dynamic vision storage");
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({0, 32}); });
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({33, 32}); });
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({4096, 4128}); });
  Refused<std::invalid_argument>(
      [&] { (void)vision::ImageEmbeddingBytes(tiny, 0); });
  const auto maximum = std::numeric_limits<std::size_t>::max();
  Refused([&] { (void)vision::CheckedAdd(maximum, 1); });
  Refused([&] { (void)vision::CheckedMultiply(maximum, 2); });
  Refused([&] { (void)vision::RequestMemoryBytes(maximum, one, 2560); });
  Require(vision::CheckedMultiply(0, maximum) == 0 &&
              vision::CheckedAdd(maximum, 0) == maximum,
          "valid arithmetic boundaries remain valid");
}

void LifetimeAndGrowth() {
  auto budget = std::make_shared<vision::AdmissionBudget>(100);
  vision::AdmissionLease a(budget), b(budget);
  a.Grow(40);
  a.Grow(30);
  Require(a.Bytes() == 40 && budget->Outstanding() == 40,
          "smaller reused request keeps its retained allocation claim");
  a.Grow(60);
  b.Grow(40);
  Require(budget->Outstanding() == 100, "exact request boundary admits");
  const auto error = Refused([&] { a.Grow(61); });
  Require(error.find("requested_bytes=1") != std::string::npos &&
              error.find("outstanding_bytes=100") != std::string::npos &&
              error.find("capacity_bytes=100") != std::string::npos,
          "resource error reports only accounting diagnostics");
  Require(a.Bytes() == 60 && budget->Outstanding() == 100,
          "refused growth leaves all previous state unchanged");
  // Neither repeated hipMemGetInfo-like observations nor a second caller
  // admitting against the same capacity can replenish the ledger.
  for (int observation = 0; observation != 3; ++observation) {
    Require(budget->Capacity() == 100, "fixed capacity is stable");
    Refused([&] { (void)budget->Reserve(1); });
  }
  a.Clear();  // physical buffers were released by its owner
  Require(budget->Outstanding() == 40 && !a.Bytes(),
          "text transition releases its image buffers and claim");
  a.Grow(60);
  a.Clear();
  a.Clear();
  b.Clear();
  Require(!budget->Outstanding(), "repeated reset cannot double-release");

  vision::AdmissionLease conservative;
  conservative.Grow(std::numeric_limits<std::size_t>::max());
  Require(!conservative.Bytes(), "default policy does not add a runtime pool");
  {
    vision::AdmissionLease survivor(budget);
    survivor.Grow(100);
    budget.reset();
    // The ledger's handles retain accounting even after its facade is gone.
    survivor.Clear();
  }
}

void ConcurrentReservations() {
  constexpr std::size_t workers = 8;
  auto budget = std::make_shared<vision::AdmissionBudget>(100);
  std::barrier start(workers + 1);
  std::latch attempted(workers), release(1);
  std::atomic<std::size_t> admitted = 0, refused = 0;
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i != workers; ++i) {
    threads.emplace_back([&] {
      vision::AdmissionLease lease(budget);
      start.arrive_and_wait();
      try {
        lease.Grow(60);
        ++admitted;
      } catch (const ResourceAdmissionError&) {
        ++refused;
      }
      attempted.count_down();
      release.wait();
    });
  }
  start.arrive_and_wait();
  attempted.wait();
  Require(
      admitted == 1 && refused == workers - 1 && budget->Outstanding() == 60,
      "concurrent requests cannot spend the same capacity");
  release.count_down();
  for (auto& thread : threads)
    thread.join();
  Require(!budget->Outstanding(), "request destruction releases reservations");
}

// The actual AdmissionLease owns logical memory while these tiny CPU buffers
// stand in for device positions, embeddings and temporary work. Inject after
// each possible partial-allocation stage; no large physical buffers are used.
void AllocationFailuresAndCancellation() {
  auto budget = std::make_shared<vision::AdmissionBudget>(100);
  for (std::size_t fail_at = 0; fail_at != 4; ++fail_at) {
    vision::AdmissionLease lease(budget);
    std::vector<std::unique_ptr<std::array<std::byte, 4>>> allocations;
    lease.Grow(80);
    Refused<std::bad_alloc>([&] {
      for (std::size_t stage = 0; stage != 4; ++stage) {
        if (stage == fail_at)
          throw std::bad_alloc();
        allocations.push_back(std::make_unique<std::array<std::byte, 4>>());
      }
    });
    Require(budget->Outstanding() == 80,
            "allocation failure retains claim for partial device allocations");
    vision::AdmissionLease peer(budget);
    peer.Grow(20);
    Refused([&] { peer.Grow(21); });
    allocations.clear();
    lease.Clear();
    peer.Clear();
    Require(!budget->Outstanding(), "failed request cleanup restores capacity");
  }
  for (const bool cancel_after_allocation : {false, true}) {
    vision::AdmissionLease lease(budget);
    std::unique_ptr<std::array<std::byte, 4>> allocation;
    lease.Grow(100);
    if (cancel_after_allocation)
      allocation = std::make_unique<std::array<std::byte, 4>>();
    // Cancellation does not release the reservation behind live buffers.
    Require(budget->Outstanding() == 100, "cancelled state remains charged");
    allocation.reset();
    lease.Clear();
  }
  vision::AdmissionLease subsequent_text(budget);
  subsequent_text.Grow(0);
  Require(!budget->Outstanding(),
          "text remains usable after all failure stages");
}

void PreparedHistoryAndRestoredLayouts() {
  const std::array history{vision::ImageDimensions{32, 32},
                           vision::ImageDimensions{64, 32}};
  const auto peak = vision::RequestMemoryBytes(120, history, 2560);
  auto budget = std::make_shared<vision::AdmissionBudget>(peak);
  vision::AdmissionLease state(budget);
  state.Grow(peak);
  const auto before = budget->Outstanding();
  // RAM checkpoint reuse keeps the same request attachment. Disk restore
  // reconstructs only RoPE layout; both use Grow and cannot release the
  // existing embedding claim just because the restored prefix is shorter.
  state.Grow(peak);
  state.Grow(vision::RequestMemoryBytes(80, {}, 2560));
  Require(budget->Outstanding() == before,
          "RAM reuse and shorter restored layout retain attached image claims");
  Refused([&] {
    state.Grow(vision::CheckedAdd(
        peak, vision::RequestMemoryBytes(160, history, 2560)));
  });
  Require(budget->Outstanding() == peak,
          "replacement peak refusal preserves old request buffers");
  state.Clear();
  // A fresh slot restoring a disk layout pays its complete positional bound;
  // subsequent image reattachment grows before encoding or continuation.
  state.Grow(vision::RequestMemoryBytes(80, {}, 2560));
  state.Grow(peak);
  Require(budget->Outstanding() == peak,
          "fresh restore followed by image history cannot bypass admission");
  state.Clear();
}

void PreparationLifetimes() {
  const auto scratch = vision::kPreparationScratchBytes;
  const auto pixels = vision::kPreparedImageBytes;
  const auto peak = scratch + 2 * pixels;
  auto budget = std::make_shared<vision::AdmissionBudget>(peak);
  {
    vision::PreparationReservation text(budget, 0);
    vision::PreparationReservation conservative(nullptr, 1);
    Require(budget->Outstanding() == 0,
            "text/default preparation creates no new reservation");
  }
  {
    auto preparation =
        std::make_shared<vision::PreparationReservation>(budget, 2);
    Require(budget->Outstanding() == peak,
            "all history pixels and transient preprocessing fit exactly");
    Refused([&] { vision::PreparationReservation peer(budget, 1); });
    preparation->Finish();
    preparation->Finish();
    Require(budget->Outstanding() == 2 * pixels,
            "successful preparation releases scratch only and only once");
    auto queued = preparation;
    auto session_history = preparation;
    preparation.reset();
    queued.reset();  // Cancellation drops one request owner.
    Require(budget->Outstanding() == 2 * pixels,
            "cancelled queued request cannot release live image history");
    session_history.reset();
    Require(budget->Outstanding() == 0,
            "last prepared-pixel owner returns its capacity");
  }
  {
    auto too_small = std::make_shared<vision::AdmissionBudget>(peak - 1);
    Refused([&] { vision::PreparationReservation request(too_small, 2); });
    Require(too_small->Outstanding() == 0,
            "failure of the second reservation rolls back transient scratch");
  }
  for (const bool finished : {false, true}) {
    Refused<std::bad_alloc>([&] {
      vision::PreparationReservation request(budget, 2);
      // Tiny storage stands in for decoded/prepared pixels. Its lifetime
      // precedes reservation destruction under cancellation or exceptions.
      auto allocation = std::make_unique<std::array<std::byte, 4>>();
      if (finished)
        request.Finish();
      throw std::bad_alloc();
    });
    Require(budget->Outstanding() == 0,
            "allocation/cancellation failure returns every preparation claim");
  }
  Refused([&] {
    vision::PreparationReservation overflow(
        budget, std::numeric_limits<std::size_t>::max());
  });
  Require(budget->Outstanding() == 0,
          "image-count overflow cannot leak claims");

  std::barrier start(3);
  std::latch attempted(2), release(1);
  std::atomic<unsigned> admitted{0}, refused{0};
  const auto prepare = [&] {
    std::shared_ptr<vision::PreparationReservation> reservation;
    start.arrive_and_wait();
    try {
      reservation = std::make_shared<vision::PreparationReservation>(budget, 2);
      ++admitted;
    } catch (const ResourceAdmissionError&) {
      ++refused;
    }
    attempted.count_down();
    release.wait();
  };
  std::thread a(prepare), b(prepare);
  start.arrive_and_wait();
  attempted.wait();
  Require(admitted == 1 && refused == 1 && budget->Outstanding() == peak,
          "concurrent preprocessing cannot spend the same capacity");
  release.count_down();
  a.join();
  b.join();
  Require(budget->Outstanding() == 0,
          "preprocessing cancellation returns the shared quota");
}
}  // namespace

int main() {
  StartupPolicies();
  RecordedStartupFailure();
  StartupCapacitySelection();
  GeometryAndHistory();
  LifetimeAndGrowth();
  ConcurrentReservations();
  AllocationFailuresAndCancellation();
  PreparedHistoryAndRestoredLayouts();
  PreparationLifetimes();
  std::cout << "vision admission checks passed\n";
}
