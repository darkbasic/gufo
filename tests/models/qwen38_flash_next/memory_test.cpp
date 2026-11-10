#include "src/models/qwen38_flash_next/memory.hpp"

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>

#include "src/models/qwen38_flash_next/snapshot_admission.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::SessionMode;
namespace vision = gufo::models::qwen::vision;

namespace {
void Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

qfn::Config TargetConfig() {
  qfn::Config c;
  c.num_layers = 48;
  c.hidden_size = 2560;
  c.hc_count = 4;
  c.full_attention_interval = 4;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}

void RecordedComponents() {
  // Fixed byte oracle for the reported full-context, seven-draft-token model.
  // No HIP type, model weights or multi-GiB physical allocation is involved.
  auto plan = qfn::PlanSessionMemory(TargetConfig(), SessionMode::kSpeculative,
                                     262144, 4224, 8, 8, 7, 20, 2720);
  Require(plan.trunk_kv_bytes == 6442450944ULL, "twelve full-context KV pairs");
  Require(plan.trunk_index_bytes == 251661312, "trunk raw and pooled indexes");
  Require(plan.recurrent_bytes == 118038528, "linear and PLE mutable state");
  Require(plan.rollback_bytes == 153953280, "seven exact rollback frontiers");
  Require(plan.mtp_kv_bytes == 536870912, "predictor retains its own KV");
  Require(plan.mtp_index_bytes == 20971776, "predictor raw and pooled index");
  Require(plan.mtp_hidden_bytes == 368640,
          "eight trunk rows plus predictor row");
  Require(plan.mtp_draft_table_bytes == 2793472, "private Q8_0 draft table");
  Require(plan.control_bytes == 20 && plan.vision_descriptor_bytes == 0,
          "executor control excludes the model-owned vision descriptor");
  plan.vision_descriptor_bytes = 64;
  Require(plan.TotalBytes() == 7527108948ULL,
          "production components reproduce the recorded base claim exactly");
  Require(4 * plan.TotalBytes() == 30108435792ULL,
          "four target sessions preserve the recorded total");
}

void ModesAndGrowth() {
  auto c = TargetConfig();
  const auto ar = qfn::PlanSessionMemory(c, SessionMode::kAutoregressive,
                                         262144, 4224, 1, 0, 0, 20, 2720);
  Require(ar.TotalBytes() == 6812150804ULL,
          "AR has no predictor, draft table, hidden verification or rollback");
  Require(ar.rollback_bytes == 0 && ar.mtp_kv_bytes == 0 &&
              ar.mtp_index_bytes == 0 && ar.mtp_hidden_bytes == 0 &&
              ar.mtp_draft_table_bytes == 0,
          "loaded predictor weights do not make an AR slot speculative");
  const auto narrow = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                             262144, 4224, 8, 8, 7, 20, 2720);
  const auto wide = qfn::PlanSessionMemory(c, SessionMode::kSpeculative, 262144,
                                           4224, 8, 64, 7, 20, 2720);
  Require(wide.TotalBytes() - narrow.TotalBytes() == 2293760 &&
              wide.rollback_bytes == narrow.rollback_bytes,
          "wide verification only grows private retained hidden rows here");
  const auto no_subset = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                                262144, 4224, 8, 8, 7, 20, 0);
  Require(no_subset.mtp_draft_table_bytes == 0 &&
              narrow.TotalBytes() - no_subset.TotalBytes() == 2793472,
          "full-vocabulary drafts do not allocate the private subset table");
  c.ple_layer = -1;
  const auto no_ple = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                             262144, 4224, 8, 8, 7, 20, 2720);
  Require(narrow.TotalBytes() - no_ple.TotalBytes() == 2949120,
          "PLE current state and seven rollback histories are counted once");
  Require(qfn::SessionIndexerCapacity(c, 4224, 262144) == 8192 &&
              qfn::SessionIndexerCapacity(c, 16, 1024) == 1024,
          "indexer retention follows production ring bounds");
  Require(
      qfn::GdnRollbackRowFloats(0, 16, 48, 128) == 786432 &&
          qfn::GdnRollbackRowFloats(1, 16, 48, 128) == 8288,
      "first frontier is full state, subsequent frontiers are exact deltas");
}

template<class Action>
void Refused(Action action, std::string_view message) {
  try {
    action();
  } catch (const gufo::core::ResourceAdmissionError&) {
    return;
  }
  Require(false, message);
}

void SnapshotAdmissionOwnership() {
  // The protected 600 includes a 200-byte mutable-spare allowance, text,
  // future cache backing and scratch. None of that is physically allocated.
  auto budget = std::make_shared<vision::AdmissionBudget>(1000, 600);
  qfn::SnapshotAdmission snapshots(budget, 200);
  auto image = budget->Reserve(250);
  auto first = snapshots.Reserve(100, true);
  auto second = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 250,
          "two mutable blocks use protected scratch without a second charge");
  auto third = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 350,
          "additional live mutable copies compete with images");
  Refused([&] { (void)snapshots.Reserve(51, false); },
          "aggregate image and private copies cannot exceed the epoch");
  auto extra = snapshots.Reserve(50, false);
  Require(budget->AccountedBytes() == 1000, "exact aggregate boundary fits");
  Refused([&] { (void)budget->Reserve(1); },
          "images cannot take an admitted private snapshot's bytes");
  auto retained = third.Convert();
  auto spare = std::move(retained);
  Require(budget->Outstanding() == 400,
          "moving a released allocation into retained spares keeps its charge");
  extra = {};
  Require(budget->Outstanding() == 350,
          "only the genuinely released allocation returns capacity");
  auto reused = std::move(spare);
  Require(budget->Outstanding() == 350,
          "reusing retained storage neither releases nor reserves twice");
  first = {};
  auto replacement = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 350,
          "freed protected allocation makes its existing allowance reusable");
  reused = {};
  Require(
      budget->Outstanding() == 250,
      "freeing the optional physical allocation releases its shared ticket");
  Refused(
      [&] {
        (void)snapshots.Reserve(std::numeric_limits<std::size_t>::max(), false);
      },
      "overflow-sized optional copies fail without changing ownership");
  Require(budget->Outstanding() == 250,
          "failed growth leaves existing images and spare claims intact");
}

void SnapshotAdmissionFailuresAndModes() {
  auto budget = std::make_shared<vision::AdmissionBudget>(100, 60);
  qfn::SnapshotAdmission snapshots(budget, 20);
  for (const bool mutable_state : {false, true}) {
    try {
      auto pending = snapshots.Reserve(20, mutable_state);
      // Inject failure after admission, before allocation/conversion.
      throw std::bad_alloc();
    } catch (const std::bad_alloc&) {
    }
    Require(budget->Outstanding() == 0,
            "failed allocation rolls back its unmaterialized ticket");
  }
  auto protected_block = snapshots.Reserve(20, true);
  auto active = snapshots.Reserve(40, true);
  Refused([&] { (void)snapshots.Reserve(1, false); },
          "failure and retry preserve the protected-spare bound");
  active = {};
  auto copy = snapshots.Reserve(40, false).Convert();
  Require(budget->Outstanding() == 40,
          "successful conversion retains capacity through asynchronous use");
  copy = {};
  Require(budget->Outstanding() == 0,
          "completion and release restore capacity for later requests");
  auto image = budget->Reserve(40);
  Refused([&] { (void)snapshots.Reserve(21, true); },
          "changing mutable size cannot enlarge the protected spare allowance");
  qfn::SnapshotAdmission conservative(nullptr, 20);
  auto fixed = std::make_shared<vision::AdmissionBudget>(1);
  qfn::SnapshotAdmission fixed_snapshots(fixed, 20);
  Require(!conservative.Reserve(1000, false) &&
              !fixed_snapshots.Reserve(1000, true) && !fixed->Outstanding(),
          "conservative and fixed-pool snapshot behavior remains unchanged");
}

void ConcurrentSnapshotAdmission() {
  auto budget = std::make_shared<vision::AdmissionBudget>(100, 80);
  qfn::SnapshotAdmission snapshots(budget, 0);
  std::barrier rendezvous(3);
  std::atomic<unsigned> admitted{}, refused{};
  const auto contender = [&](bool image) {
    rendezvous.arrive_and_wait();
    gufo::cache::ResourceReservation reservation;
    try {
      reservation = image ? budget->Reserve(20) : snapshots.Reserve(20, false);
      ++admitted;
    } catch (const gufo::core::ResourceAdmissionError&) {
      ++refused;
    }
    // Hold the winner's reservation until both admission attempts completed.
    rendezvous.arrive_and_wait();
  };
  std::thread image(contender, true);
  std::thread copy(contender, false);
  rendezvous.arrive_and_wait();
  rendezvous.arrive_and_wait();
  image.join();
  copy.join();
  Require(
      admitted == 1 && refused == 1 && budget->Outstanding() == 0,
      "concurrent images and private snapshots cannot spend the same bytes");
}
}  // namespace

int main() {
  RecordedComponents();
  ModesAndGrowth();
  SnapshotAdmissionOwnership();
  SnapshotAdmissionFailuresAndModes();
  ConcurrentSnapshotAdmission();
  std::cout << "Flash-Next production memory components passed\n";
}
