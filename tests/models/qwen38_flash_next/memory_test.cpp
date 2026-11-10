#include "src/models/qwen38_flash_next/memory.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::SessionMode;

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
}  // namespace

int main() {
  RecordedComponents();
  ModesAndGrowth();
  std::cout << "Flash-Next production memory components passed\n";
}
