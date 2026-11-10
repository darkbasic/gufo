#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_ADMISSION_HPP_

#include <cstddef>
#include <memory>
#include <utility>

#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

// Optional device copies compete with image work in the same startup epoch.
// The bounded mutable-state spare allowance is already part of deferred
// scratch. A local ledger assigns it to live allocations without deducting it
// from the epoch twice. Retaining a freed block retains its ticket as well.
class SnapshotAdmission {
public:
  SnapshotAdmission(std::shared_ptr<qwen::vision::AdmissionBudget> budget,
                    std::size_t protected_spare_bytes)
      : budget_(std::move(budget)),
        spares_({protected_spare_bytes, protected_spare_bytes, 0, 0}) {}

  [[nodiscard]] bool OnDemand() const noexcept {
    return budget_ && budget_->OnDemand();
  }

  [[nodiscard]] cache::ResourceReservation Reserve(std::size_t bytes,
                                                   bool mutable_state) {
    if (!OnDemand() || !bytes)
      return {};
    if (mutable_state) {
      try {
        return spares_.Reserve(cache::ResourceCategory::kPrivateState, bytes);
      } catch (const cache::ResourceExhausted&) {
        // Active captures are not bounded by the number of retained spares.
        // Additional copies need capacity from the same ledger as images.
      }
    }
    return budget_->ReserveShared(bytes);
  }

private:
  std::shared_ptr<qwen::vision::AdmissionBudget> budget_;
  cache::ResourceLedger spares_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif
