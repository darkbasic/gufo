"""CPU evidence checks; these do not execute GPU qualification."""

import unittest
from unittest.mock import patch

from vision_memory_admission import concurrent_progress, mixed_progress, run_mixed_phase


def snapshot(tasks=(1, 2, 3, 4), progress=(0, 0, 0, 0), active=4, now=0,
             decoded=(0, 0, 0, 0)):
    return {"time": now, "slots": [
        {"id_task": task, "is_processing": i < active,
         "n_prompt_tokens_processed": progress[i], "next_token": [{"n_decoded": decoded[i]}]}
        for i, task in enumerate(tasks)]}


class ConcurrentExecutionEvidence(unittest.TestCase):
    def test_requires_four_held_sessions(self):
        with self.assertRaisesRegex(RuntimeError, "four execution sessions"):
            concurrent_progress([snapshot(active=3), snapshot(progress=(8, 8, 8, 8), active=3)])

    def test_admission_without_execution_is_not_evidence(self):
        with self.assertRaises(RuntimeError):
            concurrent_progress([snapshot(), snapshot()])

    def test_every_held_request_must_advance(self):
        with self.assertRaises(RuntimeError):
            concurrent_progress([snapshot(), snapshot(progress=(8, 8, 8, 0))])

    def test_reassignment_does_not_fake_progress(self):
        with self.assertRaises(RuntimeError):
            concurrent_progress([snapshot(), snapshot(tasks=(5, 6, 7, 8), progress=(8, 8, 8, 8))])

    def test_four_distinct_tasks_advance(self):
        result = concurrent_progress([snapshot(), snapshot(progress=(4, 0, 0, 0)),
                                      snapshot(progress=(4, 4, 4, 4), now=3)])
        self.assertEqual(result["task_ids"], (1, 2, 3, 4))
        self.assertEqual(result["sample_time"], 3)

    def test_duplicate_task_ids_cannot_qualify(self):
        with self.assertRaises(RuntimeError):
            concurrent_progress([snapshot(tasks=(1, 1, 1, 1)),
                                 snapshot(tasks=(1, 1, 1, 1), progress=(8, 8, 8, 8))])


class MixedExecutionEvidence(unittest.TestCase):
    def test_mapped_image_and_both_text_peers_advance(self):
        result = mixed_progress([
            snapshot(active=3, now=1),
            snapshot(active=3, progress=(4, 0, 4, 0), decoded=(0, 0, 1, 0), now=2),
            snapshot(active=3, progress=(4, 4, 4, 0), decoded=(0, 0, 1, 0), now=3),
        ], (1, 2, 3), 3)
        self.assertEqual(result["start_time"], 1)
        self.assertEqual(result["sample_time"], 3)
        self.assertEqual(result["after"][3], (4, 1))

    def test_prefill_before_image_consumption_cannot_qualify(self):
        self.assertIsNone(mixed_progress([
            snapshot(active=3), snapshot(active=3, progress=(8, 8, 8, 0)),
        ], (1, 2, 3), 3))

    def test_queued_or_sequential_image_cannot_qualify(self):
        with self.assertRaisesRegex(RuntimeError, "ended or were reassigned"):
            mixed_progress([snapshot(active=2), snapshot(active=3, decoded=(1, 1, 1, 0))],
                           (1, 2, 3), 3)

    def test_stationary_text_peer_cannot_qualify(self):
        self.assertIsNone(mixed_progress([
            snapshot(active=3), snapshot(active=3, decoded=(1, 0, 1, 0)),
        ], (1, 2, 3), 3))

    def test_unmapped_image_identity_cannot_qualify(self):
        with self.assertRaisesRegex(RuntimeError, "observed scheduler identity"):
            mixed_progress([], (1, 2, 3), 4)

    def test_reassigned_peer_cannot_supply_post_refusal_progress(self):
        with self.assertRaisesRegex(RuntimeError, "ended or were reassigned"):
            mixed_progress([snapshot(active=3),
                            snapshot(tasks=(1, 5, 3, 4), active=3, decoded=(1, 1, 1, 0))],
                           (1, 2, 3), 3)

    def test_gap_cannot_stitch_sequential_execution_into_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "ended or were reassigned"):
            mixed_progress([snapshot(active=3), snapshot(active=2),
                            snapshot(active=3, decoded=(1, 1, 1, 0))], (1, 2, 3), 3)

    def test_prefill_increase_does_not_hide_decode_counter_reset(self):
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            mixed_progress([snapshot(active=3, decoded=(1, 1, 1, 0)),
                            snapshot(active=3, progress=(8, 8, 8, 0))], (1, 2, 3), 3)

    def test_duplicate_scheduler_id_cannot_qualify(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate scheduler task IDs"):
            mixed_progress([snapshot(tasks=(1, 1, 3, 4), active=3)], (1, 2, 3), 3)


class FakeFuture:
    def __init__(self, function, args, completed=False):
        self.function, self.args, self.completed = function, args, completed

    def done(self):
        return self.completed

    def result(self, timeout=None):
        return self.function(*self.args)


class FakeExecutor:
    """Drive orchestration deterministically; no network, threads or sleeps."""
    def __init__(self, completed=False):
        self.completed, self.submitted, self.closed = completed, [], False

    def submit(self, function, *args):
        self.submitted.append(args)
        return FakeFuture(function, args, self.completed)

    def shutdown(self, *, wait, cancel_futures):
        self.closed = not wait and cancel_futures


class MixedPhaseOrchestration(unittest.TestCase):
    def run_phase(self, records, *, completed=False, refusal_code="resource_exhausted",
                  failure_label=None):
        self.evidence, self.calls = {}, []
        self.executor = FakeExecutor(completed)
        observations = iter(records)

        def chat(label, body, expected_status=200):
            self.calls.append((label, expected_status))
            if label == failure_label:
                raise RuntimeError(f"{label}: HTTP failure")
            return {"error": {"code": refusal_code}} if expected_status == 503 else {"choices": [{}]}

        with patch("vision_memory_admission.ThreadPoolExecutor", return_value=self.executor):
            run_mixed_phase(chat, lambda: next(observations), [{"text": 1}, {"text": 2}],
                            {"image": True}, {"oversized": True}, {"recovery": True},
                            60, self.evidence, pause=lambda _: None)

    @staticmethod
    def records():
        # Deliberately nonconsecutive IDs; only staged admission establishes
        # which request owns each ID. The fourth execution slot stays free.
        def row(active, progress=0, decoded=0, now=0):
            return snapshot(tasks=(13, 29, 41, -1), active=active,
                            progress=(progress,) * 4, decoded=(decoded,) * 4, now=now)
        return [row(0), row(1), row(2), row(3), row(3),
                row(3, 4, 1, 1), row(3, 4, 1, 2), row(3, 8, 2, 3)]

    def test_refusal_is_bracketed_by_same_tasks_and_fresh_progress(self):
        self.run_phase(self.records())
        self.assertEqual(self.evidence["task_ids"], {
            "mixed-text-0": 13, "mixed-text-1": 29, "mixed-image": 41})
        self.assertEqual(self.evidence["before_refusal"]["after"][41], (4, 1))
        self.assertEqual(self.evidence["after_refusal"]["before"][41], (4, 1))
        self.assertEqual(self.evidence["after_refusal"]["after"][41], (8, 2))
        self.assertLessEqual(self.evidence["refusal_started"], self.evidence["refusal_finished"])
        self.assertEqual(self.calls, [("oversized-image", 503), ("mixed-text-0", 200),
                                     ("mixed-text-1", 200), ("mixed-image", 200),
                                     ("text-after-refusal", 200)])
        self.assertEqual(self.evidence["status"], "passed")
        self.assertTrue(self.executor.closed)

    def test_early_success_does_not_replace_overlap_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "completed too early"):
            self.run_phase(self.records(), completed=True)
        self.assertTrue(self.executor.closed)
        self.assertNotIn(("text-after-refusal", 200), self.calls)

    def test_unknown_active_request_prevents_attribution(self):
        records = self.records()
        records[1] = snapshot(tasks=(13, 29, 41, -1), active=2)
        with self.assertRaisesRegex(RuntimeError, "cannot attribute"):
            self.run_phase(records)
        self.assertTrue(self.executor.closed)

    def test_refusal_that_disrupts_peer_cannot_pass(self):
        records = self.records()
        records[-2] = snapshot(tasks=(13, 99, 41, -1), active=3)
        with self.assertRaisesRegex(RuntimeError, "ended or were reassigned"):
            self.run_phase(records)
        self.assertNotIn(("text-after-refusal", 200), self.calls)
        self.assertTrue(self.executor.closed)

    def test_other_error_is_not_a_resource_refusal(self):
        with self.assertRaisesRegex(RuntimeError, "explicit resource refusal"):
            self.run_phase(self.records(), refusal_code="internal_error")
        self.assertTrue(self.executor.closed)

    def test_image_must_finish_successfully_after_observed_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "mixed-image: HTTP failure"):
            self.run_phase(self.records(), failure_label="mixed-image")
        self.assertEqual(self.evidence["status"], "failed")
        self.assertIn("HTTP failure", self.evidence["failure"])
        self.assertNotIn(("text-after-refusal", 200), self.calls)
        self.assertTrue(self.executor.closed)


if __name__ == "__main__":
    unittest.main()
