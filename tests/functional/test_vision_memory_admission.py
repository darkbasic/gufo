"""CPU evidence checks; these do not execute GPU qualification."""

import unittest

from vision_memory_admission import concurrent_progress


def snapshot(tasks=(1, 2, 3, 4), progress=(0, 0, 0, 0), active=4, now=0):
    return {"time": now, "slots": [
        {"id_task": task, "is_processing": i < active,
         "n_prompt_tokens_processed": progress[i], "next_token": [{"n_decoded": 0}]}
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


if __name__ == "__main__":
    unittest.main()
