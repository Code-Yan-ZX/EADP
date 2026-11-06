"""Complete synthetic denominator and paired-statistic negative controls."""
import copy
import unittest

import metrics


def synthetic():
    jobs, events = [], []
    def one(phase, task, group, budget, protocol, rep):
        job = {"phase": phase, "task": task, "qid": group, "group_id": group, "budget": budget, "protocol": protocol, "replicate": rep}
        ids = list(range(32)) if protocol == "fixed32" else [1]
        values = {"task": task, "qid": group, "group_id": group, "budget": budget, "protocol": protocol, "output_ids": ids,
                  "generation_scalars": {}, "min_new_tokens": 32 if protocol == "fixed32" else 0, "eos_ids": [2]}
        t0 = 1000; first = t0 + budget * 1000000; complete = first + 60000000
        raw = {**values, "generation_complete_ns": complete}
        record = {**values, "eos": protocol == "natural", "truncated": False, "boundaries": {"t0_before_image_read_ns": t0,
                  "first_token_selected_ns": first, "generation_complete_ns": complete, "first_criteria_length": 1,
                  "criteria_calls": len(ids), "clock_implementation": "clock_gettime(CLOCK_MONOTONIC)",
                  "explicit_sync_boundaries": ["before_t0", "first_token", "generate_complete"]}}
        jobs.append(job)
        events.extend([{"event": "job_started", "job": job}, {"event": "raw_generation", "job": job, "raw": raw}, {"event": "job_completed", "job": job, "record": record}])
    for n in range(240): one("warmup", "DocVQA_VAL", f"warm{n}", 128, "natural", 0)
    for task in metrics.TASKS:
        for n in range(64):
            for budget in metrics.KS:
                for protocol in metrics.PROTOCOLS:
                    for rep in range(5): one("measure", task, f"g{n:03}", budget, protocol, rep)
    return jobs, events


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.jobs, cls.events = synthetic()

    def test_complete_and_paired_exact(self):
        old = metrics.B; metrics.B = 32
        try: result = metrics.summarize(self.jobs, self.events)
        finally: metrics.B = old
        self.assertEqual((result["jobs"], result["measured"], result["images"]), (7920, 7680, 192))
        x = result["tasks"]["DocVQA_VAL"]["fixed32"]["512"]["ttft_ms"]
        self.assertEqual(x["time"]["median"], 512)
        self.assertEqual(x["paired_difference_vs_k256_ms"]["median"], 256)
        self.assertEqual(x["paired_ratio_vs_k256"]["median"], 2)
        self.assertEqual(x["paired_difference_vs_k256_ms"]["bootstrap_ci95"], [256, 256])

    def test_incomplete_and_raw_mismatch_rejected(self):
        with self.assertRaises(ValueError): metrics.summarize(self.jobs[:-1], self.events[:-3])
        events = list(self.events); changed = copy.deepcopy(events[1]); changed["raw"]["qid"] = "wrong"; events[1] = changed
        with self.assertRaises(ValueError): metrics.summarize(self.jobs, events)

    def test_duplicate_replicate_rejected(self):
        jobs = list(self.jobs); events = list(self.events); index = 241
        changed = dict(jobs[index]); changed["replicate"] = 0; jobs[index] = changed
        for off in range(3): events[index * 3 + off] = {**events[index * 3 + off], "job": changed}
        with self.assertRaises(ValueError): metrics.summarize(jobs, events)


if __name__ == "__main__": unittest.main()
