import copy
import json
from pathlib import Path
import tempfile
import unittest

import watch_repair_stalls as watch


class StallPolicyTests(unittest.TestCase):
    def job(self, now=2000, old=False):
        stamp = 1000 if old else now
        files = {k: {"exists": True, "size": 20, "mtime_ns": int(stamp * 1e9), "inode": 1}
                 for k in ("prediction", "runtime", "log")}
        return {"id": "j", "status": "generating", "worker_status": "generating",
                "expected_identity": {"pid": 1, "start_ticks": 10}, "expected_argv": ["python", "worker"],
                "process": {"pid": 1, "start_ticks": 10, "argv": ["python", "worker"], "alive": True, "state": "S"},
                "files": files, "first_record": True, "started_epoch": 1000}

    def assess(self, job, memo=None, now=2000):
        return watch.assess_job(job, memo or {}, now)

    def test_healthy_quiet(self):
        self.assertEqual(self.assess(self.job())[0], [])

    def test_both_channels_stale(self):
        issues, _ = self.assess(self.job(old=True))
        self.assertEqual([x["kind"] for x in issues], ["generation_stalled"])

    def test_first_snapshot_does_not_reset_stale_age(self):
        _, memo = self.assess(self.job(old=True))
        self.assertEqual(memo["progress"]["output"]["last_change_epoch"], 1000)

    def test_log_advance_avoids_false_stall(self):
        job = self.job(old=True)
        job["files"]["log"]["mtime_ns"] = 2000 * 10**9
        self.assertEqual(self.assess(job)[0], [])

    def test_prediction_advance_avoids_false_stall(self):
        job = self.job(old=True)
        job["files"]["prediction"]["mtime_ns"] = 2000 * 10**9
        self.assertEqual(self.assess(job)[0], [])

    def test_observed_changes_reset_only_changed_channel(self):
        job = self.job(old=True)
        _, memo = self.assess(job, now=1500)
        job["files"]["log"]["size"] += 1
        issues, new = self.assess(job, memo, now=2000)
        self.assertEqual(issues, [])
        self.assertEqual(new["progress"]["output"]["last_change_epoch"], 1000)
        self.assertEqual(new["progress"]["log"]["last_change_epoch"], 2000)

    def test_loading_has_ten_minute_grace(self):
        job = self.job()
        job["first_record"] = False
        self.assertEqual(self.assess(job, now=1599)[0], [])
        self.assertEqual(self.assess(job, now=1600)[0][0]["kind"], "loading_no_first_record")

    def test_partial_first_row_is_not_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p.jsonl"
            path.write_text('{"question_id": 1}')
            self.assertFalse(watch.first_record(path))
            path.write_text('{"question_id": 1}\n{"question_id":')
            self.assertTrue(watch.first_record(path))

    def test_invalid_or_oversize_first_row_is_not_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p.jsonl"
            path.write_text("bad\n")
            self.assertFalse(watch.first_record(path))
            path.write_text(json.dumps({"text": "x" * (watch.ROW_LIMIT + 1)}) + "\n")
            self.assertFalse(watch.first_record(path))

    def test_legitimate_exit_transition(self):
        job = self.job()
        job["process"]["alive"] = False
        issues, memo = self.assess(job)
        self.assertEqual(issues, [])
        self.assertEqual(self.assess(job, memo, now=2119)[0], [])
        self.assertEqual(self.assess(job, memo, now=2120)[0][0]["kind"], "worker_missing")

    def test_completed_worker_exit_is_legal(self):
        job = self.job()
        job["status"] = "complete"
        job["process"]["alive"] = False
        self.assertEqual(self.assess(job)[0], [])

    def test_pid_reuse_is_rejected_without_signalling(self):
        job = self.job()
        job["process"]["start_ticks"] = 11
        self.assertEqual(self.assess(job)[0][0]["kind"], "worker_identity_differs")

    def test_different_argv_is_rejected(self):
        job = self.job()
        job["process"]["argv"] = ["python", "foreign_worker"]
        self.assertEqual(self.assess(job)[0][0]["kind"], "worker_identity_differs")

    def test_failed_status_requires_attention(self):
        job = self.job()
        job["worker_status"] = "failed"
        self.assertEqual(self.assess(job)[0][0]["kind"], "worker_failed")

    def test_legal_dependency_wait_has_no_generation_alarm(self):
        job = self.job(old=True)
        job["status"] = "pending"
        self.assertEqual(self.assess(job)[0], [])

    def test_events_are_change_only_and_recovery_is_once(self):
        issue = watch.make_issue("generation_stalled", "stale", "j")
        event, old = watch.events([issue], False, {})
        self.assertEqual(event[0]["kind"], "attention")
        self.assertEqual(watch.events([issue], False, {"issues": old})[0], [])
        cleared, _ = watch.events([], False, {"issues": old})
        self.assertEqual(cleared[0]["kind"], "recovered_observation")
        self.assertEqual(watch.events([], False, {"issues": {}})[0], [])

    def test_ready_event_is_once(self):
        self.assertEqual(watch.events([], True, {})[0], [{"kind": "registered_batch_ready"}])
        self.assertEqual(watch.events([], True, {"batch_ready_seen": True})[0], [])

    def test_global_gate_rejects_partial_counts_and_active_processes(self):
        groups = ["g" + str(i) for i in range(55)]
        view = {"phase2": {"status": "complete", "child_pids": [], "jobs": {"j": {"status": "complete", "returncode": 0}}},
                "dependencies": {"phase1": True}, "phase2_marker_success": True, "live_registered_processes": [],
                "publisher": {"expected_groups": groups, "published_groups": dict.fromkeys(groups), "pending_groups": []}}
        self.assertTrue(watch.batch_ready(view, groups, ["j"]))
        for mutation in (lambda x: x["phase2"].update(status="running"),
                         lambda x: x["phase2"].update(child_pids=[1]),
                         lambda x: x.update(live_registered_processes=[1]),
                         lambda x: x.update(phase2_marker_success=False),
                         lambda x: x["dependencies"].update(phase1=False),
                         lambda x: x["phase2"]["jobs"]["j"].update(returncode=1),
                         lambda x: x["publisher"].update(pending_commit="x"),
                         lambda x: x["publisher"]["published_groups"].pop(groups[0])):
            broken = copy.deepcopy(view)
            mutation(broken)
            self.assertFalse(watch.batch_ready(broken, groups, ["j"]))
        self.assertFalse(watch.batch_ready(view, groups, ["j", "missing"]))

    def test_inherited_error_and_waiting_are_not_failures(self):
        config = {"phase2_supervisor": {"pid": 1, "start_ticks": 10, "argv": ["cpu"]}}
        snapshot = {"phase2_status": "waiting_for_phase1", "supervisor": {"pid": 1, "start_ticks": 10,
                     "argv": ["cpu"], "alive": True, "state": "S"}, "jobs": [], "observed_epoch": 2000,
                    "publisher": {}, "error": "Legacy continuation supervisor interrupted"}
        self.assertEqual(watch.assess(snapshot, config, {})[0], [])

        snapshot["supervisor"]["alive"] = False
        self.assertEqual(watch.assess(snapshot, config, {})[0][0]["kind"], "supervisor_missing")
        snapshot["phase2_status"] = "complete"
        self.assertEqual(watch.assess(snapshot, config, {})[0], [])

    def test_pending_without_workers_has_grace_and_dependency_exemption(self):
        config = {"phase2_supervisor": {"pid": 1, "start_ticks": 10, "argv": ["cpu"]}}
        snapshot = {"phase2_status": "running", "supervisor": {"pid": 1, "start_ticks": 10,
                    "argv": ["cpu"], "alive": True, "state": "S"}, "jobs": [], "observed_epoch": 1000,
                    "publisher": {}, "phase2_jobs": {"pending": 2}, "dependencies": {"phase1": True}}
        issues, memo = watch.assess(snapshot, config, {})
        self.assertEqual(issues, [])
        snapshot["observed_epoch"] = 1599
        self.assertEqual(watch.assess(snapshot, config, {"jobs": memo})[0], [])
        snapshot["observed_epoch"] = 1600
        self.assertEqual(watch.assess(snapshot, config, {"jobs": memo})[0][0]["kind"], "pending_without_worker")
        snapshot["dependencies"]["phase1"] = False
        self.assertEqual(watch.assess(snapshot, config, {"jobs": memo})[0], [])
        snapshot["phase2_status"] = "waiting_for_phase1"
        snapshot["dependencies"]["phase1"] = True
        self.assertEqual(watch.assess(snapshot, config, {"jobs": memo})[0], [])


if __name__ == "__main__":
    unittest.main()
