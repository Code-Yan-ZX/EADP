#!/usr/bin/env python3
"""CPU-only waiting supervisor for root's conditional slot-B allocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from patch_only import OUT, BASE

PYTHON = "/home/dell/miniconda3/envs/llava_pruner/bin/python"
POPE = BASE.parent / "pope/run02_source_wait_only_k32"
PREDECESSOR_PID = 2334667
PREDECESSOR_TICKS = 1331274836
AUTHORIZATION = (
    "Trusted parent /root explicitly preallocated subsequent slot B to full MMB three-wait control. "
    "Launch only after POPE run02_source_wait_only_k32 completes all8910, unchanged author scoring and "
    "independent full_result_summary succeed, worker2334667/startticks1331274836 actually exits, and frozen "
    "source SHAs remain unchanged. Slot A remains root's Qwen slot; max2 models including loading. "
    "Any terminal failure must be preserved/reported without automatic rerun or taking another slot."
)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "start_ticks": int(fields[19]), "session_id": os.getsid(pid), "process_group": os.getpgid(pid)}


def registration():
    ready = json.loads((OUT / "READY.json").read_text())
    assert ready["status"] == "prepared_not_started" and ready["protocol_sha256"] == sha(OUT / "protocol.json")
    plan = json.loads((OUT / "protocol.json").read_text())
    pope_plan = json.loads((POPE / "run_plan.json").read_text())
    source = [{"path": item["path"], "sha256": item["sha256"]} for item in plan["source_files"]]
    source += [{"path": str(OUT / "protocol.json"), "sha256": sha(OUT / "protocol.json")},
               {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
               {"path": str(POPE / "run_plan.json"), "sha256": sha(POPE / "run_plan.json")}]
    source += [{"path": str(Path(pope_plan["source_root"]) / relative), "sha256": digest}
               for relative, digest in pope_plan["source_files"].items()]
    record = {"registered_utc": now(), "authorization": AUTHORIZATION,
              "assigned_GPU_slot": "B", "predecessor_pid": PREDECESSOR_PID,
              "predecessor_start_ticks": PREDECESSOR_TICKS, "supervisor_identity": identity(os.getpid()),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "frozen_files": source, "dataset": plan["dataset"], "no_accuracy_threshold": True,
              "launch_command": [PYTHON, str(OUT / "run_control.py"), "run", "--allocation",
                                 "root_slot_B_after_POPE_run02_2334667_1331274836"]}
    return record


def check_frozen(record):
    for item in record["frozen_files"]:
        assert sha(item["path"]) == item["sha256"], "Frozen source changed: " + item["path"]


def predecessor_gate(record):
    state = json.loads((POPE / "state.json").read_text())
    status = state["status"]
    failures = {"generation_failed", "score_failed", "completion_gate_failed", "blocked_by_source_baseline_failure"}
    if status in failures or status.endswith("_failed"):
        raise RuntimeError("Predecessor terminal failure: " + status)
    if status != "complete":
        return False, {"reason": "predecessor_" + status}
    assert state["generation_exit_code"] == state["score_exit_code"] == 0
    assert state["n_predictions"] == 8910
    live = identity(PREDECESSOR_PID)
    if live and live["start_ticks"] == PREDECESSOR_TICKS:
        return False, {"reason": "predecessor_worker_still_alive", "identity": live}
    summary_path = POPE / "full_result_summary.json"
    release_path = POPE / "completion_and_release_proof.json"
    if not summary_path.exists() or not release_path.exists():
        return False, {"reason": "waiting_independent_full_summary_and_exit_proof"}
    summary, release = json.loads(summary_path.read_text()), json.loads(release_path.read_text())
    assert summary["run_id"] == POPE.name and summary["n_predictions"] == 8910
    assert summary["official_scorer_exit_code"] == 0 and summary["all_8910_question_ids_order_verified"] is True
    assert summary["gpu_worker_exit_verified"] is True and all(summary["source_sha_unchanged_checks"].values())
    assert release["status"] == "all_completion_and_exit_gates_passed"
    assert release["worker_pid"] == PREDECESSOR_PID and release["worker_start_ticks"] == PREDECESSOR_TICKS
    assert not release["same_worker_still_alive"] and release["no_more_POPE_models_scheduled"]
    assert release["full_result_summary"]["sha256"] == sha(summary_path)
    raw_path = POPE / "predictions.jsonl"
    raw_sha = sha(raw_path)
    assert raw_sha == state["predictions_sha256"] == summary["predictions_sha256"] == release["predictions_sha256"]
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    pope_plan = json.loads((POPE / "run_plan.json").read_text())
    question_file = Path(pope_plan["generation_command"][pope_plan["generation_command"].index("--question-file") + 1])
    questions = [json.loads(line) for line in question_file.read_text().splitlines() if line.strip()]
    assert len(raw) == len(questions) == 8910
    assert [row["question_id"] for row in raw] == [row["question_id"] for row in questions]
    assert all(answer["prompt"] == question["text"] for answer, question in zip(raw, questions))
    for category, expected_n in (("random", 2910), ("popular", 3000), ("adversarial", 3000)):
        values = summary["categories"][category]
        assert values["n"] == expected_n == sum(values[key] for key in ("TP", "FP", "TN", "FN"))
        computed = 200 * values["TP"] / (2 * values["TP"] + values["FP"] + values["FN"])
        assert math.isclose(computed, values["f1_percent"], abs_tol=1e-12, rel_tol=0)
    independent = sum(values["f1_percent"] for values in summary["categories"].values()) / 3
    official_log = POPE / "official_score.log"
    official = 100 * float(re.search(r"Average F1 score: ([\d.eE+-]+)", official_log.read_text())[1])
    assert math.isclose(independent, summary["independent_macro_f1_percent"], abs_tol=1e-12, rel_tol=0)
    assert math.isclose(official, independent, abs_tol=1e-12, rel_tol=0)
    check_frozen(record)
    assert sha(record["dataset"]["path"]) == record["dataset"]["sha256"]
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    if len([line for line in apps.splitlines() if line.strip()]) >= 2:
        return False, {"reason": "two_GPU_processes_still_occupy_capacity", "GPU_processes": apps}
    return True, {"reason": "all_root_launch_gates_passed", "predecessor_state_sha256": sha(POPE / "state.json"),
                  "predecessor_summary_sha256": sha(summary_path), "release_proof_sha256": sha(release_path),
                  "predecessor_predictions_sha256": raw_sha, "official_score_log_sha256": sha(official_log),
                  "original_predecessor_worker_alive": False, "current_pid_identity_if_reused": live,
                  "all_8910_ids_and_prompts_verified": True, "official_and_independent_macro_equal": True,
                  "source_SHA_unchanged": True, "MMB_protocol_sha256": sha(OUT / "protocol.json"),
                  "GPU_processes_before_launch": apps}


def main(check_only):
    record = registration()
    check_frozen(record)
    if check_only:
        passed, evidence = predecessor_gate(record)
        print(json.dumps({"CPU_only": True, "would_launch": False, "gate_passed": passed, "gate": evidence}, indent=2))
        return
    assert not (OUT / "supervisor_registration.json").exists(), "Do not restart a registered controller automatically"
    save("supervisor_registration.json", record)
    state = {"status": "waiting_predecessor", "identity": identity(os.getpid()), "registered_utc": now(),
             "authorization": AUTHORIZATION, "GPU_loaded_by_supervisor": False}
    save("supervisor_state.json", state)
    print(json.dumps(state), flush=True)
    last_reason = None
    while True:
        passed, evidence = predecessor_gate(record)
        if evidence["reason"] != last_reason:
            state.update(gate=evidence, observed_utc=now())
            save("supervisor_state.json", state)
            print(json.dumps({"event": "gate", **evidence}), flush=True)
            last_reason = evidence["reason"]
        if passed:
            break
        time.sleep(5)
    check_frozen(record)
    save("launch_gate_evidence.json", {"authorization": AUTHORIZATION, "verified_utc": now(), **evidence})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    with (OUT / "supervised_run.log").open("w") as log:
        controller = subprocess.Popen(record["launch_command"], cwd=BASE.parents[2], env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="control_started", launch_utc=now(), run_controller_identity=identity(controller.pid))
        save("supervisor_state.json", state)
        print(json.dumps({"event": "control_started", "run_controller_identity": state["run_controller_identity"]}), flush=True)
        announced = False
        prediction = OUT / "v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl"
        while controller.poll() is None:
            native_state_path = OUT / "state.json"
            if native_state_path.exists():
                native_state = json.loads(native_state_path.read_text())
                if "native_process" in native_state and "native_process" not in state:
                    state["native_process"] = native_state["native_process"]
                    save("supervisor_state.json", state)
                    print(json.dumps({"event": "native_worker_started", "identity": state["native_process"]}), flush=True)
            if not announced and prediction.exists() and prediction.stat().st_size:
                with prediction.open() as stream:
                    first = json.loads(next(stream))
                state["first_prediction"] = first
                save("supervisor_state.json", state)
                print(json.dumps({"event": "first_prediction", "question_id": first["question_id"], "text": first["text"]}), flush=True)
                announced = True
            time.sleep(2)
    code = controller.wait()
    state.update(status="complete" if code == 0 else "control_failed", exit_code=code, finished_utc=now())
    if code == 0:
        verified = json.loads((OUT / "independent_verification.json").read_text())
        assert verified["success"] and verified["n_rows"] == 4377
        state["independent_verification_sha256"] = sha(OUT / "independent_verification.json")
    save("supervisor_state.json", state)
    print(json.dumps(state), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.check_only)
    except Exception as error:
        save("supervisor_failure.json", {"status": "terminal_failure_no_retry", "time_utc": now(),
                                         "error": repr(error), "identity": identity(os.getpid()), "automatic_retry": False})
        raise
