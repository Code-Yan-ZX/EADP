#!/usr/bin/env python3
"""CPU-only once-through finalizer; never launches or retries generation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(record):
    temporary = OUT / "CPU_finalizer_state.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(OUT / "CPU_finalizer_state.json")


def main():
    assert not (OUT / "CPU_finalizer_state.json").exists(), "No automatic restart"
    identity = {"pid": os.getpid(), "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])}
    comparator = OUT / "compare_completed.py"
    record = {"status": "waiting_full_control_verification", "created_utc": now(), "identity": identity,
              "GPU_loaded": False, "generation_started_by_this_finalizer": False,
              "automatic_retries": False, "comparator_sha256": sha(comparator),
              "finalizer_sha256": sha(__file__), "protocol_sha256": sha(OUT / "protocol.json")}
    save(record)
    print(json.dumps(record), flush=True)
    while True:
        failure = OUT / "supervisor_failure.json"
        if failure.exists():
            raise RuntimeError("Supervisor failed; preserve all outputs and do not retry")
        try:
            state = json.loads((OUT / "supervisor_state.json").read_text())
        except json.JSONDecodeError:
            time.sleep(2)
            continue
        if state["status"] == "control_failed":
            raise RuntimeError("Control failed; preserve all outputs and do not retry")
        if state["status"] == "complete":
            assert state["exit_code"] == 0
            verified = json.loads((OUT / "independent_verification.json").read_text())
            assert verified["success"] and verified["n_rows"] == 4377 and verified["n_circular_groups"] == 1176
            assert sha(OUT / "independent_verification.json") == state["independent_verification_sha256"]
            break
        time.sleep(5)
    assert sha(comparator) == record["comparator_sha256"]
    assert sha(OUT / "protocol.json") == record["protocol_sha256"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with (OUT / "paired_comparison.log").open("w") as stream:
        code = subprocess.run([sys.executable, str(comparator)], env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    assert code == 0, "CPU comparison failed; preserve output, no automatic retry"
    report = json.loads((OUT / "paired_full_comparison.json").read_text())
    assert report["success"] and report["n_rows"] == 4377 and report["n_circular_groups"] == 1176
    assert all(item["original_worker_exited"] for item in report["worker_exit"])
    record.update(status="complete", finished_utc=now(), comparison_path=str(OUT / "paired_full_comparison.json"),
                  comparison_sha256=sha(OUT / "paired_full_comparison.json"), GPU_slot_B_released=True)
    save(record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save({"status": "terminal_failure_no_retry", "error": repr(error), "time_utc": now(),
              "GPU_loaded": False, "generation_started_by_this_finalizer": False, "automatic_retries": False})
        raise
