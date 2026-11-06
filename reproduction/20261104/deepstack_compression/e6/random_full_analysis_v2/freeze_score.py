"""Freeze complete scorer while full random results remain unopened."""
import json
import os
from pathlib import Path

import cpu_check

D = Path(__file__).resolve().parent


def main():
    p = json.loads((D / "cpu_proof001.json").read_text())
    s = json.loads((D / "monitor_cpu001/started.json").read_text()); t = json.loads((D / "monitor_cpu001/terminal.json").read_text())
    if (p.get("status") != "passed" or p.get("tests", 0) < 9 or p.get("GT_opened") is not False or p.get("full_panel_predictions_opened") is not False or
            p.get("source_sha256_before") != p.get("source_sha256_after") or p.get("source_sha256_after") != cpu_check.sources() or
            t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid")):
        raise ValueError("CPU/source proof invalid")
    plan = {"status": "frozen", "source_sha256": p["source_sha256_after"], "cpu_proof_sha256": cpu_check.sha(D / "cpu_proof001.json"),
            "GT_opened": False, "full_panel_predictions_opened": False, "scope": "score only after complete formal and independent no-GT result acceptance and independent scoring prelaunch"}
    with (D / "score_plan001.json").open("x") as f: json.dump(plan, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
