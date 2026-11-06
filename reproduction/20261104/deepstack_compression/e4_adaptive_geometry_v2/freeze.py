import json
import os
from pathlib import Path
import cpu_check

D = Path(__file__).resolve().parent


def main():
    p = json.loads((D / "cpu_proof001.json").read_text()); s = json.loads((D / "monitor_cpu001/started.json").read_text()); t = json.loads((D / "monitor_cpu001/terminal.json").read_text())
    if (p.get("status") != "passed" or p.get("tests") != 6 or p.get("source_sha256_before") != p.get("source_sha256_after") or p.get("source_sha256_after") != cpu_check.sources() or
            t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid")):
        raise ValueError("CPU proof")
    out = {"status": "frozen_before_feature_values", "source_sha256": p["source_sha256_after"], "cpu_proof_sha256": cpu_check.sha(D / "cpu_proof001.json"),
           "GT_opened": False, "score_values_opened": False, "feature_values_opened": False, "scope": "real feature extraction requires independent prelaunch review"}
    with (D / "extraction_freeze001.json").open("x") as f: json.dump(out, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
