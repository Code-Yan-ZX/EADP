"""Record full hourly health and a conservative monotonic manual window."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manual-log", type=Path, required=True); p.add_argument("--reason", required=True); a = p.parse_args()
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(); start = time.monotonic_ns()
    try:
        r = subprocess.run([sys.executable, str(BASE / "monitor/hourly_health_check_eadp.py")], capture_output=True, text=True, timeout=30)
        detail = {"reason": a.reason, "health_exit_code": r.returncode, "health_stdout": r.stdout, "health_stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail = {"reason": a.reason, "health_error": str(exc)}
    end = time.monotonic_ns(); end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {"event": "manual_health_check", "time_namespace": os.readlink("/proc/self/ns/time"), "start_monotonic_ns": start, "end_monotonic_ns": end,
              "start_utc": start_utc, "end_utc": end_utc, "details": detail}
    with a.manual_log.open("a") as f: f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
    print(detail.get("health_stdout", ""), end="")
    if "health_error" in detail or detail.get("health_exit_code") != 0: raise SystemExit(1)


if __name__ == "__main__": main()
