"""Append a bounded health snapshot; never stop or signal processes."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path(__file__).resolve().parent.parent
PROJECT = Path("/media/disk2/YZX/research/EADP_v2")


def run(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=12)
        return {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"error": repr(e)}


def main():
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(); t0 = time.monotonic_ns()
    commands = {"gpu_compute": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                "gpu": ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                "docker": ["docker", "ps", "--format", "{{.Names}} {{.Status}}"], "vmstat": ["vmstat", "1", "2"]}
    detail = {k: run(v) for k, v in commands.items()}
    detail["meminfo"] = Path("/proc/meminfo").read_text(); detail["memory_pressure"] = Path("/proc/pressure/memory").read_text()
    detail["disk_free"] = {str(p): shutil.disk_usage(p)._asdict() for p in (Path("/"), PROJECT)}
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if len(argv) > 1 and (argv[1].endswith("/timing_supervise.py") or argv[1].endswith("/supervise.py")):
                cwd = os.readlink(proc / "cwd")
                if cwd.startswith(str(PROJECT)): found.append({"pid": int(proc.name), "cwd": cwd, "argv": argv})
        except (OSError, PermissionError): pass
    detail["project_supervisors"] = found
    record = {"event": "hourly_health_check", "start_utc": start, "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "start_monotonic_ns": t0, "end_monotonic_ns": time.monotonic_ns(), "time_namespace": os.readlink("/proc/self/ns/time"), "details": detail}
    with (BASE / "hourly_checks.jsonl").open("a") as f: f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
    print("end", record["end_utc"], "gpu", detail["gpu"].get("stdout", "").strip(), "compute", detail["gpu_compute"].get("stdout", "").strip(), "supervisors", len(found))
    print("pressure", detail["memory_pressure"].strip(), "disk_free", {k: v["free"] for k, v in detail["disk_free"].items()})
    print("meminfo", " ".join(x.strip() for x in detail["meminfo"].splitlines() if x.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:"))))
    print("vmstat_last", detail["vmstat"].get("stdout", "").strip().splitlines()[-1] if detail["vmstat"].get("stdout") else "unavailable")


if __name__ == "__main__": main()
