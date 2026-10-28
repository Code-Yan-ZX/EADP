#!/usr/bin/env python3
"""Deferred, zero-interference AnchorZip Figure-1 replay watcher.

Waits until the declared Qwen3 GPU queue is fully finished
(/tmp/p4_p2_done.marker) — or demonstrably dead with the GPU idle — then
waits for a stable idle window (3 consecutive 60s checks with mem<8GB and
util<20%) and runs, in order:

  1. export_anchorzip_figure_sample_v2.py on the 5 new candidates
     (priority: TextVQA_VAL_787, OCRBench_403, then 1536/430/517)
  2. fig1_metrics.py on the exported samples
  3. assemble_fig1_v2.py  -> anchorzip_fig1_new_candidates_20261005.zip

It never touches the GPU while any queue driver/process is alive, so the
main experiment chain (P1 -> P4 -> P2) is never contended.  Logs to
fig1_v2_watcher.log in this directory.
"""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/dell/miniconda3/envs/qwen3vl_clean/bin/python"
MARKER = "/tmp/p4_p2_done.marker"
SPEC = "TextVQA_VAL:787,OCRBench:403,TextVQA_VAL:1536,DocVQA_VAL:430,OCRBench:517"
QUEUE_PAT = ("egather_legacy|egather_p1_driver|egather_p4_p2_driver|"
             "egather_p2_perf|acu_bank|acu_run|rtg_accuracy")


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def gpu_used_util() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    m, u = out.split("\n")[0].split(",")
    return int(m), int(u)


def gpu_idle_stable(n: int = 3) -> bool:
    for _ in range(n):
        m, u = gpu_used_util()
        if m >= 8000 or u >= 20:
            return False
        time.sleep(60)
    return True


def queue_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", QUEUE_PAT], capture_output=True,
                       text=True)
    return bool(r.stdout.strip())


def main() -> None:
    log(f"watcher start (marker={MARKER}, spec={SPEC})")
    while True:
        if os.path.exists(MARKER):
            log("queue marker found — queue complete")
            break
        if not queue_alive():
            m, u = gpu_used_util()
            if m < 8000 and u < 20:
                log("no queue processes and GPU idle — queue appears dead")
                break
        time.sleep(300)
    while not gpu_idle_stable(3):
        log("GPU not stably idle yet; waiting")
        time.sleep(120)
    log("GPU stably idle — starting frozen replay (5 candidates)")
    r = subprocess.run([PY, "export_anchorzip_figure_sample_v2.py",
                        "--spec", SPEC], cwd=HERE)
    log(f"exporter rc={r.returncode}")
    for sid in [s.replace(":", "_") for s in SPEC.split(",")]:
        subprocess.run([PY, "fig1_metrics.py", sid], cwd=HERE)
        log(f"fig1_metrics {sid} done")
    r = subprocess.run([PY, "assemble_fig1_v2.py"], cwd=HERE)
    log(f"assembler rc={r.returncode}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
