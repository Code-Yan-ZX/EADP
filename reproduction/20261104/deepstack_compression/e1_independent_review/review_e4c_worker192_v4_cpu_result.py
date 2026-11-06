"""Independent retained-container review; no 888 target outputs or GPU access."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent; B = R.parent; D = B / "e4_c_multibudget_worker192_v4"; P = D / "run_preflight001"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def terminal(mon, script, out=None):
    s, t = read(mon / "started.json"), read(mon / "terminal.json"); cwd = Path(s["cwd"]); cmd = s["command"]
    if (t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid") or
            [(cwd / x).resolve() for x in cmd if Path(x).name == script.name] != [script.resolve()]): raise ValueError("supervisor identity or exit")
    if out is not None and (cmd.count("--out") != 1 or (cwd / cmd[cmd.index("--out") + 1]).resolve() != out.resolve()): raise ValueError("output identity")
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    out = R / "e4c_worker192_v4_cpu_result_review.json"
    if a.out.resolve() != out or out.exists(): raise ValueError("unexpected output")
    plan = read(D / "cpu_plan001.json"); cpu = read(D / "cpu_proof001.json"); temp = R / "e4c_worker192_v4_cpu_preflight_review.json"; temp_script = R / "review_e4c_worker192_v4_cpu_preflight.py"
    terminal(D / "monitor_cpu001", D / "freeze_cpu.py"); terminal(R / "monitor_e4c_worker192_v4_cpu_preflight", temp_script, temp); pre = read(temp)
    if (cpu.get("status") != "passed" or cpu.get("source_sha256") != plan.get("source_sha256") or cpu.get("plan_sha256") != sha(D / "cpu_plan001.json") or
            pre.get("status") != "passed" or pre.get("independent_review") is not True or pre.get("reviewer_sha256") != sha(temp_script) or pre.get("plan_sha256") != sha(D / "cpu_plan001.json") or
            pre.get("source_sha256") != plan["source_sha256"] or pre.get("errors") != [] or any(sha(p) != h for p, h in plan["source_sha256"].items())): raise ValueError("CPU/source/prelaunch identity")
    mon = D / "monitor_preflight001"; s = terminal(mon, D / "launch.py")
    launch, ex, ident, worker, proc = [read(P / name) for name in ("launch.json", "exit.json", "container_identity.json", "worker_preflight.json", "processor_preflight.json")]
    if (s.get("worker_pid") != launch["controller"]["pid"] or s.get("timeout_seconds") != 900 or s.get("grace_seconds") != 90 or
            s["command"].count("--preflight-only") != 1 or launch.get("worker_sha256") != sha(D / "worker.py") or launch.get("launcher_sha256") != sha(D / "launch.py") or
            launch.get("cpu_plan_sha256") != sha(D / "cpu_plan001.json") or launch.get("run_out") != str(P) or launch.get("timeout_seconds") != 780 or
            launch.get("approval_bundle", {}).get("review_sha256") != sha(temp) or ex.get("controller") != launch["controller"] or ex.get("container_identity") != ident or
            ex.get("returncode") != 0 or ex.get("container_exit_code") != 0 or ex.get("timed_out") is not False or ex.get("interrupted") is not None or
            ex.get("client_exited_while_container_running") is not False or ex.get("container_stopped_verified") is not True): raise ValueError("actual execution/exit")
    cmd = launch["command"]
    if "--gpus" in cmd or "--device" in cmd or "--rm" in cmd or "CUDA_VISIBLE_DEVICES=" not in cmd: raise ValueError("preflight exposed GPU")
    cid = ident["container_id"]
    if len(cid) != 64 or cid not in ident["host_cgroup"] or not ident["host_init"]["start_ticks"] or ident["host_argv"][-4:] != [str(D / "worker.py"), "--out", str(P), "--preflight-only"]: raise ValueError("container identity")
    doc = json.loads(subprocess.run(["docker", "inspect", cid], check=True, capture_output=True, text=True, timeout=15).stdout)[0]; state, host = doc["State"], doc["HostConfig"]
    if (doc["Id"] != cid or doc["Image"] != ident["image"] or state["Running"] or state["Pid"] != 0 or state["ExitCode"] != 0 or state["OOMKilled"] or
            host["NetworkMode"] != "none" or host["Devices"] or host["DeviceRequests"] or "CUDA_VISIBLE_DEVICES=" not in doc["Config"]["Env"] or
            doc["Config"]["Cmd"] != [str(D / "worker.py"), "--out", str(P), "--preflight-only"]): raise ValueError("actual Docker state")
    if (worker.get("status") != "passed" or (worker.get("inputs"), worker.get("jobs"), worker.get("existing_natural_references")) != (222, 7920, 888) or
            worker.get("fixed32_new_references_opened") is not False or worker.get("GT_opened") is not False or worker.get("scoring_performed") is not False or
            worker.get("cuda_initialized") is not False or worker.get("model_loaded") is not False or worker.get("GPU_authorized") is not False or
            worker.get("plan_sha256") != sha(D / "cpu_plan001.json") or worker.get("processor_preflight_sha256") != sha(P / "processor_preflight.json") or
            proc.get("status") != "passed" or proc.get("inputs_checked") != 222 or proc.get("references_checked") != 888 or len(proc.get("processor_inputs", [])) != 222 or
            proc.get("weight_payloads_hashed") != 4 or len(proc.get("weight_stats", {})) != 4 or proc.get("cuda_initialized") is not False or proc.get("model_loaded") is not False): raise ValueError("CPU preflight content")
    prop = read(B / "e4_c_multibudget_expansion192/proposal002.json"); rows = prop["measurement_identities"] + prop["warmup_identities"]
    identity = {(x["task"], str(x["qid"]), x["group_id"]): x for x in rows}; actual = {(x["task"], str(x["qid"]), x["group_id"]): x for x in proc["processor_inputs"]}
    if len(identity) != 222 or len(actual) != 222 or set(identity) != set(actual) or any(actual[k]["processor"] != x["processor_reference"] or actual[k]["image_start"] != x["image_start_reference"] for k, x in identity.items()): raise ValueError("processor/proposal mismatch")
    fields = {"worker_preflight_sha256": P / "worker_preflight.json", "processor_preflight_sha256": P / "processor_preflight.json", "launch_sha256": P / "launch.json", "exit_sha256": P / "exit.json", "started_sha256": mon / "started.json", "terminal_sha256": mon / "terminal.json", "reviewer_sha256": Path(__file__)}
    result = {"status": "passed", "independent_review": True, "scope": "worker192_cpu_preflight_result", "errors": [], "GT_opened": False, "model_loaded": False, "cuda_initialized": False, "GPU_authorized": False,
              "inputs": 222, "existing_natural_references": 888, "fixed32_new_references_opened": False, "container_id": cid,
              "limitations": ["only no-GPU preflight; GPU plan, observer, reference result and formal acceptance are not authorized", "processor metadata compared against accepted proposal; no full model tensor reconstruction", "weight stat/hashes at preflight cannot exclude later transient replacement"]}
    result.update({k: sha(v) for k, v in fields.items()})
    with out.open("x") as f: json.dump(result, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
