"""Independent retained-container and source review before random full GPU run."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / "e6/random_full_exec_v2"
R = B / "e1_independent_review"
OUT = R / "random_full_v2_prelaunch_review.json"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def done(mon, script):
    s, t = read(mon / "started.json"), read(mon / "terminal.json")
    assert t["reason"] == "completed" and t["worker_returncode"] == 0 and t["remaining_process_group"] == [] and s["worker_pid"] == t["worker_pid"]
    assert sum((Path(s["cwd"]) / x).resolve() == script.resolve() for x in s["command"]) == 1
    return s, t


def main():
    assert not OUT.exists() and not (D / "run_001").exists()
    plan = read(D / "runtime_plan001.json"); vp = read(D / "verification_plan001.json")
    assert plan["stage"] == "random_full_dev" and plan["questions"] == 384 and plan["generations"] == 1152
    assert plan["GT_opened"] is False and plan["scoring_performed"] is False and plan["GPU_authorized"] is False
    assert vp["runtime_plan_sha256"] == sha(D / "runtime_plan001.json") and vp["verifier_sha256"] == sha(D / "verify.py")
    assert vp["contract_sha256"] == sha(D / "verify_contract.py") and vp["snapshot_sha256"] == sha(D / "snapshot.py")
    assert all(sha(p) == h for p, h in plan["source_sha256"].items())
    done(D / "monitor_cpu001", D / "cpu_check.py"); done(D / "monitor_freeze001", D / "freeze.py")
    done(R / "monitor_random_full_v2_worker_preflight", R / "review_random_full_v2_worker_preflight.py")
    first = read(R / "random_full_v2_worker_preflight_review.json")
    assert first["status"] == "passed" and first["scope"] == "CPU worker preflight only" and first["runtime_plan_sha256"] == sha(D / "runtime_plan001.json")
    assert first["reviewer_sha256"] == sha(R / "review_random_full_v2_worker_preflight.py") and first["errors"] == []
    s, t = done(D / "monitor_preflight001", D / "launch.py")
    assert s["command"].count("--preflight-only") == 1 and s["timeout_seconds"] == plan["preflight_outer_seconds"]
    out = D / "run_preflight001"; launch = read(out / "launch.json"); ex = read(out / "exit.json"); ident = read(out / "container_identity.json"); pf = read(out / "worker_preflight.json")
    assert launch["controller"]["pid"] == s["worker_pid"] and launch["approval_bundle"]["preflight_review_sha256"] == sha(R / "random_full_v2_worker_preflight_review.json")
    assert launch["approval_bundle"]["plan_sha256"] == sha(D / "runtime_plan001.json") and launch["worker_sha256"] == sha(D / "worker.py") and launch["launcher_sha256"] == sha(D / "launch.py")
    assert launch["timeout_seconds"] == plan["preflight_inner_seconds"] and launch["run_out"] == str(out)
    cmd = launch["command"]
    assert "--gpus" not in cmd and "--device" not in cmd and "--rm" not in cmd and "CUDA_VISIBLE_DEVICES=" in cmd and cmd[-1] == "--preflight-only"
    assert pf["status"] == "passed" and pf["questions"] == 384 and pf["choices"] == 1152 and pf["GT_opened"] is False and pf["model_loaded"] is False and pf["cuda_initialized"] is False
    assert ex["returncode"] == 0 and ex["container_exit_code"] == 0 and ex["container_stopped_verified"] is True and ex["timed_out"] is False and ex["interrupted"] is None
    assert ex["client_exited_while_container_running"] is False and ex["container_identity"] == ident
    cid = ident["container_id"]; assert len(cid) == 64 and cid in ident["host_cgroup"] and ident["host_init"]["start_ticks"] > 0
    expected_args = [str(D / "worker.py"), "--out", str(out), "--preflight-only"]
    assert ident["host_argv"][-4:] == expected_args
    doc = json.loads(subprocess.run(["docker", "inspect", cid], check=True, capture_output=True, text=True, timeout=15).stdout)[0]
    host, state = doc["HostConfig"], doc["State"]
    assert doc["Id"] == cid and doc["Image"] == ident["image"] and ident["image"] in cmd
    assert state["Running"] is False and state["Pid"] == 0 and state["ExitCode"] == 0 and state["OOMKilled"] is False
    assert host["NetworkMode"] == "none" and not host["DeviceRequests"] and host["Devices"] == [] and host["PidMode"] == ""
    assert "ALL" in host["CapDrop"] and "no-new-privileges" in host["SecurityOpt"]
    assert doc["Config"]["Cmd"] == expected_args and "CUDA_VISIBLE_DEVICES=" in doc["Config"]["Env"]
    # This validates the actual no-GPU execution chain. The four weight shards are
    # re-hashed inside prepare->source_and_weights; the preflight stores only its
    # successful completion, not separate per-shard post-run digest observations.
    worker = (D / "worker.py").read_text(); latency = (B / "e4_c_latency/latency_worker.py").read_text(); real = (B / "e4_c_latency/real_worker.py").read_text()
    assert "prepared=latency.prepare('smoke')" in worker and "source_and_weights(plan)" in latency and "sha(p)!=item['expected_lfs_sha256']" in real
    proof = {"status": "passed", "independent_review": True, "runtime_plan_sha256": sha(D / "runtime_plan001.json"), "scope": "GPU full DEV only", "reviewer_sha256": sha(__file__),
             "preflight_sha256": sha(out / "worker_preflight.json"), "preflight_terminal_sha256": sha(D / "monitor_preflight001/terminal.json"), "verification_plan_sha256": sha(D / "verification_plan001.json"),
             "preflight_started_sha256": sha(D / "monitor_preflight001/started.json"), "preflight_launch_sha256": sha(out / "launch.json"), "preflight_exit_sha256": sha(out / "exit.json"), "container_id": cid,
             "GT_opened": False, "target_predictions_opened": False, "errors": [], "limitations": ["preflight did not load model or run per-image processor", "four weight hashes checked inside preflight but per-shard post-run digest not separately archived", "final GPU gate and per-image processor checks run again at execution", "random adapter includes discarded author greedy; no speed claim"]}
    with OUT.open("x") as f: json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
