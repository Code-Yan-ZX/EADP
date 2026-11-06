"""Freeze and supervise the unchanged public Qwen launcher, without result reuse."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

from container_runtime import AUTHOR, ROOT, OUT, command

RUN = OUT / "public_fa2_full"
PLAN = OUT / "public_fa2_full_plan.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    assert not RUN.exists(), "Never overwrite an existing full run"
    probe = json.loads((OUT / "source_probe_plan.json").read_text())
    assert probe["recipe"]["visual_token_num"] == 256
    sources = dict(probe["source_sha256"])
    for p in (AUTHOR / "scripts/run_eadp.sh", AUTHOR / "scripts/env.sh", AUTHOR / "VLMEvalKit/run.py",
              OUT / "container_runtime.py", OUT / "source_probe.py", Path(__file__).resolve()):
        sources[str(p)] = sha(p)
    env = {
        "MODEL_FAMILY": "qwen3-8b", "TOKENS": "256", "EADP_ALPHA": "0.5", "EADP_BETA": "2.0",
        "DATASETS": "DocVQA_VAL AI2D_TEST", "GPU_IDS": "0", "OUTPUT_ROOT": str(RUN / "outputs"),
    }
    code = "import subprocess; raise SystemExit(subprocess.call(" + repr(["/bin/bash", str(AUTHOR / "scripts/run_eadp.sh")]) + "))"
    cmd = command(["-c", code], gpu=True, name="eadp-qwen-author-fa2-full-20261028", extra_env=env)
    reports = [OUT / name for name in ("source_probe_plan.json", "qwen_weight_identity.json",
        "qwen_processor_tokenizer_identity.json", "dataset_cache_audit.json", "flash_attention_gpu_kernel_probe.json")]
    plan = {
        "prepared_utc": now(), "purpose": "Full public-source recipe reproduction, not hyperparameter selection",
        "source_sha256": sources, "environment": env, "command": cmd, "datasets": probe["datasets"],
        "frozen_evidence": {str(p): sha(p) for p in reports},
        "author_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AUTHOR, text=True).strip(),
        "model_name": probe["model_name"], "recipe": probe["recipe"],
        "launch_gates": ["four predeclared author-wrapper questions complete with FA2 and unchanged source",
            "actual smoke container/host worker has exited", "slot A explicitly retained for root",
            "no more than one other GPU model; loading counts as a model", "no prior output directory"],
        "validation": ["Both complete5349/3088 ordered ID/question/answer sets", "No failed/empty predictions",
            "Fresh author score files and independent recalculation", "Source/evidence hashes unchanged after run"],
        "score_boundary": "Public run.py defaults are retained. AI2D may attempt unavailable API matching before its documented exact-matching fallback. No host API credentials or network are provided. Preserve any scoring failure and recover CPU scoring explicitly; never regenerate or replace predictions silently.",
        "limits": ["Author environment core versions and FA2 are used; full transitive package equality is not claimed.",
            "No source/math/position/DeepStack repair. No old prediction or selection-bank reuse.",
            "Concurrent timings are not exclusive efficiency measurements.",
            "Public recipe reproduction does not establish an unpublished paper argv."],
    }
    save(PLAN, plan)
    print(json.dumps({"prepared": True, "GPU_loaded": False, "command": cmd}, indent=2))


def run(authorized):
    assert authorized, "Root must allocate slot A before --gpu-authorized"
    plan = json.loads(PLAN.read_text())
    for field in ("source_sha256", "frozen_evidence"):
        for path, expected in plan[field].items():
            assert sha(path) == expected, path
    smoke = json.loads((OUT / "author_fa2_smoke/state.json").read_text())
    assert smoke["status"] == "complete" and smoke["n"] == 4 and smoke["source_unchanged"]
    smoke_answers = OUT / "author_fa2_smoke/answers.jsonl"
    assert sha(smoke_answers) == smoke["answers_sha256"]
    assert len(smoke_answers.read_text().splitlines()) == 4
    smoke_identity = json.loads((OUT / "author_fa2_smoke/process_identity.json").read_text())
    old_stat = Path("/proc") / str(smoke_identity["host_pid"]) / "stat"
    assert not old_stat.exists() or old_stat.read_text().split()[21] != str(smoke_identity["host_start_ticks"])
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True)
    assert len([r for r in apps.splitlines() if r.strip()]) < 2, apps
    RUN.mkdir(exist_ok=False)
    state = {"status": "starting", "started_utc": now(), "controller_pid": os.getpid(),
             "plan_sha256": sha(PLAN), "GPU_processes_before_launch": apps, "allocation": "root slot A"}
    save(RUN / "state.json", state)
    with (RUN / "launcher.log").open("w") as log:
        worker = subprocess.Popen(plan["command"], stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", docker_client_pid=worker.pid)
        save(RUN / "state.json", state)
        identity_written = False
        while worker.poll() is None:
            if not identity_written:
                inspect = subprocess.run(["docker", "inspect", "eadp-qwen-author-fa2-full-20261028"], text=True, capture_output=True)
                if inspect.returncode == 0:
                    item = json.loads(inspect.stdout)[0]
                    pid = item["State"]["Pid"]
                    stat = Path("/proc") / str(pid) / "stat"
                    if pid and stat.exists():
                        save(RUN / "process_identity.json", {"container_id": item["Id"], "image": item["Image"],
                            "container_state": item["State"], "host_init_pid": pid, "host_start_ticks": stat.read_text().split()[21],
                            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "observed_utc": now()})
                        identity_written = True
            time.sleep(2)
    state.update(status="generation_launcher_exited_validation_pending", launcher_exit_code=worker.returncode,
                 launcher_finished_utc=now(), source_unchanged=True)
    for path, expected in plan["source_sha256"].items():
        assert sha(path) == expected, path
    save(RUN / "state.json", state)
    print(json.dumps(state, indent=2))
    raise SystemExit(worker.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.gpu_authorized)
