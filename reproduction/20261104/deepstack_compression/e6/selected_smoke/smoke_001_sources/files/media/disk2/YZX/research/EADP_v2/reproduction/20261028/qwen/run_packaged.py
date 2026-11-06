"""Second public-launch attempt: only restore the missing official import."""
from pathlib import Path
import argparse
import json

import run_public as public
from container_runtime import AUTHOR, OUT, command

OVERLAY = OUT / "import_overlay"
public.RUN = OUT / "public_fa2_full_run02"
public.PLAN = OUT / "public_fa2_full_run02_plan.json"


def prepare():
    previous = json.loads((OUT / "public_fa2_full/state.json").read_text())
    assert previous["launcher_exit_code"] == 1
    original_log = OUT / "public_fa2_full/outputs/logs/eadp_qwen3-8b_t256_a0.5_b2.0_DocVQA_VAL.log"
    assert "No module named 'vlmeval.utils.result_transfer'" in original_log.read_text()
    public.prepare()
    plan = json.loads(public.PLAN.read_text())
    provenance = json.loads((OVERLAY / "provenance.json").read_text())
    assert public.sha(OVERLAY / "result_transfer.py") == provenance["sha256"]
    plan["packaging_repair"] = {
        "reason": "Published run.py imports a module absent from the author Git tree; first attempt failed before generation.",
        "original_failure_log": str(original_log), "original_failure_log_sha256": public.sha(original_log),
        "prior_state_sha256": public.sha(OUT / "public_fa2_full/state.json"),
        "provenance": provenance, "import_hook": "Only vlmeval.utils.result_transfer",
        "unused_for_selected_datasets": True, "author_tracked_files_modified": False,
    }
    for p in (Path(__file__).resolve(), OVERLAY / "sitecustomize.py", OVERLAY / "result_transfer.py", OVERLAY / "provenance.json"):
        plan["source_sha256"][str(p)] = public.sha(p)
    cpu_probe = OUT / "public_import_repair_cpu_probe_v2.json"
    assert json.loads(cpu_probe.read_text())["returncode"] == 0
    for p in (cpu_probe, OUT / "actual_environment_packages.json"):
        plan["frozen_evidence"][str(p)] = public.sha(p)
    plan["environment"]["PYTHONPATH"] = str(OVERLAY)
    code = "import subprocess; raise SystemExit(subprocess.call(" + repr(["/bin/bash", str(AUTHOR / "scripts/run_eadp.sh")]) + "))"
    plan["command"] = command(["-c", code], gpu=True, name="eadp-qwen-author-fa2-full-20261028", extra_env=plan["environment"])
    public.save(public.PLAN, plan)
    print(json.dumps({"prepared": True, "GPU_loaded": False, "packaging_repair": plan["packaging_repair"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else public.run(args.gpu_authorized)
