#!/usr/bin/env python3
"""Prepare CPU-only; launch the frozen full control only with root slot allocation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from patch_only import OUT, BASE, AUTHOR, build_patch

PYTHON = "/home/dell/miniconda3/envs/llava_pruner/bin/python"
PRED = OUT / "v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl"
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare():
    assert not PRED.exists() and not (OUT / "state.json").exists(), "Preserve any started control"
    baseline = json.loads((BASE / "protocol.json").read_text())
    verified = json.loads((BASE / "independent_verification.json").read_text())
    assert verified["success"] and verified["n_rows"] == 4377
    assert baseline["parameters"]["K"] == 128 and baseline["parameters"]["alpha"] == .5 and baseline["parameters"]["beta"] == 1
    assert baseline["dataset"]["n_rows"] == 4377 and baseline["dataset"]["n_circular_groups"] == 1176
    for item in baseline["source_files"]:
        assert sha(item["path"]) == item["sha256"], item["path"]
    _, validation = build_patch()
    plan = copy.deepcopy(baseline)
    command = [PYTHON, str(OUT / "worker.py"), str(OUT / "observed")] + baseline["command"][3:]
    command[command.index("--answers-file") + 1] = str(PRED)
    plan["command"] = command
    plan["runtime_patch"] = validation
    plan["protocol"] = "Full oldEN4377 same-config control: only caller/image/text stream dependencies added in memory"
    plan["prepared_utc"] = datetime.now(timezone.utc).isoformat()
    plan["baseline"] = {"protocol_path": str(BASE / "protocol.json"), "protocol_sha256": sha(BASE / "protocol.json"),
                         "verification_path": str(BASE / "independent_verification.json"),
                         "verification_sha256": sha(BASE / "independent_verification.json"),
                         "prediction_sha256": verified["prediction_sha256"],
                         "circular_accuracy_percent": verified["circular_accuracy_percent"]}
    reference_names = ["image_manifest.json", "comparison_reference.json", "cpu_dependency_input_equivalence.json",
                       "cpu_llm_dependency_equivalence.json", "cpu_inputs_current.json", "cpu_llm_tokens_current.json"]
    for name in reference_names:
        shutil.copyfile(BASE / name, OUT / name)
        assert sha(BASE / name) == sha(OUT / name)
    extra_sources = [OUT / name for name in ("patch_only.py", "worker.py", "run_control.py", "verify_control.py")]
    extra_sources += [BASE / "protocol.json", BASE / "independent_verification.json", BASE / "verify_completed.py",
                      BASE / "vendor_scoring_install_report.json"]
    extra_sources += [OUT / name for name in reference_names]
    plan["source_files"] += [{"path": str(path), "sha256": sha(path)} for path in extra_sources]
    plan["scorer_reuse"] = {"path": str(BASE / "run_author.py"), "sha256": sha(BASE / "run_author.py"),
                            "function": "score", "mathematical_changes": False}
    plan["observer_reuse"] = {"path": str(BASE / "observe_native.py"), "sha256": sha(BASE / "observe_native.py")}
    plan["CPU_XLSX_dependencies"] = str(BASE / "vendor_scoring")
    plan["limits"] = ["Controlled against the complete as-is author recipe; no parameter, data, precision, model or scorer changes.",
                      "Only three AST statements add caller-to-image/text CUDA stream dependencies; source files remain unchanged.",
                      "Concurrent companion workload and low-level scheduling are not replayed identically.",
                      "Local exact-letter circular metric is preserved; external official-server answer matching remains separate.",
                      "A change in score is not by itself proof of stream error; per-row comparisons and repeat consistency are required."]
    save("protocol.json", plan)
    save("READY.json", {"status": "prepared_not_started", "GPU_loaded": False,
                        "coordinator_allocation_required": True, "protocol_sha256": sha(OUT / "protocol.json"),
                        "run_command": ["env", "CUDA_VISIBLE_DEVICES=0", PYTHON, str(OUT / "run_control.py"),
                                        "run", "--allocation", "ROOT_EXPLICIT_SLOT_ALLOCATION"]})
    print(json.dumps({"prepared": True, "GPU_loaded": False, "n_rows": 4377, "n_groups": 1176,
                      "parameters": plan["parameters"], "patch": validation["added_statements"]}, indent=2))


def convert_submission():
    upload = OUT / "answers_upload"
    upload.mkdir(exist_ok=True)
    plan = json.loads((OUT / "protocol.json").read_text())
    command = [PYTHON, str(AUTHOR / "LLaVA/scripts/convert_mmbench_for_submission.py"),
               "--annotation-file", plan["dataset"]["path"], "--result-dir", str(OUT),
               "--upload-dir", str(upload), "--experiment", PRED.stem]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(BASE / "vendor_scoring"))
    with (OUT / "conversion.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    save("conversion.json", {"command": command, "exit_code": result.returncode,
                             "output": str(upload / (PRED.stem + ".xlsx")), "submitted_to_external_server": False,
                             "dependency_isolation": str(BASE / "vendor_scoring")})
    assert result.returncode == 0, "CPU author converter failed; retain native outputs"


def run(allocation):
    assert allocation, "Root must explicitly allocate a slot before invocation"
    plan = json.loads((OUT / "protocol.json").read_text())
    assert not PRED.exists() and not (OUT / "state.json").exists(), "Do not overwrite an existing run"
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    assert len([line for line in apps.splitlines() if line.strip()]) < 2, "Both model slots already occupied"
    save("allocation.json", {"root_slot_allocation": allocation, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                             "GPU_processes_before_launch": apps, "protocol_sha256": sha(OUT / "protocol.json")})
    spec = importlib.util.spec_from_file_location("frozen_as_is_mmb_launcher", BASE / "run_author.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.PRED = PRED
    module.DATA = Path(plan["dataset"]["path"])
    module.command = lambda: plan["command"]
    module.convert_submission = convert_submission
    module.run()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(BASE / "vendor_scoring"))
    subprocess.run([PYTHON, str(OUT / "verify_control.py")], env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--allocation")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.allocation)
