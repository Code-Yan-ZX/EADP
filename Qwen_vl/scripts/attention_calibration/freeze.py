"""Write-once pre-generation freeze of protocol, hashes and rendered inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import torch
import transformers

from experiment_common import (HERE, ROOT, OUT, MAX_TOKENS, NativeEngine,
                               common, message, save_json, sha256)
from calibration_run import ARMS, contrasts, source_paths


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="continue to QA in the same model process")
    parser.add_argument("--benchmark", action="store_true", help="also run timing after QA")
    args = parser.parse_args()
    if args.benchmark and not args.run:
        raise ValueError("--benchmark requires --run")
    output = OUT / "run_manifest.json"
    if output.exists() or (OUT / "predictions.json").exists():
        raise RuntimeError("Refusing to change a freeze or existing predictions")
    manifest_path = OUT / "data/manifest.json"
    dev_path = OUT / "data/development_manifest.json"
    gates_path = OUT / "gates.json"
    gate = json.loads(gates_path.read_text())
    if not gate.get("pass_all") or gate.get("calibration_sha256") != sha256(HERE / "calibration.py"):
        raise RuntimeError("Current native GPU gates must pass before freeze")
    if not gate.get("source_sha256"):
        raise RuntimeError("gate lacks source hashes")
    for path, expected in gate["source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError("gated source changed: " + path)
    items = json.loads(manifest_path.read_text())["items"]
    if len(items) != 32:
        raise ValueError("expected 32 frozen primary questions")
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS)
    eng = NativeEngine(model)
    if eng.text.config._attn_implementation != "sdpa":
        raise RuntimeError("unexpected attention backend")
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    revision = (snapshot_root / "refs/main").read_text().strip()
    snapshot = snapshot_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError("cached Qwen model revision is missing")
    rendered = {}
    for index, item in enumerate(items):
        if sha256(item["image_path"]) != item["image_sha256"]:
            raise ValueError("source image differs from data freeze")
        prep = eng.prepare(message(item), "DocVQA_VAL")
        ids = prep["ids"][0].detach().cpu().tolist()
        if prep["n_vis"] != 1024:
            raise ValueError("unexpected visual token count in preflight")
        rendered[item["item_id"]] = dict(input_ids=ids, n_vis=prep["n_vis"],
            rendered_text=eng.tok.decode(ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False),
            image_sha256=item["image_sha256"])
        print(f"prompt freeze {index+1}/{len(items)}", flush=True)
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], check=True, text=True,
                              capture_output=True).stdout.strip()
    weight_files = sorted(p.name for p in snapshot.glob("*.safetensors"))
    checksums = {name: sha256(snapshot / name) for name in
                 ("config.json", "model.safetensors.index.json", "preprocessor_config.json")
                 if (snapshot / name).exists()}
    payload = dict(schema_version=1, manifest_sha256=sha256(manifest_path),
        development_manifest_sha256=sha256(dev_path), gates_sha256=sha256(gates_path),
        source_sha256={str(p): sha256(p) for p in source_paths()},
        model=dict(name=common.BASELINE_MODEL, revision=revision, snapshot=str(snapshot),
                   metadata_sha256=checksums, weight_files=weight_files,
                   dtype=str(next(eng.model.parameters()).dtype)),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, cuda_runtime=torch.version.cuda,
            device=hardware, attention_backend=eng.text.config._attn_implementation,
            rotary_inv_freq=eng.text.rotary_emb.inv_freq.detach().cpu().tolist(),
            rotary_sections=list(eng.text.rotary_emb.mrope_section),
            rotary_attention_scaling=float(eng.text.rotary_emb.attention_scaling)),
        generation=dict(max_new_tokens=MAX_TOKENS, do_sample=False, eos_ids=sorted(eng.eos_ids),
                        prompt_suffix="\nAnswer with the value only."),
        planned_arms=list(ARMS), contrasts=contrasts(), rendered_inputs=rendered,
        pre_generation=True, predictions_observed=False)
    save_json(output, payload)
    print(f"wrote {output} sha256={sha256(output)}", flush=True)
    if args.run:
        from calibration_run import main as run_main
        run_main(eng)
    if args.benchmark:
        from benchmark import main as benchmark_main
        benchmark_main(eng)


if __name__ == "__main__":
    main()
