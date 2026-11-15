"""Run frozen real-QA pilot; source-documents are never selected by outcome."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from analysis import score_exact
from experiment_common import (HERE, ROOT, OUT, K, MAX_TOKENS, SEED, NativeEngine,
    common, message, predict_native, prepare_sample, resolution, save_json, sha256)
from method import ARMS, predict_calibrated


def tensor_hash(tensor):
    tensor = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def contrasts():
    pairs = []
    def add(a, b):
        if a != b and (a, b) not in pairs:
            pairs.append((a, b))
    for name in ("count", "effective", "bias_const", "restore_prefill", "restore_continuous"):
        add("vz_maskzero", name)
    add("count", "effective")
    add("bias_const", "effective")
    add("count", "restore_prefill")
    add("bias_const", "restore_prefill")
    add("restore_prefill", "restore_continuous")
    for name in ("b2_bias_const", "b2_restore_prefill", "b2_restore_continuous"):
        add("b2_maskzero", name)
    add("b2_restore_prefill", "b2_restore_continuous")
    add("baseline", "vz_maskzero")
    add("b2_baseline", "b2_maskzero")
    for name in ARMS:
        if name not in ("full", "rres512"):
            add("rres512", name)
    return [list(x) for x in pairs]


def source_paths():
    paths = [HERE / x for x in (
        "calibration_run.py", "method.py", "experiment_common.py", "calibration.py", "rope_bias.py",
        "effective_mass.py", "gates.py", "data.py", "analysis.py", "freeze.py", "benchmark.py")]
    paths += [ROOT / "docs/attention_calibration_prereg.md",
              ROOT / "Qwen_vl/model/native_qwen3.py",
              ROOT / "Qwen_vl/model/e0_selectors.py",
              ROOT / "Qwen_vl/model/pruner.py",
              ROOT / "Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py",
              ROOT / "Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py",
              ROOT / "Qwen_vl/scripts/binding_oracle/binding_selectors.py",
              ROOT / "Qwen_vl/scripts/binding_oracle/runtime.py",
              ROOT / "Qwen_vl/scripts/discovery/common.py",
              ROOT / "Qwen_vl/scripts/discovery/instrumented.py",
              ROOT / "Qwen_vl/scripts/cf_mining/cf_mine.py",
              ROOT / "Qwen_vl/scripts/catchup_rerank/analyze.py"]
    return paths


def verify_freeze(manifest_path):
    frozen_path = OUT / "run_manifest.json"
    if not frozen_path.exists():
        raise RuntimeError("write the run freeze and pass GPU gates before primary QA")
    frozen = json.loads(frozen_path.read_text())
    if frozen["manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("data manifest differs from the pre-generation freeze")
    for p in source_paths():
        if frozen["source_sha256"].get(str(p)) != sha256(p):
            raise RuntimeError("frozen source changed: " + str(p))
    gate = json.loads((OUT / "gates.json").read_text())
    if not gate.get("pass_all"):
        raise RuntimeError("the native GPU and CPU gates have not passed")
    if gate.get("calibration_sha256") != sha256(HERE / "calibration.py"):
        raise RuntimeError("gate predates current calibration implementation")
    for path, expected in gate.get("source_sha256", {}).items():
        if sha256(path) != expected:
            raise RuntimeError("gated source changed: " + path)
    if not gate.get("source_sha256"):
        raise RuntimeError("gate lacks source hashes")
    if frozen["gates_sha256"] != sha256(OUT / "gates.json"):
        raise RuntimeError("gate record changed after run freeze")
    return frozen


def input_trace(eng, sample):
    prep, keep, vs, dss, diag = sample["prep"], *sample["vz"]
    full_positions = eng.positions_full(prep)[0]
    vz_seq = eng.keep_sequence(prep, keep)
    b2_seq = eng.keep_sequence(prep, sample["b2"])
    t = diag["tensors"]
    return dict(input_token_ids=prep["ids"][0].detach().cpu().tolist(),
        input_ids_sha256=tensor_hash(prep["ids"]),
        original_V_sha256=tensor_hash(sample["V"]),
        original_DS_sha256=[tensor_hash(d) for d in sample["DS"]],
        vz=dict(keep=keep.cpu().tolist(), dominant=t["dominant_idx"].cpu().tolist(),
            anchors=t["anchor_idx"].cpu().tolist(), cluster_id=t["cluster_id"].cpu().tolist(),
            count=sample["count"].cpu().tolist(), effective=sample["effective"].cpu().tolist(),
            V_sha256=tensor_hash(vs), DS_sha256=[tensor_hash(d) for d in dss],
            position_sha256=tensor_hash(full_positions[:, :, vz_seq])),
        b2=dict(keep=sample["b2"].cpu().tolist(),
            V_sha256=tensor_hash(sample["V"][sample["b2"]]),
            DS_sha256=[tensor_hash(d[sample["b2"]]) for d in sample["DS"]],
            position_sha256=tensor_hash(full_positions[:, :, b2_seq])))


@torch.inference_mode()
def main(eng=None):
    manifest_path = OUT / "data/manifest.json"
    frozen = verify_freeze(manifest_path)
    items = json.loads(manifest_path.read_text())["items"]
    if len(items) != 32 or len({x["parent_id"] for x in items}) != 32:
        raise ValueError("primary manifest must contain exactly 32 source parents")
    output = OUT / "predictions.json"
    config = dict(manifest_sha256=sha256(manifest_path), planned_arms=list(ARMS),
        baseline_arm="baseline", contrasts=contrasts(), seed=SEED,
        input_tokens=1024, K=K, max_tokens=MAX_TOKENS, greedy=True,
        prompt_suffix="\nAnswer with the value only.",
        run_manifest_sha256=sha256(OUT / "run_manifest.json"))
    if output.exists():
        result = json.loads(output.read_text())
        if result["config"] != config:
            raise ValueError("changed frozen config; write to a new output path")
    else:
        result = dict(config=config, records=[], complete=False)
    done = {r["item_id"] for r in result["records"]}
    if done - {x["item_id"] for x in items}:
        raise ValueError("output contains an item outside the frozen manifest")
    if eng is None:
        model = common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS)
        eng = NativeEngine(model)
    if eng.text.config._attn_implementation != "sdpa":
        raise RuntimeError("the frozen attention backend is SDPA")
    start = time.time()
    for i, item in enumerate(items):
        if item["item_id"] in done:
            continue
        sample = prepare_sample(eng, item)
        prep, V, DS = sample["prep"], sample["V"], sample["DS"]
        expected_prompt = frozen["rendered_inputs"][item["item_id"]]
        if prep["ids"][0].detach().cpu().tolist() != expected_prompt["input_ids"]:
            raise RuntimeError("rendered prompt tokens changed after pre-generation freeze")
        predictions = {"full": predict_native(eng, prep, V, DS)}
        with resolution(eng, 512):
            small = eng.prepare(message(item), "DocVQA_VAL")
            vs, ds = eng.encode(small)
            if small["n_vis"] != K:
                raise AssertionError("R-res did not produce 256 visual tokens")
            predictions["rres512"] = predict_native(eng, small, vs, ds)
        for name in ARMS:
            if name not in ("full", "rres512"):
                predictions[name] = predict_calibrated(eng, sample, name)
        if set(predictions) != set(ARMS):
            raise AssertionError("incomplete arm predictions")
        for pred in predictions.values():
            pred["correct"] = score_exact(pred["text"], item["answers"])
        row = dict(item_id=item["item_id"], parent_id=item["parent_id"],
                   image_sha256=item["image_sha256"], question=item["question"],
                   answers=item["answers"], input_trace=input_trace(eng, sample),
                   predictions=predictions)
        result["records"].append(row)
        save_json(output, result)
        print(f"[{i+1}/{len(items)}] {item['item_id']} " +
              " ".join(f"{name}={int(predictions[name]['correct'])}" for name in ARMS) +
              f" elapsed={time.time()-start:.1f}s", flush=True)
    result["complete"] = len(result["records"]) == len(items)
    result["elapsed_seconds_this_process"] = time.time() - start
    save_json(output, result)


if __name__ == "__main__":
    main()
