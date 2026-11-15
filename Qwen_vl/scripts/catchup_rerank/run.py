"""Exploratory, source-document-paired QA evaluation for two-stage routing.

No outcome-based sample filtering. All predictions, including truncated ones,
are persisted independently. Run only after interface gates pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discovery"))
sys.path.insert(0, str(HERE.parent / "cf_mining"))
sys.path.insert(0, str(HERE.parent / "binding_oracle"))
import common
import torch
from cf_mine import correct, atomic_json
from model.native_qwen3 import NativeEngine
from route import prepare_route, prepare_dense_reference, run_route

OUT = Path(common.QWEN_ROOT) / "outputs/catchup_rerank"
ARMS = ("baseline", "dense_full", "dense_2k", "ltc_2k")


@contextmanager
def resolution(eng, pixels):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    previous = fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels*pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = previous


def message(item):
    return [dict(type="image", value=item["image_path"]),
            dict(type="text", value=item["question"] + "\nAnswer with the value only.")]


def predict_native(eng, prep, V, DS, max_tokens):
    st = eng.prefill(prep, V, DS, None)
    ids, text = eng.decode(st, max_tokens)
    return dict(text=text, token_ids=ids, truncated=len(ids) >= max_tokens,
                n_vis=st.n_vis, ds_lengths=st.ds_lengths)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(OUT / "data/manifest.json"))
    p.add_argument("--output", default=str(OUT / "predictions.json"))
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=24)
    args = p.parse_args()
    gate_path = OUT / "gates.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if not gate.get("pass_all"):
        raise RuntimeError("Passing native-routing gates required before evaluation")
    if gate.get("route_sha256") != hashlib.sha256((HERE / "route.py").read_bytes()).hexdigest():
        raise RuntimeError("Routing implementation changed after gates")
    manifest_path = Path(args.manifest)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    items = manifest["items"][:args.limit]
    output = Path(args.output)
    sha = hashlib.sha256(raw).hexdigest()
    freeze_path = OUT / "run_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("Freeze the protocol and model/code hashes before prediction")
    frozen = json.loads(freeze_path.read_text())
    if frozen["manifest_sha256"] != sha:
        raise RuntimeError("Input data differs from frozen protocol")
    for name in ("route.py", "run.py", "gates.py"):
        if frozen["script_sha256"][name] != hashlib.sha256((HERE/name).read_bytes()).hexdigest():
            raise RuntimeError("Experiment implementation changed after freeze: " + name)
    config = dict(manifest_sha256=sha, max_tokens=args.max_tokens, K=256,
                  route_layers=[2, 7], seed=20261108, model=common.BASELINE_MODEL,
                  arms=list(ARMS), sample_count=len(items))
    if output.exists():
        result = json.loads(output.read_text())
        if result["config"] != config:
            raise ValueError("Output config or manifest changed; use a fresh output")
    else:
        result = dict(config=config, records=[], complete=False)
    done = {r["item_id"] for r in result["records"]}
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=args.max_tokens)
    eng = NativeEngine(model)
    start = time.time()
    for i, item in enumerate(items):
        if item["item_id"] in done:
            continue
        actual_sha = hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
        if actual_sha != item["image_sha256"]:
            raise ValueError("Source image changed: " + item["item_id"])
        msg = message(item)
        prep = eng.prepare(msg, "DocVQA_VAL")
        V, DS = eng.encode(prep)
        assert prep["n_vis"] == 1024 and V.shape[0] == 1024
        full = predict_native(eng, prep, V, DS, args.max_tokens)
        with resolution(eng, 512):
            small = eng.prepare(msg, "DocVQA_VAL")
            small_v, small_ds = eng.encode(small)
            assert small["n_vis"] == 256
            resize = predict_native(eng, small, small_v, small_ds, args.max_tokens)
        prepared = prepare_route(eng, prep, V, DS, K=256, route_layers=(2, 7))
        dense = prepare_dense_reference(eng, prep, V, DS, target_layer=7)
        preds = {"full": full, "rres512": resize}
        for variant in ARMS:
            preds[variant] = run_route(eng, prepared, variant=variant,
                                       dense_reference=dense,
                                       max_new_tokens=args.max_tokens)
            preds[variant]["truncated"] = len(preds[variant]["token_ids"]) >= args.max_tokens
        for pred in preds.values():
            pred["correct"] = any(correct(pred["text"], a) for a in item["answers"])
        row = dict(item_id=item["item_id"], parent_id=item["parent_id"],
                   question=item["question"], answers=item["answers"],
                   image_sha256=actual_sha, predictions=preds)
        result["records"].append(row)
        atomic_json(output, result)
        print(f"[{i+1}/{len(items)}] {item['item_id']} " +
              " ".join(f"{k}={int(v['correct'])}" for k, v in preds.items()) +
              f" elapsed={time.time()-start:.1f}s", flush=True)
    result["complete"] = len(result["records"]) == len(items)
    result["elapsed_seconds"] = time.time() - start
    atomic_json(output, result)


if __name__ == "__main__":
    main()
