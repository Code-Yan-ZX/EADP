"""Synchronized single-sample latency diagnostic; model initialization excluded."""
from __future__ import annotations
import json
from contextlib import contextmanager
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discovery"))
sys.path.insert(0, str(HERE.parent / "cf_mining"))
import common
import torch
from cf_mine import atomic_json
from model.native_qwen3 import NativeEngine
from route import prepare_route, run_route

OUT = Path(common.QWEN_ROOT) / "outputs/catchup_rerank"


@contextmanager
def resolution(eng, pixels):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    old = fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels*pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = old


def predict_native(eng, prep, V, DS, cap):
    state = eng.prefill(prep, V, DS, None)
    ids, text = eng.decode(state, cap)
    return dict(token_ids=ids, text=text)


@torch.inference_mode()
def main():
    gates = json.loads((OUT / "gates.json").read_text())
    if not gates.get("pass_all"):
        raise RuntimeError("Native gates failed")
    source = json.loads((Path(common.QWEN_ROOT) / "outputs/local_reward_diagnostic/data/manifest.json").read_text())
    item = next(x for x in source["items"] if x["role"] == "primary")
    msg = [dict(type="image", value=item["image_path"]),
           dict(type="text", value=item["qa_fields"][0]["question"] + "\nAnswer with the value only.")]
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=8)
    eng = NativeEngine(model)

    def run(arm, cap):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if arm == "rres512":
            with resolution(eng, 512):
                prep = eng.prepare(msg, "DocVQA_VAL")
                V, DS = eng.encode(prep)
                out = predict_native(eng, prep, V, DS, cap)
        else:
            prep = eng.prepare(msg, "DocVQA_VAL")
            V, DS = eng.encode(prep)
            shared = prepare_route(eng, prep, V, DS, K=256, route_layers=(2, 7))
            out = run_route(eng, shared, variant=arm, dense_reference=None, max_new_tokens=cap)
        torch.cuda.synchronize()
        return dict(arm=arm, max_new_tokens=cap, generated_tokens=len(out["token_ids"]),
                    total_ms=(time.perf_counter()-t0)*1000, timings=out.get("timings", {}))

    for a in ("baseline", "ltc_2k", "rres512"):
        run(a, 1)
    measurements = []
    for rep in range(3):
        for arm in (("baseline", "ltc_2k", "rres512") if rep % 2 == 0 else ("rres512", "ltc_2k", "baseline")):
            row = run(arm, 1)
            row["repeat"] = rep
            measurements.append(row)
            print(json.dumps(row), flush=True)
    atomic_json(OUT / "latency.json", dict(
        scope="single development image, batch 1, synchronized wall clock, model load excluded; includes preprocessing, vision, both route stages and first token; dense-reference offline passes excluded",
        source_item=item["page_id"], warmups_per_arm=1, rows=measurements,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
