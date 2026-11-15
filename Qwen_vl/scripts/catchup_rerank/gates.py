"""GPU integration gates on one primary development document, not the pilot."""
from __future__ import annotations
import json
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discovery"))
sys.path.insert(0, str(HERE.parent / "cf_mining"))
import common
import torch
from cf_mine import atomic_json
from model.native_qwen3 import NativeEngine
from route import identity_gate, no_reentry_gate, prepare_route, prepare_dense_reference, run_route

OUT = Path(common.QWEN_ROOT) / "outputs/catchup_rerank"


@torch.inference_mode()
def main():
    manifest = json.loads((Path(common.QWEN_ROOT) / "outputs/local_reward_diagnostic/data/manifest.json").read_text())
    item = next(x for x in manifest["items"] if x["role"] == "primary")
    msg = [dict(type="image", value=item["image_path"]),
           dict(type="text", value=item["qa_fields"][0]["question"] + "\nAnswer with the value only.")]
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=8)
    eng = NativeEngine(model)
    prep = eng.prepare(msg, "DocVQA_VAL")
    V, DS = eng.encode(prep)
    ident = identity_gate(eng, prep, V, DS, max_new_tokens=8)
    print("identity", ident, flush=True)
    # The separate native single-prune harness rebuilds a bool causal mask
    # from populated early-layer caches; the SDPA integration slices it to
    # the actual compact key length.  It is exactly lower triangular but
    # selects a numerically different GPU kernel.  In this unpadded prefill,
    # None has identical causal semantics and matches stock's SDPA kernel.
    # Restore the imported helper immediately; no panel execution is patched.
    import model.native_qwen3 as native_module
    original_mask = native_module.create_causal_mask
    try:
        native_module.create_causal_mask = lambda *a, **kw: None
        nr = no_reentry_gate(eng, prep, V, DS, K=256, route_layers=(2, 7), max_new_tokens=4)
    finally:
        native_module.create_causal_mask = original_mask
    nr["native_single_prune_mask_override"] = "None, equivalent causal SDPA for unpadded prefill; prior unmodified comparison retained separately"
    print("no_reentry", nr, flush=True)
    prepared = prepare_route(eng, prep, V, DS, K=256, route_layers=(2, 7))
    dense = prepare_dense_reference(eng, prep, V, DS, target_layer=7)
    outputs = {}
    for arm in ("baseline", "dense_full", "dense_2k", "ltc_2k", "baseline"):
        out = run_route(eng, prepared, variant=arm, dense_reference=dense, max_new_tokens=4)
        if arm == "baseline" and arm in outputs:
            repeat_ok = out["token_ids"] == outputs[arm]["token_ids"]
        else:
            outputs[arm] = out
        print("arm", arm, out, flush=True)
    n, length, k = prep["n_vis"], int(prep["ids"].shape[1]), 256
    expected_cache = [length if i < 2 else length-n+k for i in range(len(eng.text.layers))]
    baseline = outputs["baseline"]["diag"]
    structural = {}
    for arm, out in outputs.items():
        d = out["diag"]
        count = len(d["decode_positions"])
        for name, size in (("first_keep", k), ("second_keep", k), ("baseline_second_keep", k), ("shortlist", 2*k)):
            values = d[name]
            assert values == sorted(set(values)) and len(values) == size
            assert all(0 <= x < n for x in values)
        assert d["first_keep"] == baseline["first_keep"]
        assert d["baseline_second_keep"] == baseline["baseline_second_keep"]
        assert d["common_anchors"] == baseline["common_anchors"]
        assert d["common_anchors"] == sorted(set(d["first_keep"]) & set(d["baseline_second_keep"]))
        if arm in ("dense_2k", "ltc_2k"):
            assert set(d["second_keep"]).issubset(d["shortlist"])
        assert d["prefill_cache_lengths"] == expected_cache
        assert d["final_cache_lengths"] == [x+count for x in expected_cache]
        assert d["decode_positions"] == list(range(d["decode_start"], d["decode_start"]+count))
        assert d["decode_start"] == int(eng.positions_full(prep)[0].max()) + 1
        assert d["ds_injection_lengths"] == [n, n, k]
        assert d["prepared_cache_lengths_before"] == d["prepared_cache_lengths_after"]
        structural[arm] = True
    summary = dict(pass_all=bool(ident["pass_all"] and nr["pass_all"] and repeat_ok and all(structural.values())),
                   route_sha256=hashlib.sha256((HERE / "route.py").read_bytes()).hexdigest(),
                   development_item=item["page_id"], identity=ident, no_reentry=nr,
                   repeat_baseline_tokens=repeat_ok, structural_checks=structural, variants=outputs)
    atomic_json(OUT / "gates.json", summary)
    if not summary["pass_all"]:
        raise RuntimeError("GPU gate failed")


if __name__ == "__main__":
    main()
