"""E0 M2 baseline smoke (milestone M2): every ported arm passes the N4
invariants on 12 samples, then a 30-question generation smoke records
truncation rate and obvious anomalies (prereg M2 milestone)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import e0_gates as G  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")

ARMS = [("b2", 256), ("b1", 256), ("divprune", 256), ("cdpruner", 256),
        ("hiprune", 256), ("visionzip", 256), ("fastv", 256), ("pdrop", 256),
        ("sparsevlm", 256), ("b2", 64), ("visionzip", 64), ("fastv", 64),
        ("pdrop", 64), ("sparsevlm", 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="comma list like b2:256,fastv:64")
    args = ap.parse_args()

    arms = ARMS
    if args.arms:
        arms = [tuple(a.rsplit(":", 1)) if ":" in a else (a, 256)
                for a in args.arms.split(",")]
        arms = [(n, int(k)) for n, k in arms]

    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    spec = ([("TextVQA_VAL", r) for r in G.dev_rows("TextVQA_VAL", 4)]
            + [("DocVQA_VAL", r) for r in G.dev_rows("DocVQA_VAL", 4)]
            + [("OCRBench", r) for r in G.dev_rows("OCRBench", 4)])
    items = G.build_items(model, spec)

    out = dict(arms={})
    n4 = G.gate_n4(eng, items, arms=arms)
    out["n4"] = {k: dict(n=v["n"], n_bad=v["n_bad"]) for k, v in n4["arms"].items()}

    # 30-question smoke on TextVQA_VAL DEV (first 30 rows)
    smoke_rows = G.dev_rows("TextVQA_VAL", 30)
    smoke_items = G.build_items(model, [("TextVQA_VAL", r) for r in smoke_rows])
    for name, K in arms:
        n_trunc, empty, samples = 0, 0, []
        for it in smoke_items:
            res = G.with_oom_retry(
                lambda it=it, name=name, K=K: eng.generate(
                    it["message"], it["ds"], K=K, selector=name,
                    max_new_tokens=2048),
                tag=f"smoke {name}:{K} {it['idx']}")
            txt = res["text"].strip()
            n_trunc += int(res["meta"]["n_vis_kept"] >= 0 and
                           len(res["gen_ids"]) >= 2048)
            empty += int(len(txt) == 0)
            if len(samples) < 3:
                samples.append(txt[:60])
        out["arms"][f"{name}:{K}"]["smoke"] = dict(
            n=len(smoke_items), truncated=n_trunc, empty=empty,
            truncation_rate=n_trunc / len(smoke_items),
            samples=samples)
        print(f"[smoke] {name}:{K} trunc={n_trunc}/{len(smoke_items)} "
              f"empty={empty}", flush=True)

    with open(os.path.join(OUT_DIR, "e0_smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[saved] e0_smoke.json", flush=True)


if __name__ == "__main__":
    main()
