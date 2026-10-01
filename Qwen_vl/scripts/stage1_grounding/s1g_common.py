"""Stage-1 grounding-guidance discovery — shared plumbing.

Reuses the FROZEN anchor-merge-pilot panels (outputs/anchor_merge_pilot/
manifest.json: DEV 100/ds, CONFIRM 200/ds incl. the OCRBench top-up) so
sample identity and the BASE arm are byte-compatible with the round-2
results.  Selection is ONLINE per arm (Stage-1 changes the keep set), so
frozen banks are used only as a correctness reference, never as input.

Fixed for the whole round (Stage-2 freeze, user directive):
  facility-location selector (official b1), K=256, identity gather
  (no Anchor Completion), training-free, token budget unchanged.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  (LMUData / HF_HOME / offline env defaults)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "stage1_grounding")
ACC_DIR = os.path.join(OUT_DIR, "acc")
DIAG_DIR = os.path.join(OUT_DIR, "diag")
AMP_OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "anchor_merge_pilot")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ACC_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
K = 256

# frozen arm registry (user-approved round 1; docs in the report)
S1_ARMS: dict[str, dict] = {
    "A_G1": dict(mode="only", stat="g1"),
    "A_G2": dict(mode="only", stat="g2"),
    "A_G3": dict(mode="only", stat="g3"),
    "B_G1": dict(mode="cal", stat="g1"),
    "B_G2": dict(mode="cal", stat="g2"),
    "B_G3": dict(mode="cal", stat="g3"),
    "C_G1": dict(mode="cal", stat="peak"),
}
# calibration grid for the softmax strength (frozen; small-DEV pick ONCE)
LAM_GRID = [0.25, 0.5, 1.0]

FROZEN_LAM: float | None = None   # set via ensure_selectors()


def arm_selector_name(arm: str) -> str:
    return f"s1_{arm}"


def ensure_selectors(lam: float) -> None:
    """Register the round's selector variants into model.e0_selectors.
    Idempotent per (lam); all arms of a run MUST share one lam."""
    global FROZEN_LAM
    from model import e0_selectors
    from model.eadp_stage1 import make_selector
    for arm, cfg in S1_ARMS.items():
        name = arm_selector_name(arm)
        e0_selectors.register(
            name, make_selector(cfg["mode"], cfg["stat"], lam))
    FROZEN_LAM = lam


def load_manifest() -> dict:
    import json
    p = os.path.join(AMP_OUT_DIR, "manifest.json")
    with open(p) as f:
        return json.load(f)


def shard_path(split: str, arm: str, ds: str, tag: str = "") -> str:
    d = os.path.join(ACC_DIR, split, arm + (f"_{tag}" if tag else ""), f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path):
    import json
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path, shard):
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def degeneracy(text: str) -> dict:
    t = text.strip()
    toks = t.split()
    rep = 0
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        rep = max((grams.count(g) for g in set(grams)), default=0)
    return dict(empty=len(t) == 0, n_chars=len(t), repeat4=rep)


# ---------------------------------------------------------------------------
# one-sample runner: ONLINE selection via engine.generate (records
# selector_ms natively), unlike amp run_one which consumed a frozen bank.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_one(eng, msg, ds, arm: str, max_new_tokens: int = 2048,
            timings: dict | None = None, want_diag: bool = False):
    from model import eadp_stage1
    timings = timings if timings is not None else {}
    eadp_stage1.LAST_DIAG = None
    out = eng.generate(msg, ds, K=K, selector=arm_selector_name(arm),
                       max_new_tokens=max_new_tokens, timings=timings)
    diag = None
    if want_diag:
        diag = dict(arm=arm, ds=ds,
                    last=eadp_stage1.LAST_DIAG)  # last image of the sample
    return out, diag


def instruction_tokens(eng, msg, ds) -> list[str]:
    """Token strings of the instruction sequence, same tokenisation as
    NativeEngine.instruction_embeds (add_special_tokens=True)."""
    content = eng.vlm._prepare_content(msg, dataset=ds)
    parts = [c["text"] for c in content if c.get("type") == "text"]
    instr = " ".join(parts).strip() if parts else "Describe this image."
    enc = eng.tok(instr, return_tensors="pt", add_special_tokens=True,
                  truncation=True, max_length=eng.tok.model_max_length)
    return [eng.tok.decode([int(i)]) for i in enc.input_ids[0]]
