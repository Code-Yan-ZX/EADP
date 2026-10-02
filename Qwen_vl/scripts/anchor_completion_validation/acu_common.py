"""Anchor Completion Validation — shared plumbing.

Supplemental protocol: docs/anchor_completion_validation_protocol.md (frozen
2026-10-02, commit 7638b54).  Main judgment / stats / prereg fallback live in
docs/anchor_merge_full_eval_prereg.md; merge math, deterministic reduction,
bank freezing and scoring rules are inherited verbatim from the
anchor_merge_pilot / stage1_cross_stream_pilot rounds.

This round adds THREE operator ablations on the SAME frozen bank (same
keep/gid) as BASE/MAIN025, restricted to the main feature stream:

  BASE        identity gather (selector b1 = official facility, K=256)
  MAIN025     uniform group mean, lam=0.25, main only      (formal method)
  MAIN100     uniform group mean, lam=1.00, main only      (amplitude ablation)
  MAIN_SIM025 softmax(cos/0.1) weights incl anchor logit 1/0.1,
              then lam=0.25 update, main only              (weight-form ablation)
  BOTH025     uniform lam=0.25, main+DS (ONLY if the prereg §4 mechanical
              trigger fires; conditional secondary analysis)

PruMerge / Libra full-method controls: NOT implementable faithfully on this
path (protocol §4.3) — recorded as not-completed, never faked.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

import torch

# -- path bootstrap (same convention as amp_common) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs",
                       "anchor_completion_validation")
os.makedirs(OUT_DIR, exist_ok=True)

DS_MAIN = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
DS_NONREG = ["ChartQA_TEST", "MMBench_DEV_EN_V11", "MMStar", "RealWorldQA",
             "POPE"]
DS_ALL = DS_MAIN + DS_NONREG
K = 256

# frozen statistics seeds (prereg + protocol §8)
BOOT_SEED = 20261002          # primary 5000-draw stream
N_BOOT_PRIMARY = 5000
N_BOOT_SENS = 20000           # sensitivity = same stream, longer (nested)

FRESH_SEED = 20261002         # protocol §6.3 cluster shuffle
FRESH_MIN_Q = 200

ARMS_FORMAL = ["BASE", "MAIN025"]
ARMS_ABLATION = ["MAIN100", "MAIN_SIM025"]
ARMS_CONDITIONAL = ["BOTH025"]


def arm_cfg(name: str) -> dict:
    if name == "BASE":
        return dict(kind="base", lam=0.0, scope="main", selector="b1", K=256)
    if name == "MAIN025":
        return dict(kind="uniform", lam=0.25, scope="main",
                    selector="b1", K=256)
    if name == "MAIN100":
        return dict(kind="uniform", lam=1.00, scope="main",
                    selector="b1", K=256)
    if name == "MAIN_SIM025":
        return dict(kind="sim", lam=0.25, tau=0.1, scope="main",
                    selector="b1", K=256)
    if name == "BOTH025":     # conditional prereg fallback only
        return dict(kind="uniform", lam=0.25, scope="both",
                    selector="b1", K=256)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# bank (full-split frozen anchors; lean record: keep/gid/gsize/n_vis)
# ---------------------------------------------------------------------------
def bank_path(ds: str) -> str:
    return os.path.join(OUT_DIR, f"bank_full_{ds}.json.gz")


def load_bank(ds: str) -> dict:
    p = bank_path(ds)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with gzip.open(p, "rt") as f:
        return json.load(f)


def save_bank(ds: str, bank: dict) -> str:
    tmp = bank_path(ds) + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, bank_path(ds))
    return bank_sha256(ds)


def bank_sha256(ds: str) -> str:
    with open(bank_path(ds), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# generation shards (per panel/arm/ds), resumable
# ---------------------------------------------------------------------------
def shard_path(panel: str, arm: str, ds: str) -> str:
    d = os.path.join(OUT_DIR, "acc", panel, arm, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path: str, shard: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def shard_meta(panel: str, arm: str, ds: str, base_commit: str,
               max_new_tokens: int) -> dict:
    return dict(panel=panel, arm=arm, ds=ds, K=K, selector="b1",
                bank_sha256=bank_sha256(ds),
                base_commit=base_commit,
                max_new_tokens=max_new_tokens,
                protocol="anchor_completion_validation")


def degeneracy(text: str) -> dict:
    t = text.strip()
    toks = t.split()
    rep = 0
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        rep = max((grams.count(g) for g in set(grams)), default=0)
    return dict(empty=len(t) == 0, n_chars=len(t), repeat4=rep)


def repo_commit() -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=QWEN_ROOT,
        text=True).strip()[:12]


def load_engine(max_new_tokens: int = 2048):
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=max_new_tokens)
    from model.native_qwen3 import NativeEngine
    return NativeEngine(model)


# reuse the pilot's frozen math (uniform/sim merge, assignment, group_sum,
# official facility) — import, never copy-modify
from amp_common import (  # noqa: E402,F401
    compute_assignment,
    group_sum,
    merge_stream,
    official_facility_keep,
    run_one,
)
