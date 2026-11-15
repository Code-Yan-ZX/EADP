"""Shared native model and data helpers for the four-stream teacher pilot."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
for path in (HERE.parent / "discovery", QWEN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common
import torch
from model.native_qwen3 import NativeEngine

OUT = QWEN / "outputs/multistream_oracle"
K = 256
MAX_TOKENS = 32
SEED = 20261109


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def convert(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(item) for item in value]
    return value


def message(item):
    return [dict(type="image", value=item["image_path"]),
            dict(type="text", value=item["question"] + "\nAnswer with the value only.")]


@contextmanager
def resolution(eng, pixels):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    previous = fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels * pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = previous


def random_keep(item_id, n=1024, k=K):
    # Fully specified, platform-independent derivation; no outcome is read.
    raw = hashlib.sha256(f"{SEED}:{item_id}".encode()).digest()
    seed = int.from_bytes(raw[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return seed, torch.randperm(n, generator=generator)[:k].sort().values


@torch.inference_mode()
def prepare_sample(eng, item):
    if sha256(item["image_path"]) != item["image_sha256"]:
        raise ValueError("source image changed after data freeze")
    msg = message(item)
    prep = eng.prepare(msg, "DocVQA_VAL")
    V, DS = eng.encode(prep)
    if prep["n_vis"] != 1024 or V.shape[0] != 1024 or len(DS) != 3:
        raise AssertionError("expected native 1024 visual tokens and three DS streams")
    text_mean, text_seq = eng.instruction_embeds(msg, "DocVQA_VAL")
    ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
               text_mean=text_mean, text_seq=text_seq, seed=SEED)
    b2 = eng.select("b2", K, ctx)
    if b2.numel() != K or not torch.equal(b2, b2.sort().values):
        raise AssertionError("B2 selection is not ordered K256")
    seed, random = random_keep(item["item_id"])
    return dict(message=msg, prep=prep, V=V, DS=DS, b2=b2, random=random,
                random_seed=seed)
