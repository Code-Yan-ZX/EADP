"""Shared native-model plumbing for the predeclared calibration pilot."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
for directory in (HERE.parent / "discovery", HERE.parent / "binding_oracle", QWEN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common
import torch
from runtime import encode_vz
from binding_selectors import select_binding
from model.native_qwen3 import NativeEngine
from effective_mass import cluster_weights

OUT = QWEN / "outputs" / "attention_calibration"
SEED = 20261109
K = 256
MAX_TOKENS = 32


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def message(item):
    return [dict(type="image", value=item["image_path"]),
            dict(type="text", value=item["question"] + "\nAnswer with the value only.")]


@contextmanager
def resolution(eng, pixels):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
    old = (fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels)
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels * pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = old


@torch.inference_mode()
def prepare_sample(eng, item):
    if sha256(item["image_path"]) != item["image_sha256"]:
        raise ValueError("source image changed after data freeze")
    msg = message(item)
    prep = eng.prepare(msg, "DocVQA_VAL")
    V, DS, scores, keys = encode_vz(eng, prep)
    if prep["n_vis"] != 1024 or V.shape[0] != 1024 or len(DS) != 3:
        raise ValueError("expected 1024 input visual tokens and three DS streams")
    vz = select_binding(V, DS, scores, keys, K, None, None, "baseline_fixed", SEED)
    count, effective = cluster_weights(keys, vz)
    text_mean, text_seq = eng.instruction_embeds(msg, "DocVQA_VAL")
    ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
               text_mean=text_mean, text_seq=text_seq, seed=SEED)
    b2 = eng.select("b2", K, ctx)
    if vz[0].numel() != K or b2.numel() != K or not torch.equal(b2, b2.sort().values):
        raise AssertionError("selector did not produce ordered K=256")
    if int(count.sum()) != 1024:
        raise AssertionError("VisionZip cluster population is not exactly 1024")
    return dict(message=msg, prep=prep, V=V, DS=DS, scores=scores, keys=keys,
                vz=vz, count=count, effective=effective, b2=b2)


@torch.inference_mode()
def predict_native(eng, prep, V, DS, keep=None, V_sel=None, DS_sel=None):
    state = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel)
    ids, text = eng.decode(state, MAX_TOKENS)
    if state.n_vis != (prep["n_vis"] if keep is None else K):
        raise AssertionError("wrong retained visual-token count")
    if state.ds_lengths != [state.n_vis] * 3:
        raise AssertionError("DeepStack length mismatch")
    if state.decode_positions and state.decode_positions[0] != state.prefill_max_pos + 1:
        raise AssertionError("decode position drift")
    return dict(text=text, token_ids=ids, truncated=len(ids) >= MAX_TOKENS,
                eos=len(ids) < MAX_TOKENS,
                n_vis=state.n_vis, ds_lengths=state.ds_lengths,
                prefill_seq_length=state.lk, decode_positions=state.decode_positions)
