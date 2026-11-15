"""Answer-free selectors and the existing stock Qwen3/DeepStack generation path.

Only ``teacher_scores`` and ``gold_nll`` accept answers, for offline fit/val
label construction. ``prepare``, ``baseline_keep`` and ``generate`` do not.
The prompt is the repository's concise DocVQA VQA prompt for every arm.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import time

import torch

from .artifacts import sha256

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
PROMPT_SUFFIX = "\nPlease answer concisely with short words or phrases when possible."
# Also support CPU test collection from the repository root. Heavy model
# imports remain lazy and no GPU/model is created here.
if str(QWEN) not in sys.path:
    sys.path.insert(0, str(QWEN))


def source_paths():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh"))
    paths += [QWEN / name for name in (
        "model/native_qwen3.py", "model/e0_selectors.py", "model/pruner.py",
        "scripts/discovery/common.py", "scripts/discovery/instrumented.py",
        "scripts/multistream_oracle/oracle.py", "scripts/multistream_oracle/gates.py",
        "scripts/multistream_oracle/data.py", "scripts/multistream_oracle/experiment_common.py",
        "VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py",
        "VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py",
        "VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py",
        "VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py",
    )]
    protocol = ROOT / "docs/paired_pruning_experiments.md"
    if protocol.exists():
        paths.append(protocol)
    return sorted({path.resolve() for path in paths})


def load_engine(model_path=None):
    # Must be loaded outside inference_mode: offline input gradients need
    # ordinary frozen decoder parameter tensors.
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Load the teacher outside torch.inference_mode()")
    if model_path:
        os.environ["QWEN3_VL_8B_MODEL_PATH"] = str(model_path)
    os.environ.setdefault("QWEN3_VLM_ATTN_IMPL", "sdpa")
    for path in (QWEN / "scripts/discovery", QWEN):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import common
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(common.load_model(common.BASELINE_MODEL, max_new_tokens=128))
    eng.model.eval().requires_grad_(False)
    if getattr(eng.text.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("This native protocol requires the validated SDPA backend")
    return eng


@contextmanager
def low_resolution(eng, pixels=512):
    from vlmeval.vlm.qwen3_vl import model_fixed_res as fixed
    previous = fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels
    fixed.QWEN3_FIXED_RESOLUTION = pixels
    eng.vlm.min_pixels = eng.vlm.max_pixels = pixels * pixels
    try:
        yield
    finally:
        fixed.QWEN3_FIXED_RESOLUTION, eng.vlm.min_pixels, eng.vlm.max_pixels = previous


def sync(device=None):
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def prepare(eng, page, question, max_question_tokens=256):
    """Fresh processor/vision pass. Never inspect question answers or labels."""
    text = question if isinstance(question, str) else question["question"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Question must be nonempty text")
    if sha256(page["image_path"]) != page["image_sha256"]:
        raise RuntimeError("Frozen image bytes changed")
    prompt = text + PROMPT_SUFFIX
    message = [{"type": "image", "value": page["image_path"]},
               {"type": "text", "value": prompt}]
    prep = eng.prepare(message, "DocVQA_VAL")
    V, DS = eng.encode(prep)
    if V.shape[0] != prep["n_vis"] or len(DS) != 3 or any(d.shape != V.shape for d in DS):
        raise RuntimeError("Unaligned native visual/DeepStack streams")
    tokens = eng.tok(prompt, return_tensors="pt", add_special_tokens=True,
                     truncation=False)["input_ids"]
    if not 0 < tokens.shape[1] <= max_question_tokens:
        raise ValueError(f"Question needs {tokens.shape[1]} tokens; limit {max_question_tokens}; no truncation")
    sequence = eng.model.get_input_embeddings()(tokens.to(V.device))[0].detach()
    return dict(message=message, prep=prep, V=V, DS=DS,
                question_embeddings=sequence.to(torch.bfloat16),
                question_token_ids=tokens[0].cpu(), source_image_sha256=page["image_sha256"])


@torch.no_grad()
def baseline_keep(eng, ctx, name, k=256):
    n = int(ctx["prep"]["n_vis"])
    if name in ("full", "rres512"):
        return torch.arange(n, device=ctx["V"].device)
    if name == "random":
        # Seed depends on the frozen image/query, never on an output directory.
        key = "paired-pruning:"+ctx["source_image_sha256"]+":"+ctx["message"][-1]["value"]
        raw = hashlib.sha256(key.encode()).digest()
        gen = torch.Generator().manual_seed(int.from_bytes(raw[:8], "big") % (2**63 - 1))
        return torch.randperm(n, generator=gen)[:k].sort().values.to(ctx["V"].device)
    if name not in ("b1", "b2"):
        raise ValueError(f"Unknown baseline {name}")
    mean, seq = eng.instruction_embeds(ctx["message"], "DocVQA_VAL")
    selected = eng.select(name, k, dict(ctx, K=k, engine=eng, text_mean=mean, text_seq=seq))
    return selected.to(ctx["V"].device)


@torch.inference_mode()
def generate(eng, ctx, keep=None, max_tokens=128, ignore_eos=False):
    from scripts.multistream_oracle.oracle import checked_keep, tensor_sha256
    if eng.model.training:
        raise ValueError("Native generation requires model.eval()")
    sync(ctx["V"].device)
    started = time.perf_counter()
    prep, V, DS = ctx["prep"], ctx["V"], ctx["DS"]
    keep = checked_keep(keep, int(prep["n_vis"]), V.device)
    state = eng.prefill(prep, V, DS, keep, deepstack=True)
    ids, text = eng.decode(state, max_tokens, ignore_eos=ignore_eos)
    sync(ctx["V"].device)
    ready = time.perf_counter()
    # Stop timing before diagnostics perform extra RoPE, hashing, GPU->CPU
    # copies and equality checks. This is also the endpoint used by fresh
    # end-to-end TTFT / fixed-16-token timing in evaluate.py.
    sequence = eng.keep_sequence(prep, keep)
    full_positions = eng.positions_full(prep)[0]
    if not torch.equal(state.keep_idx, keep) or not torch.equal(state.keep_seq, sequence):
        raise AssertionError("Native retained sequence differs")
    if not torch.equal(state.positions_prefill, full_positions[:, :, sequence]):
        raise AssertionError("Native 3D positions changed")
    if state.ds_lengths != [keep.numel()]*3:
        raise AssertionError("Native DeepStack stream lengths changed")
    if state.cache.get_seq_length() != state.lk+len(state.decode_positions):
        raise AssertionError("Native generation cache bookkeeping failed")
    if state.decode_positions and state.decode_positions[0] != int(full_positions.max())+1:
        raise AssertionError("Native decode position drift")
    result = dict(text=text, token_ids=ids, truncated=len(ids)>=max_tokens, deepstack=True,
        actual_visual_count=int(keep.numel()), native_state_n_vis=state.n_vis,
        ds_lengths=state.ds_lengths, keep_idx=keep.cpu().tolist(),
        positions_sha256=tensor_sha256(state.positions_prefill),
        prefill_length=state.lk, cache_length=int(state.cache.get_seq_length()),
        decode_positions=state.decode_positions, gradients_enabled=False,
        generation_ready_perf_counter=ready, cached_generation_ms=(ready-started)*1000)
    result["audit"] = dict(native=True, ds_streams=3, positions_preserved=True,
                           physical_visual_count=int(result["actual_visual_count"]),
                           answer_in_selection=False, fresh_kv=True)
    # Ordinary JSON-serializable audit/prediction records, including the
    # actual retained indices, are suitable for durable per-question storage.
    result["post_generation_audit_ms"] = (time.perf_counter()-ready)*1000
    return result


def teacher_scores(eng, ctx, answer, k=256, max_answer_tokens=128):
    from scripts.multistream_oracle.oracle import score_full_streams
    return score_full_streams(eng, ctx["prep"], ctx["V"], ctx["DS"], answer,
                              K=k, checkpointing=False, expected_visual_tokens=1024,
                              max_answer_tokens=max_answer_tokens)


@torch.no_grad()
def gold_nll(eng, ctx, answer, keep=None, max_answer_tokens=128):
    from scripts.multistream_oracle.oracle import answer_token_ids, build_gold_batch, forward_gold
    targets = answer_token_ids(eng.tok, answer, ctx["V"].device,
                               max_answer_tokens=max_answer_tokens)
    batch = build_gold_batch(eng, ctx["prep"], ctx["V"], ctx["DS"], targets, keep)
    _, loss = forward_gold(eng, batch)
    value = float(loss)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite offline answer loss")
    return value


def anls(text, answers):
    """Match VLMEvalKit vqa_eval.anls_compute + hit_calculate exactly.

    The numerator normalizes whitespace/case; the denominator uses original
    string lengths, as in this repository's official scorer. Keep equality at
    0.5. Every human answer alias participates.
    """
    import editdistance
    if not answers:
        raise ValueError("Empty original answer aliases")
    prediction = str(text)
    normalized = " ".join(prediction.strip().lower().split())
    best = 0.0
    for answer in answers:
        answer = str(answer)
        target = " ".join(answer.strip().lower().split())
        length = max(len(answer.upper()), len(prediction.upper()))
        distance = 0.0 if length == 0 else editdistance.eval(target, normalized) / length
        similarity = 1.0 - distance
        best = max(best, similarity if similarity >= 0.5 else 0.0)
    return best
