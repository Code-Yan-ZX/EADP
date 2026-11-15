"""Inference-only native Qwen3 plumbing for the binding oracle pilot.

PPE here is an explicit Qwen3 interleaved-mRoPE adaptation of the published
position operator, not a reproduction of the author's SFT checkpoint/results.
No model parameter, decoder attention implementation, or feature stream changes.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discovery"))
import common
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision


@torch.no_grad()
def encode_vz(eng, prep, chunk=256):
    """Stock SDPA vision output, with final-layer received-attention scoring.

Only the score is materialized in query chunks. Softmax is float32 then cast
back to the vision dtype, matching the existing VisionZip port's convention.
The hook never changes a vision forward's input or output.
"""
    captured = {}
    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        cu = kwargs["cu_seqlens"]
        cos, sin = kwargs["position_embeddings"]
        n = hidden.shape[0]
        q, k, _ = module.qkv(hidden).reshape(n, 3, module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        scores = torch.zeros(n, device=hidden.device, dtype=torch.float32)
        offsets = cu.tolist()
        for start, stop in zip(offsets[:-1], offsets[1:]):
            keys = k[:, start:stop]
            acc = torch.zeros(stop-start, device=hidden.device, dtype=torch.float32)
            for at in range(start, stop, chunk):
                weights = torch.matmul(q[:, at:min(at+chunk, stop)], keys.transpose(-2, -1)) * module.scaling
                weights = weights.softmax(-1, dtype=torch.float32).to(hidden.dtype)
                acc += weights.float().mean(0).sum(0)
            scores[start:stop] = acc
        unit = eng.inner.visual.spatial_merge_size ** 2
        captured["score"] = scores.reshape(-1, unit).mean(-1)
        captured["key"] = k.float().mean(0).reshape(-1, unit, k.shape[-1]).mean(1)
    hook = eng.inner.visual.blocks[-1].attn.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        V, DS = eng.encode(prep)
    finally:
        hook.remove()
    return V, DS, captured["score"], captured["key"]


def native_axis_map(rotary):
    """Frequency-axis map of native Qwen3 interleaved mRoPE."""
    f = rotary.inv_freq.numel()
    axes = torch.zeros(f, dtype=torch.long, device=rotary.inv_freq.device)
    for axis in (1, 2):
        axes[axis:rotary.mrope_section[axis]*3:3] = axis
    if [int((axes == a).sum()) for a in range(3)] != list(rotary.mrope_section):
        raise ValueError("unexpected Qwen3 native interleaving")
    return axes


@torch.no_grad()
def representatives(V, diag, slots=4):
    """PPE source IDs: closest-to-centroid members, repeated when undersized.

The position operator is independent of VisionZip's aggregation. In particular
we do not quietly replace anchor+mean with unweighted cluster mean in this arm.
"""
    clusters = diag["tensors"]["cluster_id"]
    keep = diag["tensors"]["keep_idx"]
    rows = []
    anchors = diag["tensors"]["anchor_idx"]
    anchor_to_cluster = {int(anchor): row for row, anchor in enumerate(anchors)}
    for token in keep.tolist():
        members = (torch.where(clusters == anchor_to_cluster[token])[0]
                   if token in anchor_to_cluster else keep.new_tensor([token]))
        if not members.numel():
            raise ValueError("empty PPE cluster")
        feats = V[members].float()
        distances = (feats - feats.mean(0)).square().sum(-1)
        ranked = members[torch.argsort(distances, stable=True)]
        chosen = ranked[:slots]
        if chosen.numel() < slots:
            chosen = torch.cat([chosen, chosen[0].expand(slots-chosen.numel())])
        rows.append(chosen)
    result = torch.stack(rows)
    if bool((result < 0).any() or (result >= V.shape[0]).any()):
        raise AssertionError("PPE representative outside original image")
    for token, chosen in zip(keep.tolist(),result):
        if token in anchor_to_cluster:
            if not bool((clusters[chosen] == anchor_to_cluster[token]).all()):
                raise AssertionError("PPE position came from another cluster")
        elif not bool((chosen == token).all()):
            raise AssertionError("dominant singleton changed position")
    return result


@contextmanager
def ppe_prefill(eng, prep, keep, reps, slots=4):
    """Replace only image cos/sin during prefill; decode uses stock positions.

Each native axis retains its original frequencies and splits its own ordered
frequency indices into ``slots`` equal parts. A singleton is exactly stock.
Text positions, all DeepStack injections and the native decode delta stay intact.
"""
    rotary = eng.text.rotary_emb
    axes = native_axis_map(rotary)
    slot_map = torch.empty_like(axes)
    for axis in range(3):
        idx = torch.where(axes == axis)[0]
        if idx.numel() % slots:
            raise ValueError("PPE slots must divide each native mRoPE section")
        slot_map[idx] = torch.arange(slots, device=idx.device).repeat_interleave(idx.numel()//slots)
    if reps.shape != (keep.numel(), slots):
        raise ValueError("bad PPE source shape")
    fullpos = eng.positions_full(prep)[0][:, 0]
    keepseq = eng.keep_sequence(prep, keep)
    vis_slots = torch.where((keepseq >= prep["img_start"]) & (keepseq < prep["img_end"]))[0]
    sources = reps[:, slot_map] + prep["img_start"]
    source_positions = fullpos[axes[None, :], sources]
    phase = source_positions.float() * rotary.inv_freq.float()[None, :]
    phase = torch.cat((phase, phase), -1)
    called = []
    def replace(module, args, output):
        cos, sin = output
        if cos.shape[1] != keepseq.numel():
            raise ValueError("PPE hook may only cover one prefill")
        newcos, newsin = cos.clone(), sin.clone()
        newcos[:, vis_slots] = (phase.cos() * module.attention_scaling).to(cos.dtype)[None]
        newsin[:, vis_slots] = (phase.sin() * module.attention_scaling).to(sin.dtype)[None]
        called.append(True)
        return newcos, newsin
    hook = rotary.register_forward_hook(replace)
    try:
        yield called
        if len(called) != 1:
            raise AssertionError(f"PPE prefill rotary calls={len(called)}, expected one")
    finally:
        hook.remove()


@torch.no_grad()
def predict(eng, prep, V, DS, selection=None, ppe=False, max_tokens=24):
    if selection is None:
        state = eng.prefill(prep, V, DS, None)
    else:
        keep, Vs, DSs, diag = selection
        if ppe:
            reps = representatives(V, diag)
            with ppe_prefill(eng, prep, keep, reps):
                state = eng.prefill(prep, V, DS, keep, V_sel=Vs, DS_sel=DSs)
        else:
            state = eng.prefill(prep, V, DS, keep, V_sel=Vs, DS_sel=DSs)
        if state.n_vis != keep.numel() or state.ds_lengths != [keep.numel()]*3:
            raise AssertionError("native feature/DeepStack budget mismatch")
    tokens, text = eng.decode(state, max_tokens)
    info = dict(n_vis=state.n_vis, ds_lengths=state.ds_lengths,
                truncated=len(tokens)>=max_tokens, decode_positions=state.decode_positions,
                expected_decode_start=state.prefill_max_pos+1)
    if state.decode_positions and state.decode_positions[0] != state.prefill_max_pos+1:
        raise AssertionError("native decode position drift")
    return text, info
