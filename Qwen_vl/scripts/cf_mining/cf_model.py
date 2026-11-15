"""Native Qwen3-VL training plumbing for counterfactual mining experiments.

This module supplies a small *baseline* compression parameterization, not a
claim of architectural novelty.  B2's selected tokens are fixed anchors.  One
shared attention map pools the final visual stream and every DeepStack stream,
and a bounded residual gate starts at exact B2 identity.  The stock text model
handles both differentiable teacher forcing and ordinary native generation.

Cache contract (one question/image per record, tensors normally stored on CPU)::

    {"ids": LongTensor[1,L], "pos3d": LongTensor[3,1,L],
     "gthw": LongTensor[1,3], "img_start": int, "img_end": int,
     "n_vis": int, "V": Tensor[N,D], "DS": list[Tensor[N,D]],
     "keep": LongTensor[K], "answer": str, "alt_answer": str (optional)}

``gthw`` is needed only for generation through ``NativeEngine.prefill``;
teacher forcing uses cached original positions directly.  Anchor coordinates
and all original text positions are preserved.  Answer tokens start at the
maximum *full prompt* position plus one, matching the native decode convention.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AnchorPool(nn.Module):
    """Trainable residual pooling around a fixed raster-ordered anchor set.

    Random, nonzero Q/K weights avoid a dead attention parameterization.  At
    initialization ``gate == 0`` and the output is bit-identical to the anchors;
    the gate derivative is nonzero.  Q/K receive gradients after the gate moves
    away from zero.  Attention and residual arithmetic run in float32, and each
    output stream is converted back to its original dtype.
    """

    def __init__(self, d_model: int, rank: int = 32, gate_bound: float = 1.0):
        super().__init__()
        if d_model < 1 or rank < 1 or gate_bound <= 0:
            raise ValueError("d_model, rank, and gate_bound must be positive")
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.gate_bound = float(gate_bound)
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    @property
    def gate(self) -> torch.Tensor:
        return self.gate_bound * self.alpha.tanh()

    def attention(self, V: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if V.ndim != 2 or V.shape[1] != self.d_model:
            raise ValueError(f"V must have shape [N, {self.d_model}]")
        # Parameter storage remains float32, including when V is bf16.
        x = F.layer_norm(V.float(), (self.d_model,))
        q = F.linear(x[keep], self.query.weight.float())
        k = F.linear(x, self.key.weight.float())
        return torch.softmax((q @ k.T) / math.sqrt(self.rank), dim=-1)

    def pool(self, V: torch.Tensor, DS: Optional[Sequence[torch.Tensor]],
             keep: torch.Tensor):
        """Return ``(V_sel, DS_sel)`` with the same K ordered anchors.

        The weights are computed once from V and reused for every stream.
        ``keep`` must be sorted, unique, and in range, as in native B2.
        """
        keep = _checked_keep(keep, int(V.shape[0]), V.device)
        weights = self.attention(V, keep)
        gate = self.gate.float()

        def mix(stream):
            if stream.shape != V.shape:
                raise ValueError("all DeepStack streams must match V's shape")
            anchor = stream[keep].float()
            return (anchor + gate * (weights @ stream.float() - anchor)).to(stream.dtype)

        return mix(V), None if DS is None else [mix(stream) for stream in DS]

    def forward(self, V, DS, keep):
        return self.pool(V, DS, keep)


class AnchorReadout(AnchorPool):
    """B2-initialized low-rank reading interfaces for V and three DS streams.

    This generic carrier adds capacity to ``AnchorPool`` without changing its
    anchor positions or shared attention map.  Each stream gets a separate
    random value projection and zero-initialized output projection.  Thus all
    four outputs start at exact B2 identity, output projections can learn on
    the first update, and value/attention gradients become available once an
    output projection moves.  This architecture is not a novelty claim.
    """

    def __init__(self, d_model: int, rank: int = 32, gate_bound: float = 1.0):
        super().__init__(d_model, rank, gate_bound)
        self.values = nn.ModuleList(
            nn.Linear(d_model, rank, bias=False) for _ in range(4))
        self.readouts = nn.ModuleList(
            nn.Linear(rank, d_model, bias=False) for _ in range(4))
        for readout in self.readouts:
            nn.init.zeros_(readout.weight)

    def pool(self, V: torch.Tensor, DS: Optional[Sequence[torch.Tensor]],
             keep: torch.Tensor):
        keep = _checked_keep(keep, int(V.shape[0]), V.device)
        streams = [V] + ([] if DS is None else list(DS))
        if len(streams) > 4:
            raise ValueError("AnchorReadout supports V and at most three DeepStack streams")
        weights = self.attention(V, keep)
        gate = self.gate.float()
        outputs = []
        for stream, value, readout in zip(streams, self.values, self.readouts):
            if stream.shape != V.shape:
                raise ValueError("all DeepStack streams must match V's shape")
            anchor = stream[keep].float()
            pooled = anchor + gate * (weights @ stream.float() - anchor)
            normalized = F.layer_norm(stream.float(), (self.d_model,))
            latent = F.linear(normalized, value.weight.float())
            residual = F.linear(weights @ latent, readout.weight.float())
            outputs.append((pooled + residual).to(stream.dtype))
        return outputs[0], None if DS is None else outputs[1:]


class LocalReadout(AnchorReadout):
    """AnchorReadout with a fixed local spatial prior in shared attention.

    Coordinates are the original square merged-token raster, in token units.
    The Gaussian logit bias encourages each anchor to read nearby evidence;
    Q/K remain trainable and every output still starts at exact B2 identity.
    """

    def __init__(self, d_model: int, rank: int = 32,
                 gate_bound: float = 1.0, sigma: float = 1.5):
        super().__init__(d_model, rank, gate_bound)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        self.sigma = float(sigma)

    def attention(self, V: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if V.ndim != 2 or V.shape[1] != self.d_model:
            raise ValueError(f"V must have shape [N, {self.d_model}]")
        n_vis = int(V.shape[0])
        side = math.isqrt(n_vis)
        if side * side != n_vis:
            raise ValueError("LocalReadout requires a square merged-token raster")
        x = F.layer_norm(V.float(), (self.d_model,))
        q = F.linear(x[keep], self.query.weight.float())
        k = F.linear(x, self.key.weight.float())
        raster = torch.arange(n_vis, device=V.device)
        coordinates = torch.stack((raster // side, raster % side), dim=-1).float()
        distance2 = (coordinates[keep, None, :] - coordinates[None, :, :]).square().sum(-1)
        logits = (q @ k.T) / math.sqrt(self.rank)
        logits = logits - distance2 / (2.0 * self.sigma ** 2)
        return torch.softmax(logits, dim=-1)


# Short alias for runners; the descriptive class name also documents checkpoints.
Compressor = AnchorPool


def _checked_keep(keep, n_vis: int, device) -> torch.Tensor:
    keep = torch.as_tensor(keep, dtype=torch.long, device=device)
    if keep.ndim != 1 or keep.numel() == 0:
        raise ValueError("keep must be a nonempty 1-D index tensor")
    if int(keep[0]) < 0 or int(keep[-1]) >= n_vis:
        raise ValueError("keep index outside the visual span")
    if bool((keep[1:] <= keep[:-1]).any()):
        raise ValueError("keep must be sorted and duplicate-free")
    return keep


def record_to_device(record: dict, device) -> dict:
    """Copy the cache record's tensors to a device, retaining metadata."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value
    return {key: move(value) for key, value in record.items()}


def cache_record(eng, prep: dict, V: torch.Tensor, DS,
                 keep: torch.Tensor, answer: str,
                 alt_answer: Optional[str] = None) -> dict:
    """Create a detached CPU record after a frozen NativeEngine.encode pass."""
    positions, _ = eng.positions_full(prep)
    rec = {key: prep[key] for key in
           ("ids", "gthw", "img_start", "img_end", "n_vis")}
    rec.update(pos3d=positions, V=V.detach(),
               DS=None if DS is None else [x.detach() for x in DS],
               keep=keep.detach(), answer=str(answer))
    if alt_answer is not None:
        rec["alt_answer"] = str(alt_answer)
    return record_to_device(rec, "cpu")


def freeze_decoder(eng, checkpointing: bool = False):
    """Freeze all VLM parameters; only a separately created pool is trained.

    This leaves gradients enabled for visual inputs.  Checkpointing is explicit
    and reversible via ``set_decoder_checkpointing``.  Never call the text
    forward inside ``no_grad`` when training the compressor.
    """
    eng.model.requires_grad_(False)
    eng.model.eval()
    set_decoder_checkpointing(eng, checkpointing)


def set_decoder_checkpointing(eng, enabled: bool):
    """Toggle stock non-reentrant layer checkpointing, without monkeypatching.

    Transformers activates checkpointing only in training mode.  Qwen3-VL's
    decoder uses zero attention dropout in the native experiment configuration;
    reject nonzero dropout rather than silently changing the training target.
    Call this function with False before native inference if it was enabled.
    """
    if enabled:
        if float(getattr(eng.text.config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("checkpointed frozen decoding requires zero attention dropout")
        eng.text.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        eng.text.train()
    else:
        eng.text.gradient_checkpointing_disable()
        eng.text.eval()


def answer_token_ids(eng, answer: str, device=None) -> torch.Tensor:
    """Tokenize an answer without BOS and append tokenizer EOS once.

    ``forward_answer`` returns exactly one logit row per returned target ID.
    Including EOS makes empty answers valid and supervises answer termination.
    """
    ids = eng.tok.encode(str(answer), add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    ids = list(ids)
    eos = eng.tok.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    if not ids or ids[-1] != int(eos):
        ids.append(int(eos))
    return torch.tensor(ids, dtype=torch.long, device=device)


def selected_streams(record: dict, compressor=None, keep_all: bool = False):
    """Return selected streams and anchor indices for an already moved record."""
    V, DS = record["V"], record.get("DS")
    n_vis = int(record["n_vis"])
    if V.ndim != 2 or V.shape[0] != n_vis:
        raise ValueError("V does not match n_vis")
    if keep_all:
        keep = torch.arange(n_vis, device=V.device, dtype=torch.long)
        return V, DS, keep
    keep = _checked_keep(record["keep"], n_vis, V.device)
    if compressor is None:
        return V[keep], None if DS is None else [ds[keep] for ds in DS], keep
    V_sel, DS_sel = compressor.pool(V, DS, keep)
    return V_sel, DS_sel, keep


def _teacher_forcing_inputs(eng, record, V_sel, DS_sel, keep, targets):
    ids = record["ids"]
    device = V_sel.device
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("native cache currently supports batch size 1")
    if ids.device != device:
        raise ValueError("move the cache with record_to_device before forwarding")
    positions = record["pos3d"]
    if tuple(positions.shape) != (3, 1, ids.shape[1]):
        raise ValueError("pos3d must describe the complete original prompt")
    start, end = int(record["img_start"]), int(record["img_end"])
    if end - start != int(record["n_vis"]):
        raise ValueError("only one contiguous visual span is supported")
    full_len = int(ids.shape[1])
    seq = torch.cat((torch.arange(start, device=device),
                     start + keep,
                     torch.arange(end, full_len, device=device)))
    prompt_len = int(seq.numel())
    # Append answer prefix to the full prompt, then compact the original image
    # span.  The final target needs no input row: it is predicted by its prefix.
    all_ids = torch.cat((ids, targets[:-1].view(1, -1)), dim=1)
    all_seq = torch.cat((seq, torch.arange(full_len, all_ids.shape[1], device=device)))
    emb = eng.inner.get_input_embeddings()(all_ids)[0, all_seq]
    vis_slots = (all_seq >= start) & (all_seq < end)
    emb = emb.index_copy(0, vis_slots.nonzero(as_tuple=True)[0], V_sel.to(emb.dtype))
    prefix_pos = positions[:, :, seq]
    appended = positions.max() + 1 + torch.arange(targets.numel() - 1, device=device)
    appended = appended.view(1, 1, -1).expand(3, 1, -1)
    pos3d = torch.cat((prefix_pos, appended), dim=-1)
    return dict(inputs_embeds=emb.unsqueeze(0),
                attention_mask=torch.ones((1, emb.shape[0]), device=device, dtype=torch.long),
                position_ids=pos3d, visual_pos_masks=vis_slots.unsqueeze(0),
                deepstack_visual_embeds=DS_sel, use_cache=False), prompt_len


def forward_answer(eng, record: dict, compressor=None,
                   answer: Optional[str] = None, keep_all: bool = False) -> torch.Tensor:
    """Differentiable teacher-forced answer logits, shape ``[T, vocab]``.

    Targets are ``answer_token_ids(eng, answer)`` (answer plus EOS).  With no
    compressor this is B2; with ``keep_all=True`` it is the full teacher.  The
    record and compressor must already be on the decoder device.  Frozen model
    parameters do not prevent gradients through its operations into the pool.
    Call under ``torch.enable_grad()`` when using VLMEvalKit, whose vlm import
    disables autograd globally; the experiment runner does this explicitly.
    """
    answer = record["answer"] if answer is None else answer
    V_sel, DS_sel, keep = selected_streams(record, compressor, keep_all)
    targets = answer_token_ids(eng, answer, V_sel.device)
    inputs, prompt_len = _teacher_forcing_inputs(
        eng, record, V_sel, DS_sel, keep, targets)
    out = eng.text(**inputs)
    answer_hidden = out.last_hidden_state[:, prompt_len - 1:prompt_len - 1 + targets.numel()]
    return eng.model.lm_head(answer_hidden)[0]


def answer_loss(eng, record: dict, compressor=None,
                answer: Optional[str] = None, keep_all: bool = False,
                reduction: str = "mean") -> torch.Tensor:
    """Cross entropy over answer and EOS; ordinary training control objective."""
    answer = record["answer"] if answer is None else answer
    logits = forward_answer(eng, record, compressor, answer, keep_all)
    targets = answer_token_ids(eng, answer, logits.device)
    return F.cross_entropy(logits.float(), targets, reduction=reduction)


def contrastive_answer_loss(eng, record: dict, compressor,
                            positive_answer: str, negative_answer: str,
                            margin: float = 1.0) -> torch.Tensor:
    """Differentiate correct and opposing answers on the same image.

    This is ``softplus(margin - (logp_positive - logp_negative))`` using
    *summed* full-sequence log probabilities, including EOS.  Both answers are
    teacher-forced through separate differentiable stock-decoder forwards;
    common-prefix contributions therefore cancel mathematically.  Call inside
    ``torch.enable_grad()`` if the host application disabled gradients.
    """
    positive_answer, negative_answer = str(positive_answer), str(negative_answer)
    if positive_answer == negative_answer:
        raise ValueError("contrastive answers must differ")

    def sequence_logp(answer):
        logits = forward_answer(eng, record, compressor, answer=answer)
        ids = answer_token_ids(eng, answer, logits.device)
        return -F.cross_entropy(logits.float(), ids, reduction="sum")

    positive = sequence_logp(positive_answer)
    negative = sequence_logp(negative_answer)
    return F.softplus(float(margin) - (positive - negative))


@torch.no_grad()
def candidate_logprobs(eng, record: dict, candidates: Sequence[str],
                       compressor=None, keep_all: bool = False,
                       length_normalize: bool = False) -> torch.Tensor:
    """Answer+EOS sequence log probabilities used by the training-only miner.

    Full-teacher verification uses ``keep_all=True``.  Defaults are summed
    sequence log probabilities; normalization, if requested, includes EOS.
    """
    values = []
    for answer in candidates:
        logits = forward_answer(eng, record, compressor, answer, keep_all)
        targets = answer_token_ids(eng, answer, logits.device)
        token_logp = F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None])[:, 0]
        values.append(token_logp.mean() if length_normalize else token_logp.sum())
    if not values:
        return torch.empty(0, device=record["V"].device)
    return torch.stack(values)


teacher_candidate_logprobs = candidate_logprobs


@torch.no_grad()
def generate_answer(eng, record: dict, compressor=None, keep_all: bool = False,
                    max_new_tokens: int = 32) -> str:
    """Greedy native generation, including all selected DeepStack streams."""
    if record.get("gthw") is None:
        raise ValueError("cache gthw for generation through NativeEngine.prefill")
    prep = {key: record[key] for key in
            ("ids", "gthw", "img_start", "img_end", "n_vis")}
    V_sel, DS_sel, keep = selected_streams(record, compressor, keep_all)
    st = eng.prefill(prep, record["V"], record.get("DS"), keep,
                     deepstack=True, pos="mrope3d", V_sel=V_sel, DS_sel=DS_sel)
    _, answer = eng.decode(st, max_new_tokens=max_new_tokens)
    return answer
