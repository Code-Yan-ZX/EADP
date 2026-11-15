"""Local additive-logit interventions for native Qwen3-VL SDPA attention.

This module leaves the installed attention forward intact: projection, q/k
normalization, rotary embeddings, cache update and output projection remain
native. Hooks only supply an additive attention mask. An identity experiment
without an observer installs no hooks. Zero corrections with an observer also
return the original call unchanged; ``force_mask`` is a separate SDPA-kernel
control, NOT an identity claim on GPU.

Scope: inference, batch one, an unpadded compact prefill followed by append-only
DynamicCache decode. Layer-wise routing, sliding/static caches and training are
rejected. Coordinates are metadata for callbacks; no RoPE positions are edited.
Observer attention is recomputed from the same local queries/keys, not a dense
teacher trajectory, and never substitutes for the actual SDPA output.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
from typing import Callable, Mapping, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


@dataclass
class AttentionLayout:
    """Metadata in COMPACT cache order, including all text positions.

    ``key_weights`` is positive effective multiplicity for each visual token,
    either length K (visual-only, raster order) or full compact prefix length.
    Text weights must be one. ``positions_3d`` has shape [3, prefix_length].
    NativeEngine decode uses scalar position ``cache_position + rope_delta``.
    """
    visual_mask: torch.Tensor
    positions_3d: torch.Tensor
    rope_delta: int
    key_weights: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.visual_mask = torch.as_tensor(self.visual_mask, dtype=torch.bool).detach().cpu().clone()
        self.positions_3d = torch.as_tensor(self.positions_3d, dtype=torch.long).detach().cpu().clone()
        if self.visual_mask.ndim != 1 or self.positions_3d.shape != (3, self.visual_mask.numel()):
            raise ValueError("layout requires [prefix_length] visual mask and [3,prefix_length] positions")
        if not self.visual_mask.numel():
            raise ValueError("empty prefix")
        weights = torch.ones(self.visual_mask.numel(), dtype=torch.float32)
        if self.key_weights is not None:
            supplied = torch.as_tensor(self.key_weights, dtype=torch.float32).detach().cpu()
            if supplied.ndim != 1:
                raise ValueError("key_weights must be one-dimensional")
            if supplied.numel() == self.visual_mask.numel():
                weights = supplied.clone()
            elif supplied.numel() == int(self.visual_mask.sum()):
                weights[self.visual_mask] = supplied
            else:
                raise ValueError("key_weights length does not match K or compact prefix")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("key_weights must be finite and positive")
        if not bool((weights[~self.visual_mask] == 1).all()):
            raise ValueError("text key weights must be one")
        self.key_weights = weights
        self.rope_delta = int(self.rope_delta)

    @classmethod
    def from_engine(cls, eng, prep, keep_idx, key_weights=None):
        """Read NativeEngine metadata; does not encode, select or alter streams."""
        keep_seq = eng.keep_sequence(prep, keep_idx)
        fullpos = eng.positions_full(prep)[0]
        vis = (keep_seq >= prep["img_start"]) & (keep_seq < prep["img_end"])
        return cls(vis, fullpos[:, 0, keep_seq], int(fullpos.max()) + 1 - keep_seq.numel(), key_weights)

    def arrays(self, key_length, device):
        prefix = self.visual_mask.numel()
        if key_length < prefix:
            raise ValueError("key sequence shorter than fixed compact prefix: routing/chunked prefill unsupported")
        extra = key_length - prefix
        visual = torch.cat((self.visual_mask, torch.zeros(extra, dtype=torch.bool))).to(device)
        weights = torch.cat((self.key_weights, torch.ones(extra))).to(device)
        generated = torch.arange(prefix, key_length, dtype=torch.long) + self.rope_delta
        positions = torch.cat((self.positions_3d, generated.expand(3, -1)), dim=1).to(device)
        return visual, weights, positions


@dataclass
class AttentionFrame:
    layer_idx: int
    num_heads: int
    cache_position: torch.Tensor       # compact, causal positions [Q]
    query_positions_3d: torch.Tensor  # original/native coordinates [3,Q]
    key_positions_3d: torch.Tensor    # [3,K]
    query_is_visual: torch.Tensor    # [Q]
    key_is_visual: torch.Tensor      # [K]
    key_weights: torch.Tensor        # [K], generated/text tokens have weight one
    dtype: torch.dtype
    device: torch.device

    @property
    def query_length(self):
        return self.cache_position.numel()

    @property
    def key_length(self):
        return self.key_is_visual.numel()

    @property
    def is_prefill(self):
        return self.query_length == self.key_length


@dataclass
class BiasSpec:
    """All terms are additive in PRE-softmax logits, after QK scaling.

    ``layer_head_bias`` maps zero-based layer to scalar or one value per query
    head. Those values ADD to ``visual_bias`` on visual keys.
    ``pair_bias`` is a static tensor or callback(frame)->tensor; supported
    shapes broadcast to [1,H,Q,K], e.g. [Q,K], [H,Q,K], [K]. It may use native
    3D coordinates from the frame (e.g. a distance correction).
    ``query_scope='text'`` applies built-in weight/visual terms only to text
    queries; pair_bias is explicit and remains unrestricted.
    """
    log_count_scale: float = 0.0
    visual_bias: float = 0.0
    layer_head_bias: Mapping[int, object] = field(default_factory=dict)
    pair_bias: Optional[object] = None
    force_mask: bool = False
    query_scope: str = "all"
    prefill_only: bool = False

    def __post_init__(self):
        if self.query_scope not in ("all", "text"):
            raise ValueError("query_scope must be all or text")
        if not bool(torch.isfinite(torch.tensor([self.log_count_scale, self.visual_bias])).all()):
            raise ValueError("nonfinite scalar bias")
        normalized = {}
        for layer, bias in self.layer_head_bias.items():
            b = torch.as_tensor(bias, dtype=torch.float32).detach().cpu().clone()
            if b.ndim > 1 or not bool(torch.isfinite(b).all()):
                raise ValueError("head biases must be finite scalars/vectors")
            normalized[int(layer)] = b
        self.layer_head_bias = normalized

    def can_change_layer(self, layer_idx, layout):
        heads = self.layer_head_bias.get(layer_idx, torch.tensor(0.0))
        return (self.force_mask or self.pair_bias is not None or self.visual_bias != 0
                or bool((heads != 0).any())
                or (self.log_count_scale != 0 and bool((layout.key_weights != 1).any())))


def build_logit_bias(frame: AttentionFrame, spec: BiasSpec):
    """Return [1,H-or-1,Q-or-1,K] bias, preserving broadcasting when possible."""
    key_term = spec.log_count_scale * frame.key_weights.log()
    head = spec.layer_head_bias.get(frame.layer_idx)
    if head is None:
        head = torch.zeros(1, device=frame.device)
    else:
        head = head.to(frame.device).reshape(-1)
        if head.numel() not in (1, frame.num_heads):
            raise ValueError("per-head bias length differs from number of query heads")
    base = key_term[None, :] + (head[:, None] + spec.visual_bias) * frame.key_is_visual[None, :]
    bias = base[None, :, None, :]
    if spec.query_scope == "text":
        bias = bias * (~frame.query_is_visual)[None, None, :, None]
    if spec.pair_bias is not None:
        extra = spec.pair_bias(frame) if callable(spec.pair_bias) else spec.pair_bias
        extra = torch.as_tensor(extra, device=frame.device, dtype=torch.float32)
        if extra.ndim > 4 or not bool(torch.isfinite(extra).all()):
            raise ValueError("pair bias must be finite and broadcastable to [1,H,Q,K]")
        target = (1, frame.num_heads, frame.query_length, frame.key_length)
        try:
            if torch.broadcast_shapes(extra.shape, target) != target:
                raise ValueError("pair bias broadcasts beyond the attention shape")
            bias = bias + extra
        except RuntimeError as exc:
            raise ValueError("pair bias not broadcastable to [1,H,Q,K]") from exc
    return bias


def combine_attention_mask(attention_mask, bias, cache_position, key_length, dtype):
    """Add bias without reopening causal/padding exclusions.

    When native SDPA omitted its mask, materialize causal masking using CACHE
    order, not 3D RoPE coordinates. For cached multi-query calls this is the
    lower-right alignment rather than an upper-left Q-by-K triangle.
    """
    qlen = cache_position.numel()
    if attention_mask is None:
        allowed = torch.arange(key_length, device=cache_position.device)[None, :] <= cache_position[:, None]
        base = torch.zeros((1, 1, qlen, key_length), device=bias.device, dtype=torch.float32)
        base = base.masked_fill(~allowed[None, None], float("-inf"))
    else:
        if attention_mask.ndim != 4 or attention_mask.shape[-1] < key_length:
            raise ValueError("only native 4D SDPA masks or None are supported")
        if attention_mask.shape[-2] not in (1, qlen):
            raise ValueError("mask query length differs")
        supplied = attention_mask[..., :key_length]
        if supplied.dtype == torch.bool:
            base = torch.zeros_like(supplied, dtype=torch.float32).masked_fill(~supplied, float("-inf"))
        else:
            base = supplied.float()
            # Native finfo.min sentinels should remain excluded after addition.
            excluded = ~torch.isfinite(base) | (base <= torch.finfo(supplied.dtype).min / 2)
            base = base.masked_fill(excluded, float("-inf"))
    result = base + bias.float()
    return result.to(dtype)


class VisualMassObserver:
    """Read-only last-text-query diagnostics, optionally on selected layers.

    ``records`` contain per-query-head visual probability mass. ``records``
    store CPU tensors; diagnostic synchronization/extra projections must not
    be included in method latency measurements. Native and corrected records
    both refer to the SAME local trajectory Q/K for that arm.
    """
    def __init__(self, layers=None):
        self.layers = None if layers is None else frozenset(layers)
        self.records = []

    def wants(self, layer_idx):
        return self.layers is None or layer_idx in self.layers

    @torch.no_grad()
    def capture(self, module, arguments, frame, corrected_mask):
        text_queries = torch.where(~frame.query_is_visual)[0]
        if not text_queries.numel():
            return
        index = int(text_queries[-1])
        hidden = arguments["hidden_states"]
        head_dim = module.head_dim
        # Recompute one Q row and current K rows; never call cache.update.
        # Keep the actual projection's full GEMM shape: projecting only one row
        # can change bf16 rounding on GPU, even for a read-only diagnostic.
        q = module.q_norm(module.q_proj(hidden).view(1, hidden.shape[1], -1, head_dim)).transpose(1, 2)
        q = q[:, :, index:index+1]
        k = module.k_norm(module.k_proj(hidden).view(1, hidden.shape[1], -1, head_dim)).transpose(1, 2)
        cos, sin = arguments["position_embeddings"]
        # apply_rotary_pos_emb broadcasts q/k separately with their own rows.
        q, _ = apply_rotary_pos_emb(q, q, cos[:, index:index+1], sin[:, index:index+1])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        cache = arguments.get("past_key_values")
        if cache is not None and cache.get_seq_length(module.layer_idx):
            previous = cache.layers[module.layer_idx].keys
            k = torch.cat((previous, k), dim=-2)
        k = repeat_kv(k, module.num_key_value_groups)
        # Float32 diagnostics avoid bf16 softmax rounding. They are not claimed
        # to reproduce the selected fused SDPA kernel's internal accumulation.
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * module.scaling
        original_mask = combine_attention_mask(arguments.get("attention_mask"),
            torch.zeros((1, 1, 1, frame.key_length), device=frame.device),
            frame.cache_position, frame.key_length, torch.float32)
        local_original = original_mask[..., index:index+1, :] if original_mask.shape[-2] > 1 else original_mask
        original_prob = (logits + local_original).softmax(-1)
        actual = original_mask if corrected_mask is None else corrected_mask.float()
        local_actual = actual[..., index:index+1, :] if actual.shape[-2] > 1 else actual
        actual_prob = (logits + local_actual).softmax(-1)
        visual = frame.key_is_visual
        self.records.append(dict(layer_idx=frame.layer_idx,
            cache_position=int(frame.cache_position[index]), query_index=index,
            query_positions_3d=frame.query_positions_3d[:, index].detach().cpu(),
            key_length=frame.key_length, prefill=frame.query_length > 1,
            original_visual_mass=original_prob[0, :, 0, visual].sum(-1).detach().cpu(),
            corrected_visual_mass=actual_prob[0, :, 0, visual].sum(-1).detach().cpu(),
            source="same_local_QK_float32_read_only_recomputation"))


@contextmanager
def attention_calibration(text_model, layout: AttentionLayout, spec: Optional[BiasSpec] = None,
                          observer: Optional[VisualMassObserver] = None):
    """Apply only to ``text_model.layers[*].self_attn`` for prefill AND decode.

    Example::

        layout = AttentionLayout.from_engine(eng, prep, keep, key_weights=counts)
        with attention_calibration(eng.text, layout, BiasSpec(log_count_scale=1)):
            state = eng.prefill(prep, V, DS, keep, V_sel=Vs, DS_sel=DSs)
            tokens, answer = eng.decode(state, max_tokens)

    Existing PPE rotary hooks can be nested without changes. Do not leave the
    context before decoding if decode should receive the correction.
    """
    spec = BiasSpec() if spec is None else spec
    if text_model.training:
        raise ValueError("attention calibration supports inference/eval only")
    if getattr(text_model.config, "_attn_implementation", None) != "sdpa":
        raise ValueError("this adapter requires the native SDPA text backend")
    if getattr(text_model, "_attention_calibration_active", False):
        raise ValueError("nested attention calibration on one model is unsupported")
    handles = []
    text_model._attention_calibration_active = True
    try:
        for layer in text_model.layers:
            module = layer.self_attn
            active = spec.can_change_layer(module.layer_idx, layout)
            observe = observer is not None and observer.wants(module.layer_idx)
            if not active and not observe:
                continue
            signature = inspect.signature(module.forward)

            def hook(attn, args, kwargs, *, signature=signature, active=active, observe=observe):
                bound = signature.bind_partial(*args, **kwargs)
                arguments = dict(bound.arguments)
                # Qwen attention's **kwargs is not used to hide core arguments.
                hidden = arguments["hidden_states"]
                if hidden.shape[0] != 1:
                    raise ValueError("batch one only")
                cache = arguments.get("past_key_values")
                if cache is not None and (not isinstance(cache, DynamicCache)
                    or any(getattr(entry, "is_sliding", False) for entry in cache.layers)):
                    raise ValueError("only append-only DynamicCache is supported")
                past = 0 if cache is None else cache.get_seq_length(attn.layer_idx)
                qlen = hidden.shape[1]
                cp = arguments.get("cache_position")
                expected = torch.arange(past, past + qlen, device=hidden.device)
                cp = expected if cp is None else cp
                if cp.ndim != 1 or not torch.equal(cp, expected):
                    raise ValueError("cache positions must be contiguous append order")
                key_length = past + qlen
                visual, weights, positions = layout.arrays(key_length, hidden.device)
                frame = AttentionFrame(attn.layer_idx, attn.config.num_attention_heads, cp,
                    positions[:, cp], positions, visual[cp], visual, weights, hidden.dtype, hidden.device)
                corrected = None
                if active:
                    if spec.prefill_only and not frame.is_prefill:
                        # Preserve the matched explicit-mask backend control
                        # during decode while removing ALL mathematical bias,
                        # including log-size. Never invoke the pair callback.
                        bias = torch.zeros((1,1,1,key_length),device=hidden.device)
                    else:
                        bias = build_logit_bias(frame, spec)
                    # A callback may return zero. Preserve the exact native path
                    # unless force_mask explicitly requests the kernel control.
                    if spec.force_mask or bool((bias != 0).any()):
                        corrected = combine_attention_mask(arguments.get("attention_mask"), bias,
                            cp, key_length, hidden.dtype)
                if observe:
                    observer.capture(attn, arguments, frame, corrected)
                if corrected is None:
                    return None
                bound.arguments["attention_mask"] = corrected
                return bound.args, bound.kwargs

            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        yield observer
    finally:
        for handle in handles:
            handle.remove()
        del text_model._attention_calibration_active
