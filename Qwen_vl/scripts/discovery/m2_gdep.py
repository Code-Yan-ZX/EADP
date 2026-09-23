"""
M2 — Gradient-Distilled Early Pruning (GDEP): the inference engine.

This is the first module in the project that runs a **real** early-layer prune
rather than injecting a cached score into the incumbent's pre-LLM harness. It is
the single implementation used for **both** the accuracy run and the latency run,
which is what M2's pre-registration §6.4 requires.

What the engine does in ``gdep`` mode
-------------------------------------
    1. vision tower -> 1024 merged visual embeddings            (unchanged)
    2. splice into the prompt at the image-token positions
    3. decoder layers 0..4 over the FULL 1058-token sequence     (once)
    4. read the visual slice of the layer-4 residual stream, h4
    5. ONLINE scorer  s = w2 . GELU(P h4_hat + b)                 (524 673 params)
    6. selector -> 256 of 1024 visual indices
    7. compact hidden states, the layer-0..4 KV cache, position ids and mask
    8. decoder layers 5..35 over the kept 290 positions only
    9. normal greedy generation

L0-L4 run *once*. Nothing is recomputed, no cached score is read at inference
time, and the scorer is evaluated inside the measured region.

Three modes, one code path
--------------------------
    full    B0 -- no pruning, 36 layers over 1058 tokens.
    prellm  B1/B2 -- the incumbent: official EADP scoring + selector act on the
            vision tower's post-merger embeddings *before* any LLM layer runs,
            and the compacted sequence then goes through all 36 layers.
    gdep    C0/C1/C2/C3 -- the method under test.

Two inherited conventions, stated because they are degrees of freedom
---------------------------------------------------------------------
* **No deepstack.** ``s2c1_features.py``, ``m1_features.py`` and the incumbent's
  pruned path all hand the model ``inputs_embeds`` with no ``pixel_values``, so
  ``Qwen3VLModel`` never builds ``deepstack_visual_embeds`` and the deepstack
  features are not injected. GDEP keeps that convention in every mode, so the
  comparison is like-for-like. Gate G-A is what forces it: the online h4 must
  match the published M1 cache, and that cache was built without deepstack.
* **1-D text positions.** With ``inputs_embeds`` and no ``image_grid_thw``,
  ``get_rope_index`` takes its no-image branch and every position id is a plain
  running index. Same in every mode.

Position policy (prereg §2.2)
-----------------------------
``preserve``  a kept token carries the position id it had in the full sequence;
              the compacted ``position_ids`` are ``sorted(kept_indices)`` and are
              therefore not contiguous. The next generated token takes
              ``max(position_ids) + 1``.
``renumber``  the compacted sequence is assigned ``arange(0, L')``. This is what
              the incumbent's pre-LLM path does implicitly, so it is the policy
              that makes GDEP-vs-EADP like-for-like on positional encoding.
PRESERVE is the primary policy; RENUMBER is a declared control on C1 only.

KV-cache policy (prereg §2.3)
-----------------------------
``compact``   after selection the keys/values of the dropped visual tokens are
              removed from the layer-0..4 caches, so layers 5..35 attend only
              over the kept positions. This is the ONLY policy GDEP uses: it is
              what makes the pruning real and what reduces the KV footprint.
              The transient 1058-token allocation of layers 0..4 is measured
              separately from the steady-state 290-token footprint.
"""
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR, eadp_model_name, load_model          # noqa: E402
from instrumented import SELECTORS, TimedEADPPruner, attach_pruner  # noqa: E402
from model.pruner import _sim_visual_impl                           # noqa: E402

MODE_FULL = "full"
MODE_PRELLM = "prellm"
MODE_GDEP = "gdep"

POLICY_PRESERVE = "preserve"
POLICY_RENUMBER = "renumber"

D_MODEL = 4096
D_HID = 128
N_VIS = 1024
LAYER = 4                       # 0-based decoder layer index; h4 = output of layer 4

# Sentinel for "the caller did not supply an attention mask". Distinct from
# None, which is what create_causal_mask legitimately returns.
_AUTO = object()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class GDEPConfig:
    mode: str = MODE_GDEP
    layer: int = LAYER
    budget: int = 256
    selector: str = "topk"
    pos_policy: str = POLICY_PRESERVE
    n_arm: int = 960                # scorer fit-set size (240 or 960)
    seed: int = 2                   # scorer seed
    sim_source: str = "h4"          # only "h4" implemented for gdep
    tag: str = "C1"
    # Correctness-probe switch. When the selector keeps every token there is
    # nothing to compact, so the engine continues the layer loop in one pass and
    # the K = 1024 arm is provably the identity. `force_split` restores the
    # split-and-rejoin path so the size of that path's own numerical deviation
    # can be measured and reported rather than assumed (prereg G-C).
    force_split: bool = False

    def key(self) -> str:
        return (f"{self.tag}|{self.mode}|n{self.n_arm}|s{self.seed}|"
                f"{self.selector}|T{self.budget}|{self.pos_policy}")

    def hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.key().encode()).hexdigest()[:16]


def scorer_path(n_arm: int, seed: int, protocol: str = "fixed-step") -> str:
    return os.path.join(OUTPUT_DIR,
                        f"m1_{protocol}_n{n_arm}_L4_s{seed}__H8.pt")


def stats_path(n_arm: int, tag: str = "m1") -> str:
    return os.path.join(OUTPUT_DIR, f"{tag}_stats_n{n_arm}.npz")


def build_scorer(n_arm: int, seed: int, device, dtype=torch.float32):
    """Rebuild the 524 673-parameter M1 LOCAL-MLP and load its frozen weights."""
    from s2c5_models import LocalMLP

    path = scorer_path(n_arm, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalMLP(d_in=D_MODEL, d_hid=D_HID)
    model.load_state_dict(sd)
    model.eval().to(device=device, dtype=dtype)
    z = np.load(stats_path(n_arm))
    mu = torch.from_numpy(z["mu"].astype(np.float32)).to(device=device, dtype=dtype)
    sd_ = torch.from_numpy(z["sd"].astype(np.float32)).to(device=device, dtype=dtype)
    return model, mu, sd_


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class GDEPEngine:
    """Real early-layer pruning with a manual, fully instrumented layer loop."""

    def __init__(self, model, cfg: GDEPConfig = None, scorer=None,
                 mu=None, sd=None, device=None):
        self.vlm = model
        self.cfg = cfg or GDEPConfig()
        self.dev = device or next(model.model.parameters()).device
        self.model = model.model                       # Qwen3VLForConditionalGeneration
        self.text = self.model.model.language_model    # Qwen3VLTextModel
        self.layers = self.text.layers
        self.n_layers = len(self.layers)
        self.image_token_id = self.model.config.image_token_id
        self.scorer, self.mu, self.sd = scorer, mu, sd
        self.counts = {}
        self.last = {}
        # B1/B2 run the incumbent's own pruner, instrumented only for timing and
        # for the identical selector implementations the candidates use.
        self.pruner = None
        if self.cfg.mode == MODE_PRELLM:
            self.pruner = attach_pruner(model, selector=self.cfg.selector,
                                        capture=False)
            self.pruner.visual_token_num = self.cfg.budget
            self.pruner.sim_mode = "rebound"

    # ---------------------------------------------------------------- setup --
    @classmethod
    def from_checkpoint(cls, cfg: GDEPConfig, model=None, max_new_tokens: int = 2048):
        if model is None:
            model = load_model(eadp_model_name(cfg.budget, 0.5, 2.0),
                               max_new_tokens=max_new_tokens)
        model.model.eval()
        scorer = mu = sd = None
        if cfg.mode == MODE_GDEP:
            scorer, mu, sd = build_scorer(cfg.n_arm, cfg.seed,
                                          next(model.model.parameters()).device)
        return cls(model, cfg, scorer, mu, sd)

    # ------------------------------------------------------------ internals --
    def _causal(self, hidden, attn2d, cache_position, cache, pos_text):
        from transformers.masking_utils import create_causal_mask

        return create_causal_mask(
            config=self.text.config,
            input_embeds=hidden,
            attention_mask=attn2d,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=pos_text,
        )

    def _run(self, hidden, attn_mask, pos3, cache, cache_position, lo, hi,
             count=True):
        """Run decoder layers [lo, hi) and return the hidden states."""
        pos_text = pos3[0]
        pos_emb = self.text.rotary_emb(hidden, pos3)
        for i in range(lo, hi):
            hidden = self.layers[i](
                hidden,
                attention_mask=attn_mask,
                position_ids=pos_text,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=pos_emb,
            )
            if count:
                self.counts["layer_calls"] = self.counts.get("layer_calls", 0) + 1
        return hidden

    def _compact_cache(self, cache, upto, keep):
        """Drop rows `keep` complement from layers 0..upto-1 of the KV cache."""
        for i in range(upto):
            lay = cache.layers[i]
            lay.keys = lay.keys.index_select(2, keep)
            lay.values = lay.values.index_select(2, keep)
        return cache

    @staticmethod
    def _cache_bytes(cache, upto):
        tot = 0
        for i in range(upto):
            lay = cache.layers[i]
            tot += lay.keys.numel() * lay.keys.element_size()
            tot += lay.values.numel() * lay.values.element_size()
        return tot

    # ------------------------------------------------------------- the score --
    def score_online(self, h4):
        """s = w2 . GELU(P h4_hat + b) with the M1 standardisation."""
        x = h4.to(torch.float32)
        x = (x - self.mu) / self.sd
        with torch.no_grad():
            s = self.scorer(x.unsqueeze(0), None)
        return s[0].float()

    def select(self, scores, feats_for_sim, budget):
        sim = None
        if self.cfg.selector in ("block8", "block_greedy", "facility",
                                 "facility_fast", "lazy_greedy", "stochastic",
                                 "farthest", "topk_nms"):
            sim = _sim_visual_impl(feats_for_sim.unsqueeze(0).float())
        idx, extra = SELECTORS[self.cfg.selector](
            scores.unsqueeze(0), sim, budget)
        return idx[0].sort().values, extra

    # ------------------------------------------------------------ the prompt --
    def prepare(self, message, dataset_name, timings=None):
        """Processor inputs + the full-length prompt embeddings and image span."""
        self._pending_events = []
        t0 = time.perf_counter()
        inputs = self.vlm._processor_inputs(
            self.vlm._build_messages(message, dataset=dataset_name))
        if timings is not None:
            timings["image_preprocess_ms"] = (time.perf_counter() - t0) * 1e3

        ids = inputs["input_ids"]
        pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() > 0, "no image tokens in the prompt"
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == N_VIS, f"{e - s} visual tokens, expected {N_VIS}"

        pv = inputs["pixel_values"].type(self.model.visual.dtype)
        gthw = inputs["image_grid_thw"]
        with torch.no_grad():
            ev0 = ev1 = None
            if timings is not None:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            vis = unwrap_visual(self.model.visual(pv, grid_thw=gthw))
            if timings is not None:
                ev1.record()
                # read out in prefill's single sync, so the vision tower is not
                # serialised against the LLM work that follows it
                self._pending_events = [("vision_encoder_ms", ev0, ev1)]
            emb = self.model.get_input_embeddings()(ids)
        prompt = torch.cat([emb[:, :s], vis.unsqueeze(0), emb[:, e:]], dim=1)
        mask = torch.ones(1, prompt.shape[1], dtype=torch.long,
                          device=prompt.device)
        return dict(inputs=inputs, prompt=prompt, mask=mask, vis=vis,
                    vis_slice=(s, e), gthw=gthw, pv=pv)


    # --------------------------------------------------------------- prefill --
    @torch.no_grad()
    def prefill(self, prep, timings=None, injected_scores=None):
        """Run the prefill for this arm and return the decode-ready state.

        Timing keys, in the pre-registration's numbering:
            L0_L4_ms            stage 5  (GDEP only)
            scorer_ms           stage 6  (GDEP only)
            selector_ms         stage 4  (all pruned arms)
            eadp_scoring_ms     stage 3  (B1/B2 only)
            token_compaction_ms stage 7  (GDEP only)
            llm_forward_ms      stage 8  (the decoder layers that actually run:
                                         5..35 for GDEP, 0..35 for B0/B1/B2)
        """
        from transformers.cache_utils import DynamicCache

        cfg, dev = self.cfg, self.dev
        prompt, mask = prep["prompt"], prep["mask"]
        vis, (s, e) = prep["vis"], prep["vis_slice"]
        S = prompt.shape[1]
        L = cfg.layer
        cache = DynamicCache(config=self.text.config)
        self.counts = {}
        zero = ("L0_L4_ms", "scorer_ms", "selector_ms", "eadp_scoring_ms",
                "token_compaction_ms")
        info = dict(mode=cfg.mode, tag=cfg.tag, seq_full=int(S), vis_start=int(s),
                    vis_end=int(e), budget=None, n_kept=None, context_len=None,
                    n_text=int(S - (e - s)), pos_policy=cfg.pos_policy,
                    layer=L,
                    cache_policy=("compact_L0_L4" if cfg.mode == MODE_GDEP else
                                  "none_pre_llm" if cfg.mode == MODE_PRELLM else
                                  "unpruned"),
                    scorer_params=(sum(p.numel() for p in self.scorer.parameters())
                                   if self.scorer is not None else 0),
                    cfg_key=cfg.key(), cfg_hash=cfg.hash())

        _events = list(getattr(self, "_pending_events", []))
        self._pending_events = []

        def _step(name, fn):
            """Bracket a stage with CUDA events. No sync inside: the whole
            prefill is synchronised once, so the stages do not serialise each
            other and the breakdown stays a decomposition of the same run."""
            if timings is None:
                return fn()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            out = fn()
            b.record()
            _events.append((name, a, b))
            return out

        def _finish():
            if timings is None:
                return
            torch.cuda.synchronize()
            for n, a, b in _events:
                timings[n] = timings.get(n, 0.0) + a.elapsed_time(b)

        def _layers(hidden, pos1, cache, lo, hi, attn=_AUTO):
            """Run layers [lo, hi) over `hidden` at positions `pos1` (1-D).

            `attn` lets the caller reuse a mask that is known to be the right one
            for this shape and cache position, which is what makes the no-op
            compaction path byte-identical to a single uninterrupted pass.

            The default is the `_AUTO` sentinel, not `None`, because `None` is a
            *legitimate value* for this argument: `create_causal_mask` returns
            `None` whenever SDPA can take the `is_causal` path, which is exactly
            the case for a cache that is still empty. Treating a supplied `None`
            as "nothing supplied" silently rebuilt a materialised mask against a
            populated cache and cost the identity arm ~1 bf16 ulp -- which is
            what gate G-C caught.
            """
            n = hidden.shape[1]
            cp = torch.arange(n, device=dev)
            pos3 = pos1.view(1, 1, -1).expand(3, 1, -1)
            if attn is _AUTO:
                am = torch.ones(1, n, dtype=torch.long, device=dev)
                attn = self._causal(hidden, am, cp, cache, pos3[0])
            return self._run(hidden, attn, pos3, cache, cp, lo, hi)

        pos1_out = None
        h4_held = None
        scores_held = None

        # ---- B1/B2: prune the vision embeddings before any LLM layer ---------
        if cfg.mode == MODE_PRELLM:
            pruner = self.pruner
            pruner.visual_token_num = cfg.budget
            text_llm, text_seq = self._instruction_embeds(prep)

            def _prune():
                return pruner(vis, text_llm, text_seq, prep["gthw"])
            pruned, sizes = _step("selector_ms", _prune)
            st = dict(pruner.last_timing)
            info["stage_timing"] = st
            if timings is not None:
                timings["eadp_scoring_ms"] = float(sum(
                    v for k, v in st.items() if k != "facility_location"))
                timings["selector_ms"] = float(st.get("facility_location", 0.0))
            info["n_kept"] = int(sum(sizes))
            info["budget"] = cfg.budget
            hidden = torch.cat([prompt[:, :s], pruned.unsqueeze(0),
                                prompt[:, e:]], dim=1)
            info["context_len"] = int(hidden.shape[1])
            pos1_out = torch.arange(hidden.shape[1], device=dev)
            hidden = _step("llm_forward_ms",
                           lambda: _layers(hidden, pos1_out, cache, 0,
                                           self.n_layers))

        # ---- GDEP: layers 0..L at full length, score, select, compact --------
        elif cfg.mode == MODE_GDEP:
            pos_full = torch.arange(S, device=dev)
            pos3_full = pos_full.view(1, 1, -1).expand(3, 1, -1)
            am_full = torch.ones(1, S, dtype=torch.long, device=dev)
            attn_full = self._causal(prompt, am_full, pos_full, cache,
                                     pos3_full[0])
            hidden = _step("L0_L4_ms",
                           lambda: self._run(prompt, attn_full, pos3_full, cache,
                                             pos_full, 0, L + 1))

            def _score():
                h4 = hidden[0, s:e, :].to(torch.float32)
                if injected_scores is not None:
                    return injected_scores.to(dev).float(), h4
                return self.score_online(h4), h4
            scores, h4 = _step("scorer_ms", _score)
            h4_held, scores_held = h4, scores

            kept, extra = _step("selector_ms",
                                lambda: self.select(scores, h4, cfg.budget))
            info["select_extra"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                    for k, v in (extra or {}).items()}
            info["budget"] = cfg.budget
            info["n_kept"] = int(kept.numel())

            def _compact():
                keep_full = torch.cat([torch.arange(0, s, device=dev),
                                       kept + s,
                                       torch.arange(e, S, device=dev)])
                identity = bool(keep_full.numel() == S
                                and torch.equal(keep_full, pos_full))
                if identity and not cfg.force_split:
                    # Nothing is dropped, so there is nothing to compact: carry
                    # on with the same hidden states, cache, positions and mask.
                    return hidden, keep_full, pos_full, attn_full, True
                h = hidden.index_select(1, keep_full)
                self._compact_cache(cache, L + 1, keep_full)
                p = (pos_full.index_select(0, keep_full)
                     if cfg.pos_policy == POLICY_PRESERVE
                     else torch.arange(keep_full.numel(), device=dev))
                return h, keep_full, p, None, False
            hidden, keep_full, pos1_out, attn_reuse, is_identity = \
                _step("token_compaction_ms", _compact)
            info["identity_fast_path"] = bool(is_identity)
            info["split_path"] = bool(not is_identity)

            info["context_len"] = int(hidden.shape[1])
            info["pos_ids_max"] = int(pos1_out.max())
            info["pos_ids_contiguous"] = bool(
                torch.equal(pos1_out, torch.arange(pos1_out.numel(), device=dev)))
            info["select_idx"] = kept.detach().cpu().tolist()
            info["keep_full"] = keep_full.detach().cpu().tolist()
            info["scores"] = scores.detach().float().cpu().numpy()
            hidden = _step("llm_forward_ms",
                           lambda: _layers(hidden, pos1_out, cache, L + 1,
                                           self.n_layers, attn=attn_reuse))

        # ---- B0: no pruning ---------------------------------------------------
        else:
            pos1_out = torch.arange(S, device=dev)
            hidden = _step("llm_forward_ms",
                           lambda: _layers(prompt, pos1_out, cache, 0,
                                           self.n_layers))
            info["n_kept"] = N_VIS
            info["budget"] = S
            info["context_len"] = int(S)

        hidden = self.text.norm(hidden)
        logits = self.model.lm_head(hidden[:, -1:, :])[:, -1, :]
        _finish()

        for k in zero:
            if timings is not None and k not in timings:
                timings[k] = 0.0

        info["kv_bytes_early"] = (self._cache_bytes(cache, L + 1)
                                  if cfg.mode == MODE_GDEP else 0)
        info["kv_bytes_late"] = self._cache_bytes(cache, self.n_layers)
        info["kv_seq_len"] = int(cache.get_seq_length(0)) if len(cache.layers) else 0
        info["layer_calls_prefill"] = int(self.counts.get("layer_calls", 0))

        self.last = info
        state = dict(cache=cache, logits=logits, pos1=pos1_out,
                     n_ctx=torch.ones(1, hidden.shape[1], dtype=torch.long,
                                      device=dev),
                     hidden_len=int(hidden.shape[1]),
                     h4=h4_held, scores=scores_held)
        return state, info

    def _instruction_embeds(self, prep):
        """Text-side embeddings the incumbent's pruner consumes (its own API)."""
        ids = prep["inputs"]["input_ids"]
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids)
        s, e = prep["vis_slice"]
        seq = torch.cat([emb[:, :s], emb[:, e:]], dim=1)[0]
        return seq.mean(0, keepdim=True).expand(1, -1), seq.unsqueeze(0)

    # ---------------------------------------------------------------- decode --
    @torch.no_grad()
    def decode(self, state, max_new_tokens: int, ignore_eos: bool = False):
        """Greedy decode from a prefill state; returns the token id list.

        ``ignore_eos`` keeps stepping after the stop token so a benchmark can
        time a *fixed* number of steps on every arm (prereg §6.3: the decode
        benchmark generates exactly 32 tokens for every arm). It is never set
        for accuracy runs, which must stop where the model stops.
        """
        cfg, dev = self.cfg, self.dev
        cache, n_ctx, pos1 = state["cache"], state["n_ctx"], state["pos1"]
        logits = state["logits"]
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        embed = self.model.get_input_embeddings()
        preserve = (cfg.mode == MODE_GDEP and cfg.pos_policy == POLICY_PRESERVE)

        out = []
        cur = int(torch.argmax(logits, dim=-1).item())
        for _ in range(max_new_tokens):
            out.append(cur)
            if cur in eos and not ignore_eos:
                break
            cur_len = int(cache.get_seq_length(0))
            n_ctx = torch.cat([n_ctx, n_ctx.new_ones(1, 1)], dim=1)
            cp = torch.tensor([cur_len], device=dev)
            nxt = (int(pos1.max().item()) + 1) if preserve else cur_len
            pos3 = torch.tensor([[[nxt]]], device=dev).expand(3, 1, 1)
            h = embed(torch.tensor([[cur]], device=dev))
            attn = self._causal(h, n_ctx, cp, cache, pos3[0])
            h = self._run(h, attn, pos3, cache, cp, 0, self.n_layers)
            self.counts["decode_steps"] = self.counts.get("decode_steps", 0) + 1
            h = self.text.norm(h)
            logits = self.model.lm_head(h[:, -1:, :])[:, -1, :]
            cur = int(torch.argmax(logits, dim=-1).item())
        return out

    # ------------------------------------------------------------- one stop ---
    @torch.no_grad()
    def run(self, message, dataset_name, max_new_tokens: int, timings=None,
            injected_scores=None):
        prep = self.prepare(message, dataset_name, timings)
        state, info = self.prefill(prep, timings, injected_scores)
        ids = self.decode(state, max_new_tokens)
        self.counts["tokens_emitted"] = len(ids)
        text = self.vlm.processor.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return dict(prediction=self.vlm._post_process_response(text),
                    n_decode=len(ids), info=info)


def unwrap_visual(output):
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output
    return unwrap_visual_output(output)


# ---------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def dump_json(name: str, obj) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1)
    print(f"[saved] {name}")
    return path
