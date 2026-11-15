"""Frozen FireRed-OCR with the released LayoutLite score head.

The head class and OCR prompt are read from the pinned official checkout.
Transformers 4.57.6 supplies the complete, unmodified vision/text decoders.
Read-only vision hooks apply the final merger at the three DeepStack stages,
exactly as the official LayoutLite head feature extraction does. Native V and
all three native DeepStack streams are retained separately.

This diagnostic uses deterministic TopK plus four standardized-logit Gumbel
TopK masks at a fixed cardinality. These are NOT the official Bernoulli training
policy or the official two-cluster inference threshold.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn


QWEN = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = QWEN / "third_party/LayoutLite"
OFFICIAL_REVISION = "7e398a2ff4f23f0cb836e36e0e81d21d379d8fc2"
MODEL_REVISION = "7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50"
DEFAULT_MODEL = "/media/disk2/YZX/research/model/FireRed-OCR"
DEFAULT_HEAD = OFFICIAL_DIR / "checkpoints/firered_600.pt"
HEAD_SHA256 = "92e25d31ffae823b04b2b97be2259446b656afb81eb67f959fad16ac7d07e73d"


def official_components(directory=OFFICIAL_DIR):
    """Load only the score-head definition and prompt, avoiding 5.6 API imports."""
    directory = Path(directory)
    source = directory / "layoutlite/modeling_layoutlite.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "MultiLayerVisionTokenScoreMLP"]
    if len(nodes) != 1:
        raise RuntimeError("official LayoutLite score-head class missing")
    namespace = {"torch": torch, "nn": nn, "__name__": __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    utility = ast.parse((directory / "layoutlite/utils/utils.py").read_text())
    prompts = [ast.literal_eval(node.value) for node in utility.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "PROMPT_FIRERED"
                       for target in node.targets)]
    if len(prompts) != 1:
        raise RuntimeError("official FireRed OCR prompt missing")
    return namespace["MultiLayerVisionTokenScoreMLP"], prompts[0]


def diagnostic_masks(scores, keep_ratio=.5, seeds=(2026110801, 2026110802, 2026110803, 2026110804)):
    """Return one deterministic TopK and independent exact-K diagnostic samples.

    The released head emits raw logits. Exploration uses per-image z-scored
    logits (sample standard deviation), temperature one, and CPU-seeded Gumbel
    noise. Original raster ordering is restored after every selection.
    """
    scores = scores.detach().float().flatten().cpu()
    n = scores.numel()
    if n < 1 or not 0 < keep_ratio <= 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("nonempty finite logits and 0 < keep_ratio <= 1 required")
    k = max(1, min(n, round(keep_ratio * n)))
    std = scores.std() if n > 1 else scores.new_tensor(0.)
    standardized = (scores - scores.mean()) / std.clamp_min(1e-6)
    def indices(values):
        return torch.argsort(values, descending=True, stable=True)[:k].sort().values
    common = dict(N=n, K=k, keep_ratio=keep_ratio, temperature=1.,
                  normalization="per-image z-score, sample std, floor 1e-6",
                  std_epsilon=1e-6, degenerate_score_std=bool(std < 1e-6),
                  official_policy=False, raw_logit_mean=float(scores.mean()), raw_logit_std=float(std))
    output = [dict(name="det_topk", keep_idx=indices(scores), seed=None, K=k,
                   diag=dict(**common, operator="deterministic raw-score TopK"))]
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        uniform = torch.rand(n, generator=generator, dtype=torch.float32)
        uniform = uniform.clamp(torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps)
        gumbel = -torch.log(-torch.log(uniform))
        output.append(dict(name=f"sample_{index}", keep_idx=indices(standardized + gumbel),
                           seed=int(seed), K=k,
                           diag=dict(**common, operator="standardized-logit Gumbel TopK")))
    return output


def official_threshold_mask(scores, alpha):
    """Official five-iteration 1D two-cluster threshold, including its alpha sign.

    Alpha is a dataset-calibrated parameter in the original evaluation. This
    function does not imply exact K or replace the fixed-budget primary masks.
    """
    original = scores.detach().flatten().to(torch.bfloat16).cpu()
    values = original.float()
    c0, c1 = values.min(), values.max()
    if c0 == c1:
        return torch.zeros_like(original, dtype=torch.bool)
    for _ in range(5):
        high = (values - c0).abs() > (values - c1).abs()
        if bool((~high).any()):
            c0 = values[~high].mean()
        if bool(high.any()):
            c1 = values[high].mean()
    return original > c0 - float(alpha) * (c1 - c0)


class LayoutLiteRunner:
    """One frozen model load, CPU cache records, native text forward/generation."""

    def __init__(self, model_dir=DEFAULT_MODEL, head_path=DEFAULT_HEAD,
                 device="cuda", dtype=torch.bfloat16, official_dir=OFFICIAL_DIR,
                 model=None, processor=None, head=None, prompt=None):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_dir = str(model_dir)
        self.head_path = str(head_path)
        self.official_dir = Path(official_dir)
        head_class, official_prompt = official_components(self.official_dir)
        self.ocr_prompt = official_prompt if prompt is None else prompt
        if model is None:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir, dtype=dtype, attn_implementation="sdpa", local_files_only=True)
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = model.to(self.device).eval()
        self.processor = processor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.core = self.model.model
        self.visual = self.core.visual
        self.text = self.core.language_model
        self.deepstack_indexes = list(self.visual.deepstack_visual_indexes)
        if len(self.deepstack_indexes) != 3:
            raise ValueError("this diagnostic requires exactly three native DeepStack streams")
        if head is None:
            raw = Path(head_path).read_bytes()
            self.head_sha256 = hashlib.sha256(raw).hexdigest()
            if self.head_sha256 != HEAD_SHA256:
                raise ValueError("released firered_600.pt checksum mismatch")
            state = torch.load(head_path, map_location="cpu", weights_only=True)
            head = head_class(hidden_dim=state["norm.weight"].numel(),
                              mid_dim=state["fc1.weight"].shape[0])
            head.load_state_dict(state, strict=True)
        else:
            self.head_sha256 = "injected test head"
        self.head = head.to(device=self.device, dtype=dtype).eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        head_width = self.head.norm.weight.numel()
        if self.model.config.text_config.hidden_size != head_width:
            raise ValueError("released head width does not match this backbone")
        self._provenance = dict(model_dir=self.model_dir, model_repo="FireRedTeam/FireRed-OCR",
                                model_revision=MODEL_REVISION, head_path=self.head_path,
                                head_sha256=self.head_sha256, official_revision=OFFICIAL_REVISION,
                                execution="stock transformers SDPA; native three-DeepStack; original mRoPE coordinates",
                                sampler="diagnostic detTopK + zscore-logit Gumbel TopK; not official mask policy")

    def _prepare(self, image_path, prompt):
        messages = [dict(role="user", content=[dict(type="image", image=str(image_path)),
                                               dict(type="text", text=prompt)])]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {key: value.detach().cpu() for key, value in inputs.items()
                  if isinstance(value, torch.Tensor)}
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        if not required <= set(inputs) or inputs["image_grid_thw"].shape[0] != 1:
            raise ValueError("single-image processor input required")
        return inputs

    def _positions(self, inputs):
        ids = inputs["input_ids"].to(self.device)
        grid = inputs["image_grid_thw"].to(self.device)
        attention = inputs["attention_mask"].to(self.device)
        positions, _ = self.core.get_rope_index(ids, image_grid_thw=grid, attention_mask=attention)
        return positions.detach().cpu()

    @torch.inference_mode()
    def encode(self, image_path, prompt=None):
        """Encode once at the official dynamic resolution; return a CPU record."""
        prompt = self.ocr_prompt if prompt is None else prompt
        inputs = self._prepare(image_path, prompt)
        features = {}
        handles = []
        for index in self.deepstack_indexes:
            def capture(_module, _args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                features[index] = self.visual.merger(hidden).detach()
            handles.append(self.visual.blocks[index].register_forward_hook(capture))
        try:
            visual, ds = self.core.get_image_features(
                inputs["pixel_values"].to(self.device, dtype=self.dtype),
                image_grid_thw=inputs["image_grid_thw"].to(self.device))
        finally:
            for handle in handles:
                handle.remove()
        visual = torch.cat(visual, dim=0)
        if len(ds) != 3 or set(features) != set(self.deepstack_indexes):
            raise ValueError("missing native DeepStack/head features")
        levels = torch.stack([features[index] for index in self.deepstack_indexes] + [visual])
        scores = self.head(levels).squeeze(-1)
        n = visual.shape[0]
        if any(value.shape != visual.shape for value in ds) or scores.numel() != n:
            raise ValueError("head/native feature alignment failed")
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        if int(image_mask.sum()) != n:
            raise ValueError("native visual placeholder count differs from encoder")
        return dict(image_path=str(image_path), prompt=prompt, native_inputs=inputs,
                    ids=inputs["input_ids"], pos3d=self._positions(inputs), n_vis=n,
                    image_grid_thw=inputs["image_grid_thw"], V=visual.cpu(),
                    DS=[value.cpu() for value in ds], head_features=levels.cpu(), scores=scores.cpu(),
                    feature_layers=[index + 1 for index in self.deepstack_indexes] + [len(self.visual.blocks)],
                    provenance=dict(self._provenance))

    @torch.inference_mode()
    def re_prompt(self, page, prompt):
        """Reuse the identical visual encoding for an OCR or real QA prompt."""
        inputs = self._prepare(page["image_path"], prompt)
        if not torch.equal(inputs["image_grid_thw"], page["image_grid_thw"]):
            raise ValueError("new prompt changed image preprocessing geometry")
        output = dict(page)
        output.update(prompt=prompt, native_inputs=inputs, ids=inputs["input_ids"],
                      pos3d=self._positions(inputs))
        return output

    def _compact(self, page, keep_idx=None):
        n = page["n_vis"]
        if keep_idx is None:
            keep = torch.arange(n, dtype=torch.long)
        else:
            keep = torch.as_tensor(keep_idx).detach().cpu()
            if keep.dtype == torch.bool:
                if keep.numel() != n:
                    raise ValueError("boolean keep mask has wrong length")
                keep = keep.nonzero(as_tuple=True)[0]
            keep = keep.long().flatten().sort().values
        if keep.numel() < 1 or keep.unique().numel() != keep.numel() or keep.min() < 0 or keep.max() >= n:
            raise ValueError("keep indices must be nonempty, unique and in range")
        ids = page["ids"].to(self.device)
        positions = page["pos3d"].to(self.device)
        image = ids[0] == self.model.config.image_token_id
        image_positions = image.nonzero(as_tuple=True)[0]
        full_keep = ~image
        full_keep[image_positions[keep.to(self.device)]] = True
        compact_ids = ids[:, full_keep]
        compact_positions = positions[:, :, full_keep]
        visual_mask = compact_ids == self.model.config.image_token_id
        selected = page["V"][keep].to(device=self.device, dtype=self.dtype)
        ds = [value[keep].to(device=self.device, dtype=self.dtype) for value in page["DS"]]
        embeddings = self.model.get_input_embeddings()(compact_ids).clone()
        embeddings[visual_mask] = selected
        if not torch.equal(compact_positions[:, :, ~visual_mask[0]], positions[:, :, ~image]):
            raise ValueError("text mRoPE coordinates changed during compaction")
        audit = dict(n_vis=keep.numel(), input_n_vis=n, ds_lengths=[value.shape[0] for value in ds],
                     keep_idx=keep.tolist(), prompt_tokens=compact_ids.shape[1],
                     full_prompt_tokens=ids.shape[1], expected_decode_start=int(positions.max()) + 1,
                     text_positions_unchanged=True)
        return embeddings, compact_positions, visual_mask, ds, audit

    @torch.inference_mode()
    def prefill(self, page, keep_idx=None):
        """Return first-answer logits [1,vocab], native KV cache and audit."""
        embeddings, positions, visual_mask, ds, audit = self._compact(page, keep_idx)
        output = self.text(inputs_embeds=embeddings, position_ids=positions,
                           attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                           visual_pos_masks=visual_mask, deepstack_visual_embeds=ds,
                           use_cache=True, return_dict=True)
        logits = self.model.lm_head(output.last_hidden_state[:, -1:])[:, 0]
        return logits, output.past_key_values, audit

    @torch.inference_mode()
    def _decode(self, token, cache, position, cache_length):
        ids = torch.tensor([[int(token)]], dtype=torch.long, device=self.device)
        positions = torch.full((3, 1, 1), int(position), dtype=torch.long, device=self.device)
        output = self.text(input_ids=ids, position_ids=positions, past_key_values=cache,
                           attention_mask=torch.ones((1, cache_length + 1), dtype=torch.long, device=self.device),
                           use_cache=True, return_dict=True)
        return self.model.lm_head(output.last_hidden_state[:, -1:])[:, 0], output.past_key_values

    @torch.inference_mode()
    def generate(self, page, keep_idx=None, max_new_tokens=2048):
        """Greedy native decoding, preserving the full-prompt continuation index."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        started = time.monotonic()
        logits, cache, audit = self.prefill(page, keep_idx)
        eos = self.model.generation_config.eos_token_id
        eos = {int(eos)} if isinstance(eos, int) else set(eos or [])
        tokens, decode_positions = [], []
        stop_reason = "max_new_tokens"
        for index in range(max_new_tokens):
            token = int(logits.argmax(-1).item())
            tokens.append(token)
            if token in eos:
                stop_reason = "eos"
                break
            if index + 1 == max_new_tokens:
                break
            position = audit["expected_decode_start"] + index
            logits, cache = self._decode(token, cache, position, audit["prompt_tokens"] + index)
            decode_positions.append(position)
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        audit.update(token_ids=tokens, generated_tokens=len(tokens),
                     truncated=stop_reason == "max_new_tokens", stop_reason=stop_reason,
                     decode_positions=decode_positions, elapsed_seconds=time.monotonic() - started)
        return text, audit

    @torch.inference_mode()
    def identity_gate(self, page):
        """Compare cached full-token prefill and one decode against stock model."""
        inputs = {key: value.to(self.device) for key, value in page["native_inputs"].items()}
        observed = {}
        def capture(_module, _args, output):
            observed["V"], observed["DS"] = output
        handle = self.visual.register_forward_hook(capture)
        try:
            native = self.model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
        finally:
            handle.remove()
        logits, cache, audit = self.prefill(page)
        expected = native.logits[:, -1]
        token = int(expected.argmax(-1).item())
        native_step = self.model(
            input_ids=torch.tensor([[token]], device=self.device),
            attention_mask=torch.ones((1, page["ids"].shape[1] + 1), dtype=torch.long, device=self.device),
            past_key_values=native.past_key_values,
            cache_position=torch.tensor([page["ids"].shape[1]], device=self.device),
            use_cache=True, return_dict=True, logits_to_keep=1)
        ours_step, _ = self._decode(token, cache, audit["expected_decode_start"], audit["prompt_tokens"])
        result = dict(native_V_bitwise=torch.equal(observed["V"].cpu(), page["V"]),
                      native_DS_bitwise=[torch.equal(v.cpu(), cached) for v, cached in zip(observed["DS"], page["DS"])],
                      full_prefill_logits_bitwise=torch.equal(logits, expected),
                      full_prefill_max_delta=float((logits.float() - expected.float()).abs().max()),
                      first_argmax_equal=int(logits.argmax(-1)) == token,
                      decode_logits_bitwise=torch.equal(ours_step, native_step.logits[:, -1]),
                      decode_max_delta=float((ours_step.float() - native_step.logits[:, -1].float()).abs().max()),
                      n_vis=page["n_vis"], ds_lengths=[v.shape[0] for v in page["DS"]],
                      expected_decode_start=audit["expected_decode_start"],
                      text_positions_unchanged=audit["text_positions_unchanged"])
        result["pass_all"] = all((result["native_V_bitwise"], all(result["native_DS_bitwise"]),
                                  result["full_prefill_logits_bitwise"], result["decode_logits_bitwise"],
                                  result["text_positions_unchanged"]))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--head", default=str(DEFAULT_HEAD))
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--identity-gate", action="store_true")
    args = parser.parse_args()
    runner = LayoutLiteRunner(args.model, args.head)
    page = runner.encode(args.image, prompt=args.prompt)
    if args.identity_gate:
        print(json.dumps(runner.identity_gate(page), indent=2))
    text, audit = runner.generate(page, max_new_tokens=args.max_new_tokens)
    print(json.dumps(dict(prediction=text, audit=audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
