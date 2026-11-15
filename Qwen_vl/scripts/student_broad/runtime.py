"""Native inference with public, task-specific prompts and frozen students."""
from __future__ import annotations

import string
import time

import torch

from scripts.paired_pruning import runtime as native
from scripts.paired_pruning.evaluate import choose_keep as student_keep

PROMPT_SUFFIX = native.PROMPT_SUFFIX
load_engine = native.load_engine
low_resolution = native.low_resolution
sync = native.sync
MAX_QUESTION_TOKENS = 4096
MCQ = frozenset(("AI2D_TEST", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11", "MMStar", "RealWorldQA"))
YORN = frozenset(("HallusionBench", "MME", "POPE"))
VQA = frozenset(("ChartQA_TEST", "InfoVQA_VAL"))
PUBLIC_FIELDS = {"dataset", "item_id", "image_path", "image_sha256", "question", "options", "hint", "has_image"}


def prompt_text(item):
    dataset, question = item["dataset"], item["question"]
    if dataset in MCQ:
        prompt = f"Hint: {item['hint']}\n" if item.get("hint") else ""
        prompt += f"Question: {question}\n"
        options = item.get("options", {})
        if options:
            prompt += "Options:\n"
            for key in string.ascii_uppercase:
                if key in options:
                    prompt += f"{key}. {options[key]}\n"
            prompt += "Answer with the option letter only."
        return prompt.rstrip()
    if dataset in YORN:
        return question + " Please answer yes or no."
    if dataset in VQA:
        return question + PROMPT_SUFFIX
    raise ValueError(f"Unsupported task prompt {dataset}")


def public_item(item):
    return dict(dataset=item["dataset"], item_id=str(item["item_id"]),
                image_path=item["image_path"], image_sha256=item["image_sha256"],
                question=item["question"], options=dict(item.get("options", {})),
                hint=item.get("hint"), has_image=item.get("has_image", True))


def message_for(item):
    message = [{"type": "image", "value": item["image_path"]}] if item.get("has_image", True) else []
    return message + [{"type": "text", "value": prompt_text(item)}]


@torch.no_grad()
def prepare(engine, item, max_question_tokens=MAX_QUESTION_TOKENS):
    if set(item) != PUBLIC_FIELDS:
        raise ValueError("Inference accepts only public image/question/hint/options fields")
    if item["has_image"] and native.sha256(item["image_path"]) != item["image_sha256"]:
        raise ValueError("Frozen image changed")
    message = message_for(item)
    prep = engine.prepare(message, item["dataset"])
    if item["has_image"]:
        visual, deepstack = engine.encode(prep)
        if visual.ndim != 2 or visual.shape[0] != prep["n_vis"] or len(deepstack) != 3:
            raise ValueError("Invalid native visual streams")
        if any(stream.shape != visual.shape for stream in deepstack):
            raise ValueError("DeepStack is not aligned to the visual stream")
    else:
        if prep["has_image"] or prep["n_vis"]:
            raise ValueError("Text-only control unexpectedly received image tokens")
        parameter = next(engine.model.parameters())
        visual = torch.empty((0, engine.model.config.text_config.hidden_size),
                             device=parameter.device, dtype=parameter.dtype)
        deepstack = []
    prompt = message[-1]["value"]
    ids = engine.tok(prompt, return_tensors="pt", add_special_tokens=True,
                     truncation=False)["input_ids"]
    if not 0 < ids.shape[1] <= max_question_tokens:
        raise ValueError(f"Question exceeds {max_question_tokens}; silent truncation is forbidden")
    question = engine.model.get_input_embeddings()(ids.to(visual.device))[0].detach()
    return dict(message=message, prep=prep, V=visual, DS=deepstack,
                question_embeddings=question.to(torch.bfloat16),
                question_token_ids=ids[0].cpu(), dataset=item["dataset"],
                source_image_sha256=item["image_sha256"], has_image=item["has_image"])


def adapt_student(model):
    if model.config["ordered"] or model.config["mode_heads"] != 1:
        raise ValueError("Only the selected unordered, one-head bag checkpoints are eligible")
    # The bag scorer never reads its sinusoidal position buffer. Raising this
    # input-length guard changes no learned tensor and truncates no question.
    model.config["max_question_tokens"] = MAX_QUESTION_TOKENS
    return model.eval().requires_grad_(False)


@torch.no_grad()
def choose_keep(engine, context, arm, models, k=256):
    if context["prep"]["n_vis"] == 0 or arm in ("full", "rres512"):
        return None
    if arm in ("b1", "b2"):
        mean, sequence = engine.instruction_embeds(context["message"], context["dataset"])
        return engine.select(arm, k, dict(context, K=k, engine=engine,
                                          text_mean=mean, text_seq=sequence)).to(context["V"].device)
    if arm not in models:
        raise ValueError(f"Unknown frozen arm {arm}")
    return student_keep(engine, context, {"arm": "bag_rank", "name": arm}, models, k)


@torch.inference_mode()
def generate(engine, context, keep=None, max_tokens=128, ignore_eos=False):
    if context["prep"]["n_vis"]:
        return native.generate(engine, context, keep, max_tokens, ignore_eos)
    from scripts.multistream_oracle.oracle import tensor_sha256
    sync(context["V"].device)
    started = time.perf_counter()
    state = engine.prefill(context["prep"], None, None, None, deepstack=False)
    ids, text = engine.decode(state, max_tokens, ignore_eos=ignore_eos)
    sync(context["V"].device)
    ready = time.perf_counter()
    positions = engine.positions_full(context["prep"])[0]
    if not torch.equal(state.positions_prefill, positions):
        raise AssertionError("Native text-only positions changed")
    if state.cache.get_seq_length() != state.lk + len(state.decode_positions):
        raise AssertionError("Text-only cache bookkeeping failed")
    return dict(text=text, token_ids=ids, truncated=len(ids) >= max_tokens,
        deepstack=False, actual_visual_count=0, native_state_n_vis=0, ds_lengths=[], keep_idx=[],
        positions_sha256=tensor_sha256(state.positions_prefill), prefill_length=state.lk,
        cache_length=int(state.cache.get_seq_length()), decode_positions=state.decode_positions,
        gradients_enabled=False, generation_ready_perf_counter=ready,
        cached_generation_ms=(ready-started)*1000, post_generation_audit_ms=(time.perf_counter()-ready)*1000,
        audit=dict(native=True, ds_streams=0, positions_preserved=True, physical_visual_count=0,
                   answer_in_selection=False, fresh_kv=True, text_only_control=True))
