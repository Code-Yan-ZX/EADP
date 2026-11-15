"""Dataset-aware adapter; the original experiment and weights stay immutable."""
from __future__ import annotations

import torch

from scripts.paired_pruning import runtime as native
from scripts.paired_pruning.evaluate import choose_keep as student_keep

PROMPT_SUFFIX = native.PROMPT_SUFFIX
load_engine = native.load_engine
generate = native.generate
low_resolution = native.low_resolution
sync = native.sync


def public_item(item):
    """Only these input fields may cross the inference boundary."""
    return {key: item[key] for key in
            ("dataset", "item_id", "image_path", "image_sha256", "question")}


def message_for(item):
    return [{"type": "image", "value": item["image_path"]},
            {"type": "text", "value": item["question"] + PROMPT_SUFFIX}]


@torch.no_grad()
def prepare(engine, item, max_question_tokens=256):
    if set(item) != {"dataset", "item_id", "image_path", "image_sha256", "question"}:
        raise ValueError("Inference accepts only public image/question fields")
    if native.sha256(item["image_path"]) != item["image_sha256"]:
        raise ValueError("Frozen image changed")
    message = message_for(item)
    prep = engine.prepare(message, item["dataset"])
    visual, deepstack = engine.encode(prep)
    if visual.ndim != 2 or visual.shape[0] != prep["n_vis"] or len(deepstack) != 3:
        raise ValueError("Invalid native visual streams")
    if any(stream.shape != visual.shape for stream in deepstack):
        raise ValueError("DeepStack is not aligned to the visual stream")
    prompt = message[-1]["value"]
    ids = engine.tok(prompt, return_tensors="pt", add_special_tokens=True,
                     truncation=False)["input_ids"]
    if not 0 < ids.shape[1] <= max_question_tokens:
        raise ValueError(f"Question exceeds {max_question_tokens}; silent truncation is forbidden")
    question = engine.model.get_input_embeddings()(ids.to(visual.device))[0].detach()
    return dict(message=message, prep=prep, V=visual, DS=deepstack,
                question_embeddings=question.to(torch.bfloat16),
                question_token_ids=ids[0].cpu(), dataset=item["dataset"],
                source_image_sha256=item["image_sha256"])


@torch.no_grad()
def choose_keep(engine, context, arm, models, k=256):
    if arm in ("full", "rres512"):
        return None
    if arm in ("b1", "b2"):
        mean, sequence = engine.instruction_embeds(context["message"], context["dataset"])
        return engine.select(arm, k, dict(context, K=k, engine=engine,
                                          text_mean=mean, text_seq=sequence)).to(context["V"].device)
    if arm not in models:
        raise ValueError(f"Unknown frozen arm {arm}")
    # Reuse precisely the feature dtype and four-stream inference convention
    # used for the selected checkpoints, without changing the old source.
    return student_keep(engine, context, {"arm": "bag_rank", "name": arm}, models, k)
