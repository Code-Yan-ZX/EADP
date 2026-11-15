"""Native, answer-free generation; oracle-dependent controls are labeled."""
from __future__ import annotations

from contextlib import nullcontext

import torch

from scripts.student_transfer import runtime as transfer
from scripts.student_transfer.metrics import score_item as original_score

load_engine = transfer.load_engine
public_item = transfer.public_item
message_for = transfer.message_for
prepare = transfer.prepare
sync = transfer.sync
low_resolution = transfer.low_resolution


def score_item(dataset, prediction, answers):
    aliases = {"InfoVQA_TRAIN": "DocVQA_VAL", "TextVQA_TRAIN": "TextVQA_VAL"}
    if dataset not in aliases:
        raise ValueError(f"Unsupported diagnostic dataset: {dataset}")
    return original_score(aliases[dataset], prediction, answers)


def validate_native(result, arm, source_count, cap, answer_dependent=False):
    expected_source = 256 if arm == "rres512" else 1024
    expected = 1024 if arm == "full" else 256
    keep = result["keep_idx"]
    if source_count != expected_source:
        raise ValueError("Unexpected source visual budget")
    if (len(keep) != expected or keep != sorted(set(keep))
            or any(type(index) is not int or not 0 <= index < source_count for index in keep)):
        raise ValueError("Invalid retained native indices")
    if result["actual_visual_count"] != expected or result["ds_lengths"] != [expected] * 3:
        raise ValueError("Visual and DeepStack budgets do not match")
    audit = result["audit"]
    if not (audit.get("native") and audit.get("positions_preserved")
            and audit.get("ds_streams") == 3 and audit.get("fresh_kv") is True
            and audit.get("answer_in_selection") is bool(answer_dependent)):
        raise ValueError("Native or answer-dependence audit failed")
    if len(result["token_ids"]) > cap or type(result["truncated"]) is not bool:
        raise ValueError("Generation trace/cap invalid")


@torch.no_grad()
def choose_keep(engine, context, arm, models, geometry=None):
    if arm in ("full", "rres512"):
        if geometry is not None:
            raise ValueError("A resolution baseline cannot take an extra selection")
        return None
    if geometry is not None:
        keep = geometry["keep_idx"]
        if len(keep) != 256 or keep != sorted(set(keep)):
            raise ValueError("Geometry must have exactly 256 distinct sorted indices")
        return torch.tensor(keep, dtype=torch.long, device=context["V"].device)
    return transfer.choose_keep(engine, context, arm, models)


def inference(engine, item, arm, models, *, geometry=None, cap=128,
              ignore_eos=False, context=None, answer_dependent=False):
    if arm == "rres512" and context is not None:
        raise ValueError("R-res must prepare its own genuinely low-resolution features")
    manager = low_resolution(engine, 512) if arm == "rres512" else nullcontext()
    with manager:
        used = context if context is not None else prepare(engine, public_item(item))
        keep = choose_keep(engine, used, arm, models, geometry)
        result = transfer.generate(engine, used, keep, cap, ignore_eos)
        # The frozen legacy generator assumes a deployable answer-free selector.
        # Only the conditional factorial controls are chosen after viewing gold
        # QA scores; preserve this distinction without editing that old helper.
        result["audit"] = dict(result["audit"], answer_in_selection=bool(answer_dependent),
                               generation_answer_free=True)
        result.update(selector_answer_dependent=bool(answer_dependent),
                      selection_scope="offline_oracle_control" if answer_dependent else "fixed_input_rule",
                      source_visual_count=int(used["prep"]["n_vis"]))
        validate_native(result, arm, result["source_visual_count"], cap, answer_dependent)
        if geometry is not None and result["keep_idx"] != geometry["keep_idx"]:
            raise ValueError("Generation selection differs from the declared geometry")
        return result, used
