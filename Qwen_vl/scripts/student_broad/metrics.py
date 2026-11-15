"""Offline, source-pinned benchmark metrics; no model or paid judge imports.

The scalar per-row score is useful for error analysis. Native benchmark
aggregates (MMBench circular, MME points, POPE F1, Hallusion grouped scores)
must be read from ``aggregate_metrics`` rather than substituted by that mean.
"""
from __future__ import annotations

import ast
import copy
from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import string
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd


VLM_ROOT = Path(__file__).resolve().parents[2] / "VLMEvalKit" / "vlmeval"
METRIC_SOURCES = {
    "vqa_eval": VLM_ROOT / "dataset/utils/vqa_eval.py",
    "punctuation": VLM_ROOT / "smp/misc.py",
    "matching": VLM_ROOT / "utils/matching_util.py",
    "multiple_choice": VLM_ROOT / "dataset/utils/multiple_choice.py",
    "yorn": VLM_ROOT / "dataset/utils/yorn.py",
    "image_mcq": VLM_ROOT / "dataset/image_mcq.py",
    "image_yorn": VLM_ROOT / "dataset/image_yorn.py",
    "image_vqa": VLM_ROOT / "dataset/image_vqa.py",
}
MCQ_DATASETS = frozenset(("AI2D_TEST", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11", "MMStar", "RealWorldQA"))
YN_DATASETS = frozenset(("HallusionBench", "MME", "POPE"))


def source_metadata():
    return {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in METRIC_SOURCES.items()}


def _pure_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"official metric definitions changed in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@lru_cache(maxsize=1)
def _official_functions():
    # VERBOSE accidentally changes upstream A parsing. Freeze its default 0
    # rather than let an unrelated shell setting alter benchmark correctness.
    namespace = {"Optional": Optional, "string": string, "cp": copy,
                 "os": SimpleNamespace(environ={"VERBOSE": 0})}
    _pure_functions(METRIC_SOURCES["punctuation"], {"process_punctuation"}, namespace)
    _pure_functions(METRIC_SOURCES["vqa_eval"],
                    {"levenshtein_distance", "anls_compute", "relaxed_correctness"}, namespace)
    _pure_functions(METRIC_SOURCES["matching"], {"can_infer_option", "can_infer_text", "can_infer"}, namespace)
    _pure_functions(METRIC_SOURCES["yorn"], {"YOrN_Extraction"}, namespace)
    return namespace


def _metadata(metadata):
    metadata = metadata or {}
    return {**metadata.get("metadata", {}), **metadata}


def metric_name(dataset):
    if dataset in MCQ_DATASETS:
        return "circular exact-matching accuracy" if dataset.startswith("MMBench") else "exact-matching accuracy"
    if dataset == "InfoVQA_VAL":
        return "ANLS"
    if dataset == "ChartQA_TEST":
        return "relaxed accuracy"
    if dataset == "MME":
        return "MME perception + reasoning points"
    if dataset == "POPE":
        return "POPE F1"
    if dataset == "HallusionBench":
        return "HallusionBench aAcc (exact matching)"
    raise ValueError(f"unsupported dataset metric: {dataset}")


def parse_prediction(dataset, prediction, metadata=None):
    """Return an option/Yes/No, or None for unresolved and refused answers.

RealWorldQA is a multiple-choice dataset in the pinned local VLMEvalKit and
uses its ordinary exact-matching extractor, including option-text fallback.
No lowercasing option letters or custom Chinese-answer heuristics are added.
"""
    if not isinstance(prediction, str):
        raise ValueError("prediction must be a string")
    metric_name(dataset)
    if dataset in MCQ_DATASETS:
        meta = _metadata(metadata)
        choices = meta.get("options")
        if choices is None:
            choices = {key: meta[key] for key in string.ascii_uppercase if key in meta}
        if not isinstance(choices, dict) or len(choices) < 2:
            raise ValueError(f"{dataset} requires at least two option texts")
        if any(key not in string.ascii_uppercase or not isinstance(value, str) or not value
               for key, value in choices.items()):
            raise ValueError("malformed multiple-choice options")
        parsed = _official_functions()["can_infer"](prediction, copy.deepcopy(choices))
        return parsed if parsed in choices else None
    if dataset in YN_DATASETS:
        parsed = _official_functions()["YOrN_Extraction"](prediction)
        return parsed if parsed in ("Yes", "No") else None
    return prediction.strip()


def score_item(dataset, prediction, answers, category=None, metadata=None):
    if not isinstance(prediction, str):
        raise ValueError("prediction must be a string")
    if not isinstance(answers, (list, tuple)) or not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError("answers must be a nonempty string sequence")
    metric_name(dataset)
    if dataset in MCQ_DATASETS or dataset in YN_DATASETS:
        if len(answers) != 1:
            raise ValueError("categorical benchmarks require one ground-truth label")
        parsed = parse_prediction(dataset, prediction, metadata)
        if dataset in YN_DATASETS and answers[0] not in ("Yes", "No"):
            raise ValueError("yes/no ground truth must preserve official Yes/No labels")
        if dataset in MCQ_DATASETS and answers[0] not in string.ascii_uppercase:
            raise ValueError("MCQ ground truth must be an uppercase option letter")
        return float(parsed is not None and parsed == answers[0])
    if dataset == "InfoVQA_VAL":
        similarity = 1. - min(_official_functions()["anls_compute"](answer, prediction) for answer in answers)
        return float(similarity if similarity >= .5 else 0.)
    return float(any(_official_functions()["relaxed_correctness"](answer, prediction.strip()) for answer in answers))


def _rating(function_name, frame):
    namespace = {"np": np, "pd": pd, "defaultdict": defaultdict,
                 "load": lambda _: frame.copy(), "d2df": lambda values: pd.DataFrame([values])}
    _pure_functions(METRIC_SOURCES["yorn"], {function_name}, namespace)
    result = namespace[function_name]("in-memory")
    # Convert numpy scalars and NaNs without emitting nonstandard JSON NaN.
    return [{str(key): (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
             for key, value in row.items()} for row in result.to_dict(orient="records")]


def aggregate_metrics(dataset, items, records):
    """Return native metrics for one arm over the complete supplied manifest.

``items`` are manifest rows. ``records`` may be an item_id->record mapping or
    a sequence of records containing item_id and text/prediction. A saved error
    counts as wrong; an absent prediction keeps the run explicitly incomplete.
    Native percentages in ``official`` are kept in their original units.
    """
    metric_name(dataset)
    items = list(items)
    if not items:
        raise ValueError("cannot aggregate an empty benchmark")
    item_ids = [str(item["item_id"]) for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate manifest item IDs")
    if not isinstance(records, dict):
        records = list(records)
        ids = [str(row["item_id"]) for row in records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate prediction item IDs")
        records = dict(zip(ids, records))
    records = {str(key): row for key, row in records.items()}
    rows, unparsed, failed = [], 0, 0
    for item, item_id in zip(items, item_ids):
        if item_id not in records:
            continue
        record, meta = records[item_id], _metadata(item)
        prediction = record.get("text", record.get("prediction", ""))
        error = bool(record.get("error") or record.get("failed") or record.get("status") in ("error", "failed"))
        parsed = None if error else parse_prediction(dataset, prediction, meta)
        score = 0. if error else score_item(dataset, prediction, item["answers"], item.get("category"), meta)
        if "score" in record and abs(float(record["score"]) - score) > 1e-9:
            raise ValueError(f"saved metric disagrees with raw prediction: {dataset}/{item_id}")
        failed += error
        unparsed += parsed is None
        rows.append({**meta, "item_id": item_id, "answer": item["answers"][0],
                     "prediction": prediction, "extracted": parsed or "Unknown", "score": score})
    result = {"primary_metric": metric_name(dataset), "primary_score": None,
              "primary_unit": "points" if dataset == "MME" else "fraction",
              "n_expected": len(items), "n_observed": len(rows), "unparsed": unparsed,
              "failed": int(failed), "row_score": None, "official": [],
              "observed_row_score": float(np.mean([row["score"] for row in rows])) if rows else None}
    if len(rows) != len(items):
        return result
    result["row_score"] = result["observed_row_score"]
    result["primary_score"] = result["row_score"]
    frame = pd.DataFrame(rows)
    if dataset.startswith("MMBench"):
        groups = defaultdict(list)
        for row in rows:
            source_index = row.get("source_index", row.get("index"))
            if source_index is None:
                raise ValueError("MMBench requires original source_index")
            index = int(source_index)
            group = int(row.get("g_index", index % 1_000_000))
            groups[group].append((index, row))
        base_rows = []
        for group, members in groups.items():
            base = [row for index, row in members if index == group]
            if len(base) != 1:
                raise ValueError(f"MMBench circular group {group} has no unique base row")
            base_rows.append({**base[0], "hit": float(all(row["score"] for _, row in members))})
        result["primary_score"] = float(np.mean([row["hit"] for row in base_rows]))
        result["n_circular_groups"] = len(base_rows)
        result["circular_group_sizes"] = dict(pd.Series([len(group) for group in groups.values()]).value_counts().sort_index().items())
        result["official"] = [{"split": "Overall", "Overall": 100. * result["primary_score"]}]
        result["category_scores"] = {str(category): 100. * float(np.mean([row["hit"] for row in base_rows if row.get("category") == category]))
                                     for category in sorted({row["category"] for row in base_rows if row.get("category") is not None})}
    elif dataset == "MME":
        image_groups = []
        for row in rows:
            path = row.get("mme_image_group", row.get("original_image_path", row.get("image_path")))
            if path is None:
                raise ValueError("MME requires the original image grouping key")
            image_groups.append(str(path))
        frame["image_path"] = image_groups
        counts = frame.groupby(["category", "image_path"]).size()
        if not (counts == 2).all():
            raise ValueError("MME must contain exactly two questions per category/image group")
        result["official"] = _rating("MME_rating", frame)
        result["primary_score"] = float(result["official"][0]["perception"] + result["official"][0]["reasoning"])
    elif dataset == "HallusionBench":
        frame["index"] = [str(row.get("source_index", row.get("index", ""))) for row in rows]
        if any(len(index.split("_")) < 6 for index in frame["index"]):
            raise ValueError("HallusionBench requires original six-field source indices")
        result["official"] = _rating("Hallusion_rating", frame)
        overall = next(row for row in result["official"] if row["split"] == "Overall")
        result["primary_score"] = overall["aAcc"] / 100.
    elif dataset == "POPE":
        result["official"] = _rating("POPE_rating", frame)
        overall = next(row for row in result["official"] if row["split"] == "Overall")
        result["primary_score"] = overall["Overall"] / 100.
        result["note"] = "Official F1 treats Unknown as non-Yes; row accuracy still counts Unknown wrong. Comma-separated categories are exploded by the original evaluator."
    else:
        result["official"] = [{"split": "Overall", "Overall": 100. * result["row_score"]}]
    return result
