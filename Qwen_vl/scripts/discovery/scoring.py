"""
Per-sample and aggregate scoring that reuses VLMEvalKit's own metric code.

Two layers:
  * ``official_scores``  -- dumps a prediction xlsx and calls ``dataset.evaluate``
    (the exact code path that produced the published baseline numbers);
  * ``per_sample_hits``  -- per-instance scores via the same ``process_line`` /
    ``hit_calculate`` helpers, needed for paired comparisons and failure analysis.

``validate_scoring.py`` checks layer 2 reproduces layer 1 on the full splits.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

METHOD_BY_DATASET = {
    "TextVQA_VAL": "vqa_score",
    "DocVQA_VAL": "anls",
}


def _method(dataset_name: str) -> str:
    for key, meth in METHOD_BY_DATASET.items():
        if key in dataset_name:
            return meth
    return "vqa_score"


def ocrbench_hit(answer, prediction) -> int:
    """Exact replication of OCRBench.evaluate's per-answer containment rule."""
    import ast

    answers = answer if isinstance(answer, list) else ast.literal_eval(str(answer))
    pred = str(prediction)
    for a in answers:
        a_cmp = str(a).lower().strip().replace("\n", " ")
        p_cmp = pred.lower().strip().replace("\n", " ")
        if a_cmp in p_cmp:
            return 1
    return 0


def ocrbench_hit_hmer(answer, prediction) -> int:
    import ast

    answers = answer if isinstance(answer, list) else ast.literal_eval(str(answer))
    pred = str(prediction)
    for a in answers:
        a_cmp = str(a).strip().replace("\n", " ").replace(" ", "")
        p_cmp = pred.strip().replace("\n", " ").replace(" ", "")
        if a_cmp in p_cmp:
            return 1
    return 0


def per_sample_hits(dataset_name: str, rows, predictions) -> np.ndarray:
    """Per-instance score in [0, 1] using the official metric definitions."""
    if "OCRBench" in dataset_name:
        out = []
        for r, p in zip(rows, predictions):
            if r["category"] == "Handwritten Mathematical Expression Recognition":
                out.append(ocrbench_hit_hmer(r["answer"], p))
            else:
                out.append(ocrbench_hit(r["answer"], p))
        return np.asarray(out, dtype=float)

    method = _method(dataset_name)
    frame = pd.DataFrame(
        [
            {"answer": str(r["answer"]), "prediction": str(p), "question": str(r.get("question", ""))}
            for r, p in zip(rows, predictions)
        ]
    )
    lines = [frame.iloc[i] for i in range(len(frame))]
    res = [process_line(line, method=method) for line in lines]
    return np.asarray(hit_calculate(res, dataset_name), dtype=float)


def official_scores(dataset_name: str, rows, predictions, dataset=None, tag: str = "tmp") -> dict:
    """Aggregate score via VLMEvalKit's own ``dataset.evaluate``."""
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump

    if dataset is None:
        dataset = build_dataset(dataset_name)

    frame = pd.DataFrame([dict(r) for r in rows])
    frame["prediction"] = list(predictions)
    out_dir = os.path.join(tempfile.gettempdir(), "eadp_discovery_scores")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_{dataset_name}.xlsx")
    dump(frame, path)
    return dataset.evaluate(path)


def summarize(dataset_name: str, rows, predictions, dataset=None, tag: str = "tmp") -> dict:
    """
    ``acc_pct`` is the one metric comparable across the three benchmarks:
      * TextVQA -> mean VQA score x100  (== the official "Overall")
      * DocVQA  -> mean ANLS x100      (== the official "Overall")
      * OCRBench-> % questions correct (x10 == the official "Final Score" on 1000 items)
    """
    hits = per_sample_hits(dataset_name, rows, predictions)
    n = int(len(hits))
    acc = float(np.mean(hits) * 100) if n else float("nan")
    return {"n": n, "acc_pct": acc, "sum_hits": float(np.sum(hits))}, hits
