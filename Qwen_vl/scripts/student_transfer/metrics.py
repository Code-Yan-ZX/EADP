"""Dataset-specific scores matching the pinned, local VLMEvalKit evaluators.

Only pure metric functions are loaded by AST. Importing VLMEvalKit normally
loads model backends, dataset download helpers and optional GPU packages; none
of those is needed to score saved predictions. The original sources and their
hashes are exposed so the experiment freeze can cover this dependency.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path


VLM_ROOT = Path(__file__).resolve().parents[2] / "VLMEvalKit" / "vlmeval"
METRIC_SOURCES = {
    "vqa_eval": VLM_ROOT / "dataset/utils/vqa_eval.py",
    "punctuation": VLM_ROOT / "smp/misc.py",
    "ocrbench": VLM_ROOT / "dataset/utils/ocrbench.py",
}
OCR_CATEGORIES = (
    "Regular Text Recognition", "Irregular Text Recognition",
    "Artistic Text Recognition", "Handwriting Recognition",
    "Digit String Recognition", "Non-Semantic Text Recognition",
    "Scene Text-centric VQA", "Doc-oriented VQA", "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
)


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
    namespace = {}
    _pure_functions(METRIC_SOURCES["punctuation"], {"process_punctuation"}, namespace)
    _pure_functions(METRIC_SOURCES["vqa_eval"],
                    {"_process_digit_article", "process_answer", "levenshtein_distance", "anls_compute"},
                    namespace)
    return namespace


def metric_name(dataset):
    name = str(dataset).lower()
    if name.startswith("docvqa"):
        return "ANLS"
    if name.startswith("textvqa"):
        return "VQA consensus"
    if name == "ocrbench":
        return "OCRBench substring accuracy"
    raise ValueError(f"unsupported dataset metric: {dataset}")


def score_item(dataset, prediction, answers, category=None):
    """Return [0,1], using each task's original answer normalization/rule.

    DocVQA preserves VLMEvalKit's original-string distance denominator and
    accepts exactly 0.5 similarity. TextVQA preserves annotator multiplicity
    and averages ten leave-one-annotator-out agreements (not min(count/3,1)).
    OCRBench uses case-sensitive, space-free math substring matching, and
    lowercase substring matching for the remaining categories.
    """
    if not isinstance(prediction, str):
        raise ValueError("prediction must be a string")
    if not isinstance(answers, (list, tuple)) or not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError("answers must be a nonempty string sequence")
    name = metric_name(dataset)
    if name == "ANLS":
        distance = min(_official_functions()["anls_compute"](a, prediction) for a in answers)
        similarity = 1.0 - distance
        return float(similarity if similarity >= .5 else 0.)
    if name == "VQA consensus":
        normalize = _official_functions()["process_answer"]
        pred, references = normalize(prediction), [normalize(a) for a in answers]
        return float(sum(min(1., sum(other == pred for j, other in enumerate(references) if j != i) / 3.)
                         for i in range(len(references))) / len(references))
    if category not in OCR_CATEGORIES:
        raise ValueError(f"unknown OCRBench category: {category}")
    if category == "Handwritten Mathematical Expression Recognition":
        normalize = lambda value: value.strip().replace("\n", " ").replace(" ", "")
    else:
        normalize = lambda value: value.lower().strip().replace("\n", " ")
    pred = normalize(prediction)
    return float(any(normalize(a) in pred for a in answers))
