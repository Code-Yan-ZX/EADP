"""Train 4 target-conditioned, otherwise identical forward-only scorers.

Only fit/val manifests and their caches are read.  One independent checkpoint
per target and seed is selected solely by validation teacher head recall.
The test manifest/cache are deliberately not inputs to this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

from student_model import StudentScorer

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
OUT = QWEN / "outputs" / "student_distill"
TEACHERS = ("magnitude_v", "magnitude_joint", "signed_v", "signed_joint")
SEEDS = (11, 23, 37)
N_VIS, K, HEAD = 1024, 256, 32
BATCH, UPDATES, VAL_EVERY = 8, 900, 30
LR, WEIGHT_DECAY, RANK_WEIGHT, MARGIN = 1e-3, 1e-4, 0.5, 1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     delete=False) as f:
        temp = Path(f.name)
        torch.save(obj, f)
    temp.replace(path)


def read_manifest(path: Path, expected: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("role") != expected or not isinstance(data.get("items"), list):
        raise ValueError(f"invalid {expected} manifest: {path}")
    items = data["items"]
    ids = [item["item_id"] for item in items]
    parents = [item["parent_id"] for item in items]
    if not items or len(ids) != len(set(ids)):
        raise ValueError(f"empty or duplicate {expected} IDs")
    if any(not isinstance(i, str) or Path(i).name != i or i in (".", "..") for i in ids):
        raise ValueError("unsafe item ID")
    return ids, set(parents)


def verify_training_inputs(fit_manifest, val_manifest, cache_dir, fit_ids, val_ids):
    """Verify the pre-label freeze and all fit/val caches, without reading test."""
    freeze_path, index_path = OUT / "train_freeze.json", OUT / "cache_index.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not freeze.get("pre_teacher") or not index.get("complete"):
        raise RuntimeError("missing pre-teacher freeze or incomplete cache")
    if index.get("train_freeze_sha256") != sha256(freeze_path):
        raise RuntimeError("cache was generated under a different train freeze")
    for role, path in (("fit", fit_manifest), ("val", val_manifest)):
        if freeze["manifest_sha256"].get(role) != sha256(path):
            raise RuntimeError(f"{role} manifest differs from training freeze")
    for name in ("student_model.py", "train_student.py"):
        path = (HERE / name).resolve()
        if freeze["source_sha256"].get(str(path)) != sha256(path):
            raise RuntimeError(f"student source differs from training freeze: {name}")
    items = index.get("items", {})
    if set(items) != set(fit_ids) | set(val_ids):
        raise RuntimeError("cache index does not contain exactly fit and val IDs")
    for role, ids in (("fit", fit_ids), ("val", val_ids)):
        for item_id in ids:
            entry = items[item_id]
            path = (cache_dir / (item_id + ".pt")).resolve()
            if entry.get("role") != role or Path(entry["path"]).resolve() != path:
                raise RuntimeError(f"cache role/path differs: {item_id}")
            if sha256(path) != entry.get("sha256"):
                raise RuntimeError(f"cache SHA mismatch: {item_id}")
    return sha256(freeze_path), sha256(index_path)


def load_rows(ids, cache_dir: Path, feature_dim: int = 4096):
    rows = []
    for item_id in ids:
        path = cache_dir / (item_id + ".pt")
        row = torch.load(path, map_location="cpu", weights_only=True)
        if row["item_id"] != item_id:
            raise ValueError(f"cache ID mismatch: {path}")
        features, question, scores = row["features"], row["question"], row["teacher_scores"]
        if (features.shape != (4, N_VIS, feature_dim) or
            question.shape != (feature_dim,) or
            features.dtype != torch.bfloat16 or question.dtype != torch.bfloat16):
            raise ValueError(f"invalid feature/ question cache {path}")
        if not torch.isfinite(features).all() or not torch.isfinite(question).all():
            raise ValueError(f"nonfinite input {path}")
        for key in TEACHERS:
            if scores[key].shape != (N_VIS,) or scores[key].dtype != torch.float32:
                raise ValueError(f"invalid {key} score in {path}")
            if not torch.isfinite(scores[key]).all():
                raise ValueError(f"nonfinite {key} score in {path}")
        rows.append({"item_id": item_id, "features": features,
                     "question": question, "teacher_scores": scores})
    return rows


def targets(scores: torch.Tensor):
    """Stable raster tie-break. Top-K membership and head-vs-outside pairs."""
    if scores.ndim != 2 or scores.shape[1] != N_VIS or not torch.isfinite(scores).all():
        raise ValueError("bad teacher score shape or values")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    binary = torch.zeros_like(scores)
    binary.scatter_(1, order[:, :K], 1.0)
    return binary, order[:, :HEAD], order[:, K:]


def training_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor):
    labels, positives, negatives = targets(teacher_scores)
    bce = F.binary_cross_entropy_with_logits(student_scores, labels,
             pos_weight=student_scores.new_tensor((N_VIS - K) / K))
    pos = student_scores.gather(1, positives)
    neg = student_scores.gather(1, negatives)
    weights = 1.0 / torch.arange(1, HEAD + 1, device=pos.device, dtype=pos.dtype)
    weights = weights / weights.mean()
    ranking = (F.relu(MARGIN - pos.unsqueeze(-1) + neg.unsqueeze(1)) *
               weights.view(1, -1, 1)).mean()
    return bce + RANK_WEIGHT * ranking


def batch(rows, indices, teacher, device):
    selected = [rows[int(i)] for i in indices]
    return (torch.stack([r["features"] for r in selected]).to(device),
            torch.stack([r["question"] for r in selected]).to(device),
            torch.stack([r["teacher_scores"][teacher] for r in selected]).to(device))


def label_diagnostics(rows):
    """Record degenerate maps and score ties; never drop or relabel a row."""
    result = {}
    for teacher in TEACHERS:
        counts = {"all_equal": 0, "head_boundary_tie": 0, "k_boundary_tie": 0}
        for row in rows:
            values = row["teacher_scores"][teacher]
            ordered = values[torch.argsort(values, descending=True, stable=True)]
            counts["all_equal"] += int(bool(ordered[0] == ordered[-1]))
            counts["head_boundary_tie"] += int(bool(ordered[HEAD-1] == ordered[HEAD]))
            counts["k_boundary_tie"] += int(bool(ordered[K-1] == ordered[K]))
        result[teacher] = counts
    return result


@torch.no_grad()
def validate(model, rows, teacher, device):
    model.eval()
    total = 0
    per_item = []
    for start in range(0, len(rows), BATCH):
        features, question, scores = batch(rows, range(start, min(start+BATCH, len(rows))), teacher, device)
        prediction = model(features, question)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("nonfinite validation prediction")
        teacher_top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :8]
        student_top = torch.argsort(prediction, dim=1, descending=True, stable=True)[:, :K]
        value = (teacher_top.unsqueeze(-1) == student_top.unsqueeze(1)).any(-1).sum(-1)
        total += int(value.sum())
        per_item.extend((value.float() / 8).cpu().tolist())
    return total / (8 * len(rows)), per_item


def run_one(fit, val, teacher, seed, device, output_dir):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = StudentScorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, best_step, best_state = -1.0, None, None
    queue, history, recent_losses = [], [], []
    model.train()
    for step in range(1, UPDATES + 1):
        while len(queue) < BATCH:
            queue.extend(torch.randperm(len(fit), generator=generator).tolist())
        ids, queue = queue[:BATCH], queue[BATCH:]
        features, question, scores = batch(fit, ids, teacher, device)
        prediction = model(features, question)
        loss = training_loss(prediction, scores)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss for {teacher} seed {seed} step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        recent_losses.append(float(loss.detach().cpu()))
        if step % VAL_EVERY == 0:
            score, per_item = validate(model, val, teacher, device)
            history.append({"step": step, "val_head_recall8": score,
                            "val_per_item": per_item,
                            "mean_loss": sum(recent_losses) / len(recent_losses)})
            recent_losses = []
            if score > best_score:  # ties deliberately retain the earliest step
                best_score, best_step = score, step
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            print(f"{teacher} seed={seed} update={step}/{UPDATES} val={score:.6f} best={best_score:.6f}@{best_step}", flush=True)
            model.train()
    if best_state is None:
        raise AssertionError("validation did not run")
    name = f"{teacher}__seed_{seed}.pt"
    checkpoint = {"model_state_dict": best_state, "config": model.config,
                  "teacher": teacher, "seed": seed, "best_step": best_step,
                  "val_head_recall8": best_score, "updates_completed": UPDATES}
    atomic_torch(output_dir / name, checkpoint)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    return name, {"teacher": teacher, "seed": seed, "updates_completed": UPDATES,
                  "best_step": best_step, "val_head_recall8": best_score,
                  "elapsed_seconds": elapsed, "peak_allocated_bytes": peak_allocated,
                  "peak_reserved_bytes": peak_reserved,
                  "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-manifest", type=Path, default=OUT / "data/fit_manifest.json")
    parser.add_argument("--val-manifest", type=Path, default=OUT / "data/val_manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=OUT / "cache")
    parser.add_argument("--output-dir", type=Path, default=OUT / "checkpoints")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if (args.output_dir / "train_complete.json").exists():
        raise FileExistsError("completed training exists; use a new output directory")
    fit_ids, fit_parents = read_manifest(args.fit_manifest, "fit")
    val_ids, val_parents = read_manifest(args.val_manifest, "val")
    if set(fit_ids) & set(val_ids) or fit_parents & val_parents:
        raise ValueError("fit/val ID or parent overlap")
    freeze_sha, cache_index_sha = verify_training_inputs(
        args.fit_manifest, args.val_manifest, args.cache_dir, fit_ids, val_ids)
    fit, val = load_rows(fit_ids, args.cache_dir), load_rows(val_ids, args.cache_dir)
    label_stats = {"fit": label_diagnostics(fit), "val": label_diagnostics(val)}
    print("label_diagnostics", json.dumps(label_stats), flush=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        preload_started = time.perf_counter()
        for row in fit + val:
            row["features"] = row["features"].to(device)
            row["question"] = row["question"].to(device)
        torch.cuda.synchronize(device)
        preload = {"device": str(device), "elapsed_seconds": time.perf_counter()-preload_started,
                   "allocated_bytes": torch.cuda.memory_allocated(device),
                   "reserved_bytes": torch.cuda.memory_reserved(device),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    else:
        preload = {"device": str(device), "allocated_bytes": None, "reserved_bytes": None,
                   "peak_allocated_bytes": None, "elapsed_seconds": 0.0}
    print("feature_preload", json.dumps(preload), flush=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints, runs = {}, []
    for teacher in TEACHERS:
        for seed in SEEDS:
            name, record = run_one(fit, val, teacher, seed, device, output_dir)
            checkpoints[name] = sha256(output_dir / name)
            runs.append(record)
            atomic_json(output_dir / "training_progress.json", {"complete": False, "runs": runs})
    if len(runs) != len(TEACHERS) * len(SEEDS):
        raise AssertionError("missing target/seed run")
    payload = {"complete": True, "teachers": list(TEACHERS), "seeds": list(SEEDS),
               "fit_count": len(fit), "val_count": len(val), "updates_per_run": UPDATES,
               "batch_size": BATCH, "validation_every": VAL_EVERY,
               "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
               "rank_weight": RANK_WEIGHT, "rank_margin": MARGIN,
               "label_diagnostics": label_stats,
               "feature_preload": preload,
               "train_freeze_sha256": freeze_sha, "cache_index_sha256": cache_index_sha,
               "fit_manifest_sha256": sha256(args.fit_manifest),
               "val_manifest_sha256": sha256(args.val_manifest),
               "sources_sha256": {name: sha256(HERE / name) for name in
                                   ("student_model.py", "train_student.py")},
               "checkpoints": checkpoints, "runs": runs}
    atomic_json(output_dir / "train_complete.json", payload)
    print(f"COMPLETE {output_dir / 'train_complete.json'}", flush=True)


if __name__ == "__main__":
    main()
