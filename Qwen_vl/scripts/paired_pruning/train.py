"""Offline paired pruning training. Run from Qwen_vl with python -m scripts.paired_pruning.train.

Only fit/val caches are opened. Checkpoints are chosen by validation objective
loss (earliest tie), independently for every arm/seed; test answers never enter
training or model selection. Existing runs resume only with identical sources,
protocol, manifest, cache index, and training configuration.

Sets objective: E(h,c) = -mean_token BCE(logit_h, complete_bank_mask_c).
Each head normalizes E/temperature over THIS FINITE BANK. Gate probabilities mix
these bank distributions. Success mass, a success/failure energy margin, and
whole-mask matching anchors train multimode. Each head also matches its nearest
whole successful mask. The gate learns a detached softmax target from those
per-head reconstruction costs, including banks with no failed candidates.
This is not the probability of an
arbitrary fixed-K subset among N-choose-K subsets. Inference ignores the bank.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

from .artifacts import load_protocol
from .models import PruningScorer

ARMS = {"query": ("bag_rank", "ordered_rank", "ordered_pair"),
        "sets": ("canonical", "token_marginal", "multimode")}
HERE = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); temporary = Path(handle.name)
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", delete=False) as handle:
        torch.save(value, handle); temporary = Path(handle.name)
    temporary.replace(path)


def rank_anchor(scores, teacher, k):
    """Per-question BCE plus teacher-head/outsider rank loss; all modes share labels."""
    if scores.ndim == 2:
        scores = scores[:, None, :]
    if teacher.shape != (scores.shape[0], scores.shape[-1]):
        raise ValueError("teacher/scores shape mismatch")
    n = scores.shape[-1]
    if not 0 < k < n:
        raise ValueError("rank objective requires 0 < K < N")
    order = torch.argsort(teacher, descending=True, stable=True, dim=-1)
    target = torch.zeros_like(teacher).scatter(1, order[:, :k], 1.0)
    bce = F.binary_cross_entropy_with_logits(scores, target[:, None].expand_as(scores),
               pos_weight=scores.new_tensor((n-k)/k), reduction="none").mean((1, 2))
    head = min(32, k)
    pos = scores.gather(-1, order[:, None, :head].expand(-1, scores.shape[1], -1))
    neg = scores.gather(-1, order[:, None, k:].expand(-1, scores.shape[1], -1))
    weights = 1/torch.arange(1, head+1, device=scores.device, dtype=scores.dtype)
    weights = weights/weights.mean()
    rank = (F.relu(1-pos[..., None]+neg[..., None, :])*weights[None, None, :, None]).mean((1, 2, 3))
    return bce + 0.5*rank


def normalized_utility(teacher):
    """Scale-invariant log compression; degenerate maps remain defined, not dropped."""
    if bool((teacher < 0).any()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("query target requires finite nonnegative magnitude teacher scores")
    relative = teacher / teacher.mean(-1, keepdim=True).clamp_min(1e-30)
    logged = torch.log1p(relative)
    centered = logged - logged.mean(-1, keepdim=True)
    return (centered / centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5)).clamp(-8, 8)


def paired_utility_loss(scores, teacher, page_groups):
    """Equal-page mean of all within-page question-pair utility differences."""
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError("query pair objective expects one mode")
        scores = scores[:, 0]
    if len(page_groups) != len(scores):
        raise ValueError("one page group required for each question")
    targets = normalized_utility(teacher)
    centered = scores-scores.mean(-1, keepdim=True)
    predicted = centered/centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
    losses = []
    for group in dict.fromkeys(page_groups):
        rows = [i for i, value in enumerate(page_groups) if value == group]
        pairs = []
        for i, j in itertools.combinations(rows, 2):
            delta = targets[i]-targets[j]
            weights = 1+delta.abs()
            term = F.smooth_l1_loss(predicted[i]-predicted[j], delta, reduction="none")
            pairs.append((term*weights).sum()/weights.sum())
        if pairs:
            losses.append(torch.stack(pairs).mean())
    return torch.stack(losses).mean() if losses else scores.sum()*0


def mask_energies(scores, masks):
    """[H,N] and [C,N] -> negative whole-mask mean BCE [H,C]."""
    if scores.ndim != 2 or masks.ndim != 2 or scores.shape[-1] != masks.shape[-1]:
        raise ValueError("invalid whole-mask energy shapes")
    logits = scores[:, None, :].expand(-1, masks.shape[0], -1)
    return -F.binary_cross_entropy_with_logits(logits, masks[None].float().expand_as(logits),
                                               reduction="none").mean(-1)


def finite_bank_log_probs(scores, gates, masks, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    energies = mask_energies(scores, masks)
    per_head = F.log_softmax(energies/temperature, dim=-1)
    mixed = torch.logsumexp(F.log_softmax(gates, dim=-1)[:, None]+per_head, dim=0)
    return mixed, energies


def _assignment_anchor(cost, masks):
    """One-to-one matching of <=H distinct, far-apart success masks to modes.

    Enumeration is tiny (at most 4! by default); the minimum-cost assignment
    routes gradients through selected whole-mask BCE terms. Additional
    successes remain covered by the separate all-success best-head anchor.
    """
    h, c = cost.shape
    chosen = [0]
    distances = (masks[:, None] != masks[None, :]).float().mean(-1)
    while len(chosen) < min(h, c):
        nearest = distances[:, chosen].min(-1).values
        nearest[chosen] = -1
        chosen.append(int(nearest.argmax()))
    assignments = list(itertools.permutations(range(h), len(chosen)))
    indices = torch.tensor(assignments, device=cost.device)
    columns = torch.tensor(chosen, device=cost.device)[None]
    totals = cost[indices, columns].mean(-1)
    return totals.min()


def set_question_loss(scores, gates, teacher, bank_masks, success, nll, arm, k,
                      temperature=0.1, margin=0.05):
    """One question, same model family for all controls; failed masks never become positives."""
    if arm not in ARMS["sets"]:
        raise ValueError("unknown sets arm")
    if (bank_masks.ndim != 2 or bank_masks.shape[0] == 0 or
        bank_masks.shape[1] != scores.shape[-1] or
        success.shape != bank_masks.shape[:1] or success.dtype != torch.bool or
        nll.shape != success.shape):
        raise ValueError("invalid candidate bank")
    if not bool((bank_masks.sum(-1) == k).all()):
        raise ValueError("every bank candidate must have exactly K tokens")
    # Identical candidates cannot have conflicting success labels. Remove same-
    # label duplicates so candidate multiplicity cannot change mixture mass.
    unique = []
    for i in range(len(bank_masks)):
        previous = next((j for j in unique if torch.equal(bank_masks[i], bank_masks[j])), None)
        if previous is None:
            unique.append(i)
        elif bool(success[i] != success[previous]):
            raise ValueError("identical mask has conflicting success labels")
    bank_masks, success, nll = bank_masks[unique], success[unique], nll[unique]
    if not bool(success.any()):
        fallback = rank_anchor(scores[None], teacher[None], k).mean()
        gate = F.cross_entropy(gates[None], gates.new_zeros(1, dtype=torch.long))
        return fallback+0.1*gate, {"fallback": 1, "success_masks": 0}
    positives = bank_masks[success]
    if arm == "canonical":
        positive_nll = nll[success]
        if not bool(torch.isfinite(positive_nll).all()):
            raise ValueError("successful candidate NLL must be finite")
        target = positives[positive_nll.argmin()].float()
        loss = F.binary_cross_entropy_with_logits(scores, target[None].expand_as(scores))
        loss = loss+0.1*F.cross_entropy(gates[None], gates.new_zeros(1, dtype=torch.long))
    elif arm == "token_marginal":
        target = positives.float().mean(0)
        loss = F.binary_cross_entropy_with_logits(scores, target[None].expand_as(scores))
        loss = loss+0.1*F.cross_entropy(gates[None], gates.new_zeros(1, dtype=torch.long))
    else:
        logp, energies = finite_bank_log_probs(scores, gates, bank_masks, temperature)
        success_mass = -torch.logsumexp(logp[success], dim=0)
        complete_energy = temperature*torch.logsumexp(
            F.log_softmax(gates, -1)[:, None]+energies/temperature, dim=0)
        failure_margin = scores.sum()*0
        if bool((~success).any()):
            failure_margin = F.relu(margin-complete_energy[success, None]+complete_energy[None, ~success]).mean()
        success_cost = -energies[:, success]
        best_match = success_cost.min(dim=0).values.mean()
        assignment = _assignment_anchor(success_cost, positives)
        nearest_success = success_cost.min(dim=1).values
        # A small/all-success bank leaves some heads unassigned and makes the
        # success-mass term constant. Train every head toward an ENTIRE success
        # mask, and explicitly teach the deployment gate which heads can do so.
        # No coordinate-wise marginal target is used by either term.
        gate_target = F.softmax(-nearest_success.detach()/temperature, dim=0)
        gate_supervision = -(gate_target*F.log_softmax(gates, dim=0)).sum()
        # Anchors are necessary even if EVERY bank candidate is successful:
        # success mass alone would then be identically one, with no gradient.
        loss = (success_mass+failure_margin+0.5*best_match+0.5*assignment+
                0.25*nearest_success.mean()+0.1*gate_supervision)
    return loss, {"fallback": 0, "success_masks": int(success.sum())}


def objective(output, batch, arm, k, loss_config):
    scores, gates = output["scores"], output["gate_logits"]
    if arm in ARMS["query"]:
        anchor = rank_anchor(scores, batch["teacher"], k)
        # Each page has equal weight even if it supplies more questions.
        means = [anchor[[i for i, p in enumerate(batch["page_groups"]) if p == group]].mean()
                 for group in dict.fromkeys(batch["page_groups"])]
        loss = torch.stack(means).mean()
        pair = paired_utility_loss(scores, batch["teacher"], batch["page_groups"])
        if arm == "ordered_pair":
            loss = loss+float(loss_config["pair_weight"])*pair
        return loss, {"questions": len(scores), "fallback": 0, "pair_loss": float(pair.detach())}
    values, fallback = [], 0
    for i, row in enumerate(batch["questions"]):
        loss, info = set_question_loss(scores[i], gates[i], batch["teacher"][i],
            row["bank_masks"].to(scores.device), row["bank_success"].to(scores.device),
            row["bank_nll"].to(scores.device), arm, k,
            float(loss_config["temperature"]), float(loss_config["margin"]))
        values.append(loss); fallback += info["fallback"]
    per_question = torch.stack(values)
    means = [per_question[[i for i, p in enumerate(batch["page_groups"]) if p == group]].mean()
             for group in dict.fromkeys(batch["page_groups"])]
    return torch.stack(means).mean(), {"questions": len(scores), "fallback": fallback}


class PageCache:
    def __init__(self, run_dir, protocol, experiment):
        self.run_dir, self.protocol, self.experiment = Path(run_dir), protocol, experiment
        self.index_path = self.run_dir/"cache/index.json"
        self.manifest_path = self.run_dir/"data/manifest.json"
        self.index = json.loads(self.index_path.read_text())
        # Manifest bytes are hashed for provenance, but never parsed here:
        # the full manifest also contains held-out test answer strings.
        if self.index.get("complete") is not True:
            raise ValueError("cache is incomplete")
        self.protocol_sha = sha256(self.run_dir/"protocol.json")
        if self.index.get("protocol_sha256") != self.protocol_sha:
            raise ValueError("cache/protocol SHA mismatch")
        self.pages, self.entries = {"fit": [], "val": []}, {}
        for page_id, entry in self.index["pages"].items():
            role = entry.get("role")
            if role not in self.pages:
                raise ValueError("training cache index must contain ONLY fit/val pages")
            path = Path(entry["path"])
            if not path.is_absolute():
                path = (self.run_dir/path).resolve()
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"cache SHA mismatch: {page_id}")
            self.entries[page_id] = (role, entry, path)
            self.pages[role].append({"page_id": page_id})
        for role, pages in self.pages.items():
            expected_count = protocol.get("split_pages", {}).get(role)
            if not pages or (expected_count is not None and len(pages) != expected_count):
                raise ValueError(f"empty or wrong-size {role} cache roster")
        self.loaded = OrderedDict()
        parents, item_ids = {"fit": set(), "val": set()}, set()
        self.diagnostics = {role: dict(pages=len(self.pages[role]), questions=0,
                                      no_success_questions=0) for role in self.pages}
        for page_id, (role, _, _) in self.entries.items():
            data = self.get(page_id)
            parents[role].add(data["parent_id"])
            for row in data["questions"]:
                if row["item_id"] in item_ids:
                    raise ValueError("duplicate question ID in training caches")
                item_ids.add(row["item_id"])
                self.diagnostics[role]["questions"] += 1
                if self.experiment == "sets":
                    self.diagnostics[role]["no_success_questions"] += int(not bool(row["bank_success"].any()))
        if parents["fit"] & parents["val"]:
            raise ValueError("fit/validation parent-document overlap")
        for role in self.diagnostics:
            info = self.diagnostics[role]
            info["no_success_fraction"] = (info["no_success_questions"]/info["questions"]
                                             if experiment == "sets" else None)

    def get(self, page_id):
        if page_id in self.loaded:
            self.loaded.move_to_end(page_id)
            return self.loaded[page_id]
        role, entry, path = self.entries[page_id]
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data.get("page_id") != page_id or not isinstance(data.get("parent_id"), str) or
            data.get("role") != role or data.get("protocol_sha256") != self.protocol_sha):
            raise ValueError(f"cache page metadata mismatch: {page_id}")
        cfg = self.protocol["model_config"]
        expected = (cfg.get("streams", 4), self.protocol["n"], cfg.get("feature_dim", 4096))
        features = data["features"]
        if tuple(features.shape) != expected or features.dtype != torch.bfloat16 or not bool(torch.isfinite(features).all()):
            raise ValueError(f"invalid visual features: {page_id}")
        question_ids = [q["item_id"] for q in data["questions"]]
        if (len(question_ids) != entry.get("questions") or
            len(question_ids) != len(set(question_ids)) or
            (self.protocol.get("questions_per_page") is not None and
             len(question_ids) != self.protocol["questions_per_page"])):
            raise ValueError("cache question count differs from frozen index/protocol or IDs duplicate")
        if len(question_ids) < 2:
            raise ValueError("paired experiment requires >=2 questions per training page")
        for question in data["questions"]:
            q, teacher = question["question_embeddings"], question["teacher_scores"]
            if (q.ndim != 2 or q.shape[1] != expected[-1] or q.dtype != torch.bfloat16 or
                not 1 <= q.shape[0] <= cfg.get("max_question_tokens", 256) or
                not bool(torch.isfinite(q).all()) or teacher.shape != (expected[1],) or
                not bool(torch.isfinite(teacher).all()) or bool((teacher < 0).any())):
                raise ValueError("invalid query/teacher cache")
            if self.experiment == "sets":
                masks, success = question["bank_masks"], question["bank_success"]
                c = len(masks)
                if (masks.dtype != torch.bool or masks.shape != (c, expected[1]) or c < 1 or
                    success.dtype != torch.bool or success.shape != (c,) or
                    question["bank_nll"].shape != (c,) or question["bank_anls"].shape != (c,) or
                    len(question["bank_names"]) != c or not bool((masks.sum(-1) == self.protocol["k"]).all()) or
                    not bool(torch.isfinite(question["bank_nll"]).all()) or
                    not bool(torch.isfinite(question["bank_anls"]).all())):
                    raise ValueError("invalid candidate bank cache")
        self.loaded[page_id] = data
        while len(self.loaded) > 2:
            self.loaded.popitem(last=False)
        return data


def collate_pages(pages, device):
    rows, features, groups = [], [], []
    for page in pages:
        for row in page["questions"]:
            rows.append(row); features.append(page["features"]); groups.append(page["page_id"])
    max_length = max(row["question_embeddings"].shape[0] for row in rows)
    dim = rows[0]["question_embeddings"].shape[-1]
    question = torch.zeros(len(rows), max_length, dim, dtype=torch.bfloat16)
    mask = torch.zeros(len(rows), max_length, dtype=torch.bool)
    for i, row in enumerate(rows):
        m = row["question_embeddings"].shape[0]
        question[i, :m] = row["question_embeddings"]; mask[i, :m] = True
    return dict(features=torch.stack(features).to(device), question=question.to(device),
                question_mask=mask.to(device), teacher=torch.stack([r["teacher_scores"] for r in rows]).to(device),
                page_groups=groups, questions=rows)


def _model_call(model, batch):
    return model(batch["features"], batch["question"], batch["question_mask"])


@torch.no_grad()
def validation(model, cache, arm, device, loss_config):
    model.eval(); losses, fallback, count = [], 0, 0
    for page in cache.pages["val"]:
        batch = collate_pages([cache.get(page["page_id"])], device)
        loss, info = objective(_model_call(model, batch), batch, arm, cache.protocol["k"], loss_config)
        losses.append(float(loss)); fallback += info["fallback"]; count += info["questions"]
    model.train()
    return {"loss": sum(losses)/len(losses), "pages": len(losses), "questions": count,
            "fallback": fallback, "fallback_fraction": fallback/count}


def matched_page_indices(n_pages, batch_pages, seed, step):
    """Stateless arm-independent page batches, exactly reproducible after resume."""
    generator = torch.Generator().manual_seed(int(seed)*1000003+int(step))
    return torch.randperm(n_pages, generator=generator)[:min(batch_pages, n_pages)].tolist()


def model_config(protocol, experiment, arm):
    config = dict(protocol["model_config"])
    config["ordered"] = arm != "bag_rank"
    config["mode_heads"] = int(protocol.get("mode_heads", 4)) if experiment == "sets" else 1
    if config["mode_heads"] > 4:
        raise ValueError("enumerated mode matching supports at most 4 mode heads")
    return config


def checkpoint_identity(payload):
    """Semantic identity independent of torch.save's archive serialization."""
    digest = hashlib.sha256(json.dumps({key: payload[key] for key in
        ("format_version", "kind", "config", "metadata")}, sort_keys=True, allow_nan=False).encode())
    for name, tensor in sorted(payload["state_dict"].items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def materialize_selected(path, best_checkpoint, preserve_mismatch=False):
    """Selected file is a derived export of the atomic resume transaction."""
    path = Path(path)
    if path.exists():
        try:
            present = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint_identity(present) == checkpoint_identity(best_checkpoint):
                return
        except (ValueError, KeyError, RuntimeError, EOFError):
            pass
        if preserve_mismatch:
            orphan = path.with_name(path.name+".orphan."+sha256(path)[:16])
            if orphan.exists():
                if sha256(orphan) != sha256(path):
                    raise ValueError("orphan checkpoint name collision")
            else:
                path.replace(orphan)
    atomic_torch(path, best_checkpoint)


def train_arm(cache, arm, seed, device, settings, provenance):
    output_dir = cache.run_dir/"checkpoints"
    selected_path = output_dir/f"{arm}__seed_{seed}.pt"
    latest_path = output_dir/f"{arm}__seed_{seed}.resume.pt"
    arm_meta = dict(arm=arm, seed=seed, experiment=cache.experiment,
                    provenance=provenance, training=settings,
                    selection="minimum validation objective; earliest tie",
                    inference="gate argmax -> one complete mode -> stable TopK -> raster order")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = PruningScorer(model_config(cache.protocol, cache.experiment, arm)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    start, best, best_step = 0, math.inf, 0
    history, fallback_exposures, question_exposures = [], 0, 0
    best_checkpoint, elapsed_before = None, 0.0
    if latest_path.exists():
        resume = torch.load(latest_path, map_location="cpu", weights_only=True)
        if resume.get("metadata") != arm_meta or resume.get("config") != model.config:
            raise ValueError(f"resume provenance/config differs: {latest_path}")
        model.load_state_dict(resume["state_dict"]); optimizer.load_state_dict(resume["optimizer"])
        start, best, best_step, history = resume["step"], resume["best_loss"], resume["best_step"], resume["history"]
        fallback_exposures, question_exposures = resume["fallback_exposures"], resume["question_exposures"]
        best_checkpoint, elapsed_before = resume["best_checkpoint"], resume["elapsed_seconds"]
        if checkpoint_identity(best_checkpoint) != resume["best_checkpoint_identity"]:
            raise ValueError("resume best-checkpoint identity differs")
        torch.set_rng_state(resume["rng_cpu"])
        if str(device).startswith("cuda") and resume.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(resume["rng_cuda"])
        materialize_selected(selected_path, best_checkpoint, preserve_mismatch=True)
    elif selected_path.exists():
        orphan = torch.load(selected_path, map_location="cpu", weights_only=True)
        expected_meta = orphan.get("metadata", {})
        if any(expected_meta.get(key) != value for key, value in arm_meta.items()):
            raise ValueError("orphan selected checkpoint is from a different run")
        destination = selected_path.with_name(selected_path.name+".orphan."+sha256(selected_path)[:16])
        if destination.exists():
            raise ValueError("unresolved duplicate orphan checkpoint; inspect run artifacts")
        selected_path.replace(destination)
        print(f"Preserved orphan selected checkpoint at {destination}; replaying this arm from step 0", flush=True)
    started = time.monotonic()
    model.train()
    for step in range(start+1, settings["steps"]+1):
        indices = matched_page_indices(len(cache.pages["fit"]), settings["batch_pages"], seed, step)
        pages = [cache.get(cache.pages["fit"][i]["page_id"]) for i in indices]
        batch = collate_pages(pages, device)
        optimizer.zero_grad(set_to_none=True)
        loss, info = objective(_model_call(model, batch), batch, arm, cache.protocol["k"], settings["loss"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss: {arm} seed {seed} step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        fallback_exposures += info["fallback"]; question_exposures += info["questions"]
        if step % settings["val_every"] == 0 or step == settings["steps"]:
            val = validation(model, cache, arm, device, settings["loss"])
            if not math.isfinite(val["loss"]):
                raise FloatingPointError("nonfinite validation objective")
            history.append(dict(step=step, train_loss=float(loss.detach()), grad_norm=float(norm), validation=val))
            if val["loss"] < best:
                best, best_step = val["loss"], step
                metadata = dict(arm_meta, step=step, val_loss=best, validation=val,
                                parameters=sum(p.numel() for p in model.parameters()))
                best_checkpoint = dict(format_version=1, kind="selected", config=model.config,
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, metadata=metadata)
            elapsed = elapsed_before+time.monotonic()-started
            # Commit authoritative state FIRST. An interruption before exporting
            # selected.pt is repaired idempotently from this single transaction.
            atomic_torch(latest_path, dict(format_version=1, kind="resume", config=model.config,
                metadata=arm_meta, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step, best_loss=best, best_step=best_step, history=history,
                fallback_exposures=fallback_exposures, question_exposures=question_exposures,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
                best_checkpoint=best_checkpoint, best_checkpoint_identity=checkpoint_identity(best_checkpoint),
                elapsed_seconds=elapsed))
            materialize_selected(selected_path, best_checkpoint)
            print(f"{arm} seed={seed} step={step}/{settings['steps']} val={val['loss']:.6f} best={best:.6f}@{best_step}", flush=True)
    return dict(arm=arm, seed=seed, path=str(selected_path.resolve()), sha256=sha256(selected_path),
                best_step=best_step, val_loss=best, updates=settings["steps"], history=history,
                fallback_exposures=fallback_exposures, question_exposures=question_exposures,
                fallback_exposure_fraction=fallback_exposures/max(1, question_exposures),
                elapsed_training_seconds=elapsed if start < settings["steps"] else elapsed_before)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=tuple(ARMS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    protocol_path = run_dir/"protocol.json"
    protocol = load_protocol(run_dir)
    steps, seeds = int(protocol.get("steps", 900)), list(protocol.get("seeds", [11, 23, 37]))
    if (args.steps is not None and args.steps != steps) or (args.seeds is not None and args.seeds != seeds):
        parser.error("--steps/--seeds must exactly match the frozen protocol.json")
    if steps < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive updates and unique seeds required")
    settings = dict(steps=steps, batch_pages=int(protocol.get("batch_pages", 2)),
        val_every=int(protocol.get("val_every", 30)), learning_rate=float(protocol.get("learning_rate", 0.001)),
        weight_decay=float(protocol.get("weight_decay", 0.0001)),
        loss=dict(pair_weight=float(protocol.get("pair_weight", 0.25)),
                  temperature=float(protocol.get("bank_temperature", 0.1)), margin=float(protocol.get("bank_margin", 0.05))))
    if settings["batch_pages"] < 1 or settings["val_every"] < 1 or settings["learning_rate"] <= 0:
        raise ValueError("invalid training schedule")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU training fallback")
    torch.use_deterministic_algorithms(True)
    cache = PageCache(run_dir, protocol, args.experiment)
    provenance = dict(protocol_sha256=sha256(protocol_path), cache_index_sha256=sha256(cache.index_path),
        manifest_sha256=sha256(cache.manifest_path),
        source_sha256={str(HERE/name): sha256(HERE/name) for name in ("models.py", "train.py")},
        torch_version=str(torch.__version__), device=args.device)
    freeze_path = run_dir/"checkpoints"/f"{args.experiment}_train_freeze.json"
    freeze = dict(provenance=provenance, settings=settings, seeds=seeds, arms=list(ARMS[args.experiment]))
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != freeze:
            raise ValueError("training freeze differs; do not overwrite previous experimental evidence")
    else:
        atomic_json(freeze_path, freeze)
    summary_path = run_dir/"checkpoints"/f"{args.experiment}_summary.json"
    if summary_path.exists():
        completed = json.loads(summary_path.read_text())
        if completed.get("status") == "complete":
            if any(completed.get(key) != value for key, value in provenance.items()):
                raise ValueError("completed training summary provenance differs")
            if (completed.get("arms") != list(ARMS[args.experiment]) or completed.get("seeds") != seeds or
                completed.get("experiment") != args.experiment):
                raise ValueError("completed training experiment/arm/seed roster differs")
            expected = {(arm, seed) for arm in ARMS[args.experiment] for seed in seeds}
            records = completed["checkpoints"]
            if (len(records) != len(expected) or
                {(row["arm"], row["seed"]) for row in records} != expected or
                any(sha256(row["path"]) != row["sha256"] for row in records)):
                raise ValueError("completed training checkpoint roster or hashes differ")
            print(f"{args.experiment} training already complete; preserving frozen summary bytes", flush=True)
            return
    records = []
    for arm in ARMS[args.experiment]:
        for seed in seeds:
            records.append(train_arm(cache, arm, int(seed), args.device, settings, provenance))
            atomic_json(run_dir/"checkpoints"/f"{args.experiment}_summary.json",
                dict(status="running", experiment=args.experiment, arms=list(ARMS[args.experiment]), seeds=seeds,
                     **provenance, checkpoints=records))
    # Recheck the frozen upstream ledger before publishing completion.
    load_protocol(run_dir)
    if sha256(cache.index_path) != provenance["cache_index_sha256"]:
        raise ValueError("cache index changed during training")
    for _, entry, path in cache.entries.values():
        if sha256(path) != entry["sha256"]:
            raise ValueError("cache page changed during training")
    summary = dict(status="complete", experiment=args.experiment, arms=list(ARMS[args.experiment]), seeds=seeds,
                   **provenance, checkpoints=records, label_diagnostics=cache.diagnostics)
    atomic_json(run_dir/"checkpoints"/f"{args.experiment}_summary.json", summary)
    summaries = {}
    for experiment in ARMS:
        path = run_dir/"checkpoints"/f"{experiment}_summary.json"
        if path.exists():
            summaries[experiment] = json.loads(path.read_text())
    atomic_json(run_dir/"checkpoints/summary.json", summaries)


if __name__ == "__main__":
    main()
