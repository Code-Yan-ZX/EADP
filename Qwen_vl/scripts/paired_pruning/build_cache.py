"""Build resumable fit/validation labels using true native K-token subsets.

The dense gradient is only an offline candidate proposal. A sufficient mask
must also pass actual autoregressive QA; a low NLL alone is never success.
Test pages are deliberately absent from this cache and from its API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .artifacts import atomic_json, atomic_torch_save, load_protocol, sha256
from . import runtime


def seed_for(seed, *parts):
    value = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def mask_from_keep(keep, n, k):
    keep = torch.as_tensor(keep, dtype=torch.long, device="cpu").reshape(-1)
    if len(keep) != k or keep.unique().numel() != k or int(keep.min()) < 0 or int(keep.max()) >= n:
        raise ValueError("Candidate is not a unique in-range fixed-K selection")
    mask = torch.zeros(n, dtype=torch.bool)
    mask[keep] = True
    return mask


def candidate_bank(teacher, b1, b2, n, k, count, seed):
    """Deterministic, unique masks. No QA outcomes influence proposal order."""
    if not 8 <= count <= 64:
        raise ValueError("bank_size must be between 8 and 64")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = teacher["scores"]["magnitude_v"].float().cpu()
    if scores.shape != (n,) or not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid dense magnitude teacher")
    proposals = []

    def add(name, keep):
        mask = mask_from_keep(keep, n, k)
        if not any(torch.equal(mask, row[1]) for row in proposals):
            proposals.append((name, mask))

    for key in ("magnitude_v", "magnitude_joint", "signed_v"):
        add(key, teacher["keep_indices"][key])
    add("b1", b1)
    add("b2", b2)
    add("random", torch.randperm(n, generator=gen)[:k])
    # Uniform coverage with exactly K indices, including both raster ends.
    add("spatial", torch.linspace(0, n - 1, k).round().long())
    transformed = torch.log1p(scores / scores.mean().clamp_min(1e-12))
    transformed = (transformed - transformed.mean()) / transformed.std().clamp_min(1e-6)
    attempts = 0
    while len(proposals) < count:
        temperature = (0.25, 0.5, 1.0, 2.0, 4.0)[attempts % 5]
        uniform = torch.rand(n, generator=gen).clamp(1e-6, 1 - 1e-6)
        perturbed = transformed + temperature * (-torch.log(-torch.log(uniform)))
        keep = torch.argsort(perturbed, descending=True, stable=True)[:k]
        add(f"gumbel_{attempts}_t{temperature:g}", keep)
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Cannot construct enough distinct fixed-budget masks")
    return proposals


def sufficient(full, candidate, min_anls=0.9, tolerance=0.02):
    return bool(not full["truncated"] and not candidate["truncated"]
                and full["anls"] >= min_anls
                and candidate["anls"] >= max(min_anls, full["anls"] - tolerance))


def verify_partial(value, protocol_digest, page, question):
    expected = (protocol_digest, page["page_id"], question["item_id"], page["image_sha256"])
    actual = tuple(value.get(name) for name in ("protocol_sha256", "page_id", "item_id", "image_sha256"))
    if actual != expected:
        raise RuntimeError("Partial label artifact belongs to different frozen inputs")


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build(run_dir):
    run = Path(run_dir).resolve()
    protocol = load_protocol(run)
    digest = sha256(run / "protocol.json")
    gate = json.loads((run / "gates/gpu.json").read_text())
    if gate.get("status") != "PASS" or gate.get("protocol_sha256") != digest:
        raise RuntimeError("A passing GPU native gate for this protocol is required")
    manifest = json.loads((run / "data/manifest.json").read_text())
    index_path = run / "cache/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(
        schema_version=1, complete=False, protocol_sha256=digest, pages={})
    if index["protocol_sha256"] != digest:
        raise RuntimeError("Training cache belongs to a different protocol")
    expected_pages = [(role, page) for role in ("fit", "val") for page in manifest["splits"][role]]
    expected_ids = {page["page_id"] for _, page in expected_pages}
    if not set(index["pages"]) <= expected_ids:
        raise RuntimeError("Training cache contains unexpected/test pages")
    for info in index["pages"].values():
        if sha256(info["path"]) != info["sha256"]:
            raise RuntimeError("Committed training cache file was modified")
    if index.get("complete"):
        if set(index["pages"]) != expected_ids:
            raise RuntimeError("Complete cache is missing pages")
        print("fit/validation cache already complete", flush=True)
        return
    eng = runtime.load_engine(protocol["model_path"])
    n, k = protocol["n"], protocol["k"]
    for page_number, (role, page) in enumerate(expected_pages):
        if page["page_id"] in index["pages"]:
            continue
        started = time.perf_counter()
        rows = []
        features = None
        partial_dir = run / "cache/partial" / page["page_id"]
        for question in page["questions"]:
            partial_path = partial_dir / (question["item_id"] + ".pt")
            value = load_torch(partial_path) if partial_path.exists() else None
            if value is not None:
                verify_partial(value, digest, page, question)
            # One fresh vision pass per question; all candidate evaluations
            # reuse exactly these V/DS streams, but receive fresh native KV.
            ctx = runtime.prepare(eng, page, question, protocol["max_question_tokens"])
            if ctx["V"].shape != (n, protocol["model_config"]["feature_dim"]):
                raise RuntimeError("Unexpected fixed native image geometry")
            current_features = torch.stack((ctx["V"], *ctx["DS"])).to(torch.bfloat16).cpu()
            if features is None:
                features = current_features
            elif not torch.equal(features, current_features):
                raise RuntimeError("Same-page visual streams differ between questions")
            if value is None:
                teacher = runtime.teacher_scores(eng, ctx, question["answers"][0], k=k)
                full = runtime.generate(eng, ctx, max_tokens=protocol["max_new_tokens"])
                full["anls"] = runtime.anls(full["text"], question["answers"])
                full["nll"] = teacher["mean_answer_nll"]
                b1 = runtime.baseline_keep(eng, ctx, "b1", k).cpu()
                b2 = runtime.baseline_keep(eng, ctx, "b2", k).cpu()
                proposals = candidate_bank(teacher, b1, b2, n, k, protocol["bank_size"],
                                           seed_for(protocol["seed"], page["page_id"], question["item_id"]))
                value = dict(protocol_sha256=digest, page_id=page["page_id"],
                    item_id=question["item_id"], image_sha256=page["image_sha256"],
                    question_embeddings=ctx["question_embeddings"].cpu(),
                    question_token_ids=ctx["question_token_ids"],
                    teacher_scores=teacher["scores"]["magnitude_v"],
                    full=full, proposed_names=[name for name, _ in proposals],
                    proposed_masks=torch.stack([mask for _, mask in proposals]), results=[])
                atomic_torch_save(partial_path, value)
            if not torch.equal(value["question_embeddings"], ctx["question_embeddings"].cpu()):
                raise RuntimeError("Cached question features do not match fresh tokenization")
            for candidate_index in range(len(value["results"]), len(value["proposed_names"])):
                mask = value["proposed_masks"][candidate_index]
                keep = mask.nonzero().flatten().to(ctx["V"].device)
                prediction = runtime.generate(eng, ctx, keep, protocol["max_new_tokens"])
                prediction["anls"] = runtime.anls(prediction["text"], question["answers"])
                prediction["nll"] = runtime.gold_nll(eng, ctx, question["answers"][0], keep)
                prediction["success"] = sufficient(value["full"], prediction,
                                                    protocol["success_min_anls"],
                                                    protocol["success_tolerance"])
                value["results"].append(prediction)
                atomic_torch_save(partial_path, value)
                print(f"cache {role} page {page_number+1}/{len(expected_pages)} "
                      f"{question['item_id']} candidate {candidate_index+1}/{len(value['proposed_names'])}", flush=True)
            rows.append(dict(item_id=question["item_id"],
                question_embeddings=value["question_embeddings"],
                question_token_ids=value["question_token_ids"], teacher_scores=value["teacher_scores"],
                bank_masks=value["proposed_masks"], bank_names=value["proposed_names"],
                bank_success=torch.tensor([result["success"] for result in value["results"]]),
                bank_nll=torch.tensor([result["nll"] for result in value["results"]]),
                bank_anls=torch.tensor([result["anls"] for result in value["results"]]),
                full_anls=value["full"]["anls"], full_nll=value["full"]["nll"],
                source_partial_path=str(partial_path)))
            del ctx
        path = run / "cache/pages" / (page["page_id"] + ".pt")
        atomic_torch_save(path, dict(schema_version=1, protocol_sha256=digest,
            page_id=page["page_id"], parent_id=page["parent_id"], role=role,
            image_sha256=page["image_sha256"], features=features, questions=rows))
        index["pages"][page["page_id"]] = dict(path=str(path), sha256=sha256(path), role=role,
            questions=len(rows), wall_seconds=time.perf_counter()-started)
        atomic_json(index_path, index)
        print(f"committed {role}/{page['page_id']} ({index['pages'][page['page_id']]['wall_seconds']:.1f}s)", flush=True)
    index["complete"] = set(index["pages"]) == expected_ids
    atomic_json(index_path, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.run_dir)


if __name__ == "__main__":
    main()
