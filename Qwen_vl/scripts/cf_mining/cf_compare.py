"""Matched hard-teacher training controls for the semantic-difference pilot.

This runner tests data selection and refresh with one compression architecture.
Random/fixed/online/replay share the same objective. Plain ordinary/replay_ce
controls omit the optional opposing-answer margin; their lower forward count
is reported. ``replay`` shuffles the exact online positive exposure multiset,
including repeated examples, from the common initial checkpoint.

All mining uses fit examples only.  Fixed final checkpoints are evaluated;
held-out predictions never select a checkpoint or a mining candidate.
``--dry-run`` inspects the manifest and produces a plan without loading torch,
the native model, or any visual-feature cache.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUT = ROOT / "Qwen_vl" / "outputs" / "cf_mining"


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_json(path, payload):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize(value):
    """Keep exactly the miner's numeric correctness convention."""
    value = str(value).strip().lower().strip("'\"`")
    value = re.sub(r"[\s,$£€]", "", value).replace("−", "-")
    suffix = "%" if value.endswith("%") else ""
    number = value[:-1] if suffix else value
    try:
        return str(Decimal(number).normalize()) + suffix
    except InvalidOperation:
        return value


def correct(prediction, answer):
    return normalize(prediction) == normalize(answer)


def split_name(record):
    value = record.get("split", "")
    return {"train": "fit", "val": "validation"}.get(value, value)


def group_id(record):
    for name in ("doc_id", "image_id", "source_image", "pair_id"):
        if record.get(name) is not None:
            return str(record[name])
    raise ValueError("a mining record has no source identity")


def make_plan(mining, args):
    records = mining.get("records", [])
    ids = [str(record["pair_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair_id in mining records")
    # Check the complete available manifest, before sampling or filtering by
    # teacher validity.  A failed teacher pair must not hide source leakage.
    for identity in ("doc_id", "image_id", "source_image"):
        seen = {}
        for record in records:
            key = record.get(identity)
            if key is None:
                continue
            split = split_name(record)
            if str(key) in seen and seen[str(key)] != split:
                raise ValueError(f"{identity} crosses fit/evaluation partitions: {key}")
            seen[str(key)] = split
    rng = random.Random(args.seed)
    valid = [record for record in records if record.get("teacher_valid")]
    for record in valid:
        if normalize(record["original_answer"]) == normalize(record["edited_answer"]):
            raise ValueError(f"counterfactual answers normalize identically: {record['pair_id']}")
    fit = sorted((record for record in valid if split_name(record) == "fit"),
                 key=lambda record: str(record["pair_id"]))
    rng.shuffle(fit)
    fit = fit[:args.pool_pairs]
    blind = [record for record in fit if record.get("blind")]
    # Strict answer collapse is direct evidence of failed discrimination.
    # Broader compression pair failures fill the gate only when necessary.
    blind.sort(key=lambda record: not bool(record.get("collapse")))
    gate = blind[:args.pairs]
    heldout = sorted((record for record in valid if split_name(record) != "fit"),
                     key=lambda record: str(record["pair_id"]))
    random.Random(args.seed + 1).shuffle(heldout)
    heldout = heldout[:args.eval_pairs]
    source = [record for record in mining.get("sources", [])
              if split_name(record) != "fit"]
    source.sort(key=lambda record: str(record["pair_id"]))
    random.Random(args.seed + 2).shuffle(source)
    source = source[:args.eval_sources]
    return dict(fit=fit, blind=blind, gate=gate, heldout=heldout, sources=source)


def public_plan(plan, args):
    return dict(config=vars(args), counts={key: len(value) for key, value in plan.items()},
                pair_ids={key: [record["pair_id"] for record in value]
                          for key, value in plan.items()},
                objective="teacher answer CE+EOS and cached bucket KL; optional opposing-answer margin on edited-pair arms",
                matched_training="two positive answer exposures per optimizer update; contrast adds negative forwards",
                replay="same online exposure multiset shuffled from common initialization",
                checkpoint_selection="fixed final step; no held-out selection",
                limitation="DocVQA VAL pilot partitions; not official benchmark training")


class CacheLoader:
    def __init__(self, torch_module, manifest_path, capacity=8):
        self.torch = torch_module
        self.directory = Path(manifest_path).resolve().parent
        self.capacity = capacity
        self.cache = OrderedDict()

    def resolve(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.directory / path
        return str(path.resolve())

    def __call__(self, path):
        key = self.resolve(path)
        if key not in self.cache:
            self.cache[key] = self.torch.load(key, map_location="cpu", weights_only=False)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def store(self, path, record):
        key = self.resolve(path)
        temporary = key + ".teacher.tmp"
        self.torch.save(record, temporary)
        Path(temporary).replace(key)
        self.cache[key] = record
        self.cache.move_to_end(key)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def pair_summary(rows):
    size = len(rows)
    failures = sum(not row["both_correct"] for row in rows)
    collapses = sum(row["collapse"] and not row["both_correct"] for row in rows)
    return dict(n_pairs=size, n_documents=len({row["group_id"] for row in rows}),
                both_correct=sum(row["both_correct"] for row in rows),
                broad_pair_failures=failures, strict_answer_collapses=collapses,
                broad_pair_failure_rate=failures / size if size else None,
                strict_answer_collapse_rate=collapses / size if size else None,
                both_correct_rate=sum(row["both_correct"] for row in rows) / size if size else None,
                original_correct_rate=sum(row["original_correct"] for row in rows) / size if size else None,
                edited_correct_rate=sum(row["edited_correct"] for row in rows) / size if size else None,
                collapse_rate=sum(row["collapse"] for row in rows) / size if size else None)


def paired_bootstrap(before, after, seed, n_boot=2000):
    """Pair question results and resample whole source documents."""
    old = {row["pair_id"]: row for row in before}
    groups = {}
    for row in after:
        if row["pair_id"] not in old:
            continue
        previous = old[row["pair_id"]]
        groups.setdefault(row["group_id"], []).append(
            int(row["both_correct"]) - int(previous["both_correct"]))
    values = [value for group in groups.values() for value in group]
    if not values:
        return dict(n_pairs=0, n_documents=0, delta=None, ci95=None)
    mean = sum(values) / len(values)
    if len(groups) < 2:
        return dict(n_pairs=len(values), n_documents=len(groups), delta=mean, ci95=None,
                    note="fewer than two independent source documents")
    keys, draws = list(groups), []
    rng = random.Random(seed)
    for _ in range(n_boot):
        sample = [value for key in rng.choices(keys, k=len(keys)) for value in groups[key]]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return dict(n_pairs=len(values), n_documents=len(keys), delta=mean,
                ci95=[draws[int(.025 * (n_boot - 1))], draws[int(.975 * (n_boot - 1))]])


def collapse_changes(before, after):
    old = {row["pair_id"]: row for row in before}
    paired = [(old[row["pair_id"]], row) for row in after if row["pair_id"] in old]
    return dict(initial_strict_collapses=sum(a["collapse"] and not a["both_correct"] for a, _ in paired),
                strict_collapses_repaired_to_both_correct=sum(
                    a["collapse"] and not a["both_correct"] and b["both_correct"] for a, b in paired),
                new_strict_collapses=sum(
                    not (a["collapse"] and not a["both_correct"])
                    and b["collapse"] and not b["both_correct"] for a, b in paired))


class Experiment:
    def __init__(self, args, plan):
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, str(HERE.parent / "discovery"))
        import common
        import cf_model
        from model.native_qwen3 import NativeEngine
        self.args, self.plan, self.torch, self.F, self.api = args, plan, torch, F, cf_model
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        self.wrapper = common.load_model(common.BASELINE_MODEL, max_new_tokens=args.max_new_tokens)
        self.eng = NativeEngine(self.wrapper)
        cf_model.freeze_decoder(self.eng, checkpointing=args.checkpointing)
        self.device = next(self.eng.model.parameters()).device
        width = int(self.eng.text.config.hidden_size)
        torch.manual_seed(args.seed)
        cls = {"pool": cf_model.AnchorPool, "readout": cf_model.AnchorReadout,
               "local_readout": cf_model.LocalReadout}[args.carrier]
        self.pool = cls(width, rank=args.rank).to(self.device)
        self.trainable_parameters = sum(p.numel() for p in self.pool.parameters() if p.requires_grad)
        self.initial = {key: value.detach().cpu().clone()
                        for key, value in self.pool.state_dict().items()}
        self.initial_hash = hashlib.sha256(b"".join(
            tensor.numpy().tobytes() for tensor in self.initial.values())).hexdigest()
        self.versions = {name: parameter._version for name, parameter in self.eng.model.named_parameters()}
        self.initial_decoder_sample = self.decoder_sample_checksum()
        self.cache = CacheLoader(torch, args.mining)
        self.by_id = {str(record["pair_id"]): record for record in plan["fit"]}
        self.forward_counts = dict(training_answer=0, mining_answer=0,
                                   evaluation_generation=0, training_negative_answer=0)
        self.objectives = set()

    def reset(self):
        self.pool.load_state_dict(self.initial)
        self.torch.manual_seed(self.args.seed)
        return self.torch.optim.Adam([
            dict(params=[self.pool.alpha], lr=self.args.gate_lr),
            dict(params=[parameter for name, parameter in self.pool.named_parameters()
                         if name != "alpha"], lr=self.args.lr)])

    def moved(self, path, answer=None):
        record = self.api.record_to_device(self.cache(path), self.device)
        if record.get("teacher_answer") is not None:
            record["answer"] = str(record["teacher_answer"])
        elif answer is not None:
            record["answer"] = str(answer)
        return record

    def prepare_teacher_kl(self, path):
        """Backfill full-teacher top-64 plus exact remaining probability mass."""
        self.api.set_decoder_checkpointing(self.eng, False)
        completed, seen = [], set()
        t0 = time.time()
        for pair in self.plan["fit"]:
            for side in ("original", "edited"):
                cache_path = pair["cache_" + side]
                answer = str(pair.get(side + "_teacher_prediction", pair[side + "_answer"]))
                if not correct(answer, pair[side + "_answer"]):
                    raise ValueError("soft distillation requires a correct full-teacher response")
                key = self.cache.resolve(cache_path)
                if key in seen:
                    continue
                seen.add(key)
                cached = self.cache(cache_path)
                reusable = (cached.get("teacher_answer") == answer and
                            all(name in cached for name in
                                ("teacher_top_idx", "teacher_top_prob", "teacher_rest")))
                if not reusable:
                    # Replace any older target explicitly: moved() otherwise
                    # preserves a previously cached teacher answer by design.
                    record = self.api.record_to_device(cached, self.device)
                    record["answer"] = answer
                    with self.torch.no_grad():
                        logits = self.api.forward_answer(self.eng, record, keep_all=True).float()
                        probability = self.F.softmax(logits, dim=-1)
                        top_probability, top_index = probability.topk(min(64, probability.shape[-1]), dim=-1)
                        rest = (1 - top_probability.sum(-1)).clamp_min(0)
                    cached = dict(cached, answer=answer,
                                  ground_truth_answer=str(pair[side + "_answer"]),
                                  teacher_top_idx=top_index.cpu(),
                                  teacher_top_prob=top_probability.cpu(), teacher_rest=rest.cpu(),
                                  teacher_answer=answer, teacher_bucket_k=int(top_index.shape[-1]),
                                  teacher_source="verified full native Qwen3-VL greedy response plus EOS")
                    self.cache.store(cache_path, cached)
                    del record, logits, probability, top_probability, top_index, rest
                completed.append(dict(cache=key, answer=answer, reused=reusable))
                atomic_json(path, dict(complete=False, records=completed,
                                       n_full_teacher_forwards=sum(not row["reused"] for row in completed),
                                       wall_seconds=time.time() - t0))
                if len(completed) % 8 == 0:
                    print(f"[teacher KL] {len(completed)} image/answer records prepared", flush=True)
        result = dict(complete=True, records=completed,
                      n_full_teacher_forwards=sum(not row["reused"] for row in completed),
                      wall_seconds=time.time() - t0)
        # No control may silently mix hard-only and soft targets after opting
        # into the distillation protocol, including a partly resumed backfill.
        for pair in self.plan["fit"]:
            for side in ("original", "edited"):
                cached = self.cache(pair["cache_" + side])
                expected = str(pair.get(side + "_teacher_prediction", pair[side + "_answer"]))
                if cached.get("teacher_answer") != expected or not all(
                        key in cached for key in ("teacher_top_idx", "teacher_top_prob", "teacher_rest")):
                    raise RuntimeError("incomplete full-teacher KL cache coverage")
        atomic_json(path, result)
        return result

    def train_loss(self, record, negative_answer=None):
        rest = record.get("teacher_rest", record.get("teacher_rest_prob"))
        teacher_available = all(value is not None for value in
                                (record.get("teacher_top_idx"), record.get("teacher_top_prob"), rest))
        if self.args.prepare_teacher_kl and not teacher_available:
            raise RuntimeError("--prepare-teacher-kl requires soft targets for every training image")
        if teacher_available and record.get("teacher_answer", record["answer"]) != record["answer"]:
            raise ValueError("cached soft targets were prepared on a different answer prefix")
        logits = self.api.forward_answer(self.eng, record, self.pool).float()
        targets = self.api.answer_token_ids(self.eng, record["answer"], self.device)
        ce = self.F.cross_entropy(logits, targets)
        logq = self.F.log_softmax(logits, dim=-1)
        loss = ce
        if negative_answer is not None and self.args.relation_weight > 0:
            negative_logits = self.api.forward_answer(
                self.eng, record, self.pool, answer=negative_answer).float()
            negative_targets = self.api.answer_token_ids(self.eng, negative_answer, self.device)
            positive_logp = logq.gather(1, targets[:, None]).sum()
            negative_logp = self.F.log_softmax(negative_logits, dim=-1).gather(
                1, negative_targets[:, None]).sum()
            loss = loss + self.args.relation_weight * self.F.softplus(
                self.args.relation_margin - positive_logp + negative_logp)
            self.objectives.add("opposing_answer_sequence_margin")
            self.forward_counts["training_negative_answer"] += 1
        if not teacher_available or self.args.kl_weight == 0:
            self.objectives.add("hard_teacher_answer_ce_plus_eos")
            return loss
        indices = record["teacher_top_idx"].long()
        probabilities = record["teacher_top_prob"].float()
        other = self.torch.as_tensor(rest, device=self.device).float().reshape(-1)
        if indices.shape != probabilities.shape or indices.shape[0] != logits.shape[0]:
            raise ValueError("cached teacher bucket shape does not match answer+EOS logits")
        if other.numel() != logits.shape[0]:
            raise ValueError("cached teacher remainder does not match answer+EOS logits")
        mass = probabilities.sum(-1) + other
        if not self.torch.allclose(mass, self.torch.ones_like(mass), atol=2e-3, rtol=0):
            raise ValueError("teacher top buckets and remainder must sum to one")
        selected = logq.gather(-1, indices)
        remainder = (1 - selected.exp().sum(-1)).clamp_min(1e-7).log()
        p = probabilities.clamp_min(1e-12)
        r = other.clamp_min(1e-12)
        kl = ((probabilities * (p.log() - selected)).sum(-1)
              + other * (r.log() - remainder)).mean()
        self.objectives.add("hard_teacher_answer_ce_plus_eos_and_cached_bucket_kl")
        return loss + self.args.kl_weight * kl

    def checkpoint(self, path, step, arm, history):
        self.torch.save(dict(compressor={key: value.detach().cpu()
                                        for key, value in self.pool.state_dict().items()},
                             initial_sha256=self.initial_hash, step=step, arm=arm,
                             config=vars(self.args), history=history), path)

    def frozen_ok(self):
        return all(not parameter.requires_grad and parameter._version == self.versions[name]
                   for name, parameter in self.eng.model.named_parameters())

    def decoder_sample_checksum(self):
        """Supplement version checks with fixed sampled values from every tensor."""
        digest = hashlib.sha256()
        with self.torch.no_grad():
            for name, parameter in self.eng.model.named_parameters():
                flat = parameter.detach().reshape(-1)
                if flat.numel() == 0:
                    continue
                count = min(16, flat.numel())
                # Float32 linspace rounds the last index out of bounds for
                # large embedding tensors. Construct exact integer indices.
                indices = self.torch.tensor(
                    [i * (flat.numel() - 1) // max(1, count - 1)
                     for i in range(count)], device=flat.device, dtype=self.torch.long)
                sample = flat[indices].float().cpu().numpy()
                digest.update(name.encode("utf-8"))
                digest.update(sample.tobytes())
        return digest.hexdigest()

    def identity_audit(self, pair):
        """Check stream identity and teacher-forcing's first prediction natively.

        bf16 GEMMs can differ by shape when the answer prefix is appended.
        Compare argmax and record the full absolute difference as well as the
        exact zero-gate stream identity; the latter must be bitwise exact.
        """
        self.api.set_decoder_checkpointing(self.eng, False)
        record = self.moved(pair["cache_original"], pair["original_answer"])
        with self.torch.no_grad():
            V, DS, keep = self.api.selected_streams(record, self.pool)
            main_equal = bool(self.torch.equal(V, record["V"][keep]))
            ds_equal = all(self.torch.equal(value, original[keep])
                           for value, original in zip(DS or [], record.get("DS") or []))
            prep = {key: record[key] for key in
                    ("ids", "gthw", "img_start", "img_end", "n_vis")}
            state = self.eng.prefill(prep, record["V"], record.get("DS"), keep,
                                     V_sel=V, DS_sel=DS)
            native = state.logits.reshape(-1).float()
            forced = self.api.forward_answer(self.eng, record, self.pool)[0].float()
            result = dict(pair_id=str(pair["pair_id"]), zero_gate_main_bitwise=main_equal,
                          zero_gate_deepstack_bitwise=ds_equal,
                          first_logit_max_absolute_delta=float((native - forced).abs().max()),
                          first_logit_mean_absolute_delta=float((native - forced).abs().mean()),
                          first_argmax_match=int(native.argmax()) == int(forced.argmax()),
                          native_first_token=int(native.argmax()), forced_first_token=int(forced.argmax()),
                          frozen_decoder_sample_sha256=self.initial_decoder_sample)
        if not main_equal or not ds_equal or not result["first_argmax_match"]:
            raise RuntimeError(f"native zero-gate/teacher-forcing identity audit failed: {result}")
        return result

    def prediction(self, path, answer):
        record = self.moved(path, answer)
        prediction = self.api.generate_answer(self.eng, record, self.pool,
                                              max_new_tokens=self.args.max_new_tokens)
        self.forward_counts["evaluation_generation"] += 1
        return prediction

    def evaluate_pairs(self, records, path, check_identity=False):
        self.api.set_decoder_checkpointing(self.eng, False)
        self.pool.eval()
        rows = []
        for pair in records:
            row = dict(pair_id=str(pair["pair_id"]), split=split_name(pair), group_id=group_id(pair))
            for side in ("original", "edited"):
                answer = pair[side + "_answer"]
                prediction = self.prediction(pair["cache_" + side], answer)
                row[side + "_prediction"] = prediction
                row[side + "_answer"] = answer
                row[side + "_correct"] = correct(prediction, answer)
                if check_identity and side + "_student_prediction" in pair:
                    cached = pair[side + "_student_prediction"]
                    row[side + "_cached_prediction"] = cached
                    if normalize(cached) != normalize(prediction):
                        raise RuntimeError(f"zero-gate B2 generation identity failed: {pair['pair_id']} {side}")
            row["both_correct"] = row["original_correct"] and row["edited_correct"]
            row["collapse"] = normalize(row["original_prediction"]) == normalize(row["edited_prediction"])
            rows.append(row)
            atomic_json(path, dict(summary=pair_summary(rows), records=rows, complete=False))
        result = dict(summary=pair_summary(rows), records=rows, complete=True,
                      splits={split: pair_summary([row for row in rows if row["split"] == split])
                              for split in sorted({row["split"] for row in rows})})
        atomic_json(path, result)
        return result

    def evaluate_sources(self, records, path):
        self.api.set_decoder_checkpointing(self.eng, False)
        rows = []
        for source in records:
            prediction = self.prediction(source["cache"], source["answer"])
            rows.append(dict(pair_id=str(source["pair_id"]), split=split_name(source),
                             answer=source["answer"], prediction=prediction,
                             correct=correct(prediction, source["answer"]),
                             teacher_correct=source.get("teacher_correct")))
            atomic_json(path, dict(records=rows, complete=False))
        n = len(rows)
        result = dict(records=rows, n_sources=n,
                      accuracy=sum(row["correct"] for row in rows) / n if n else None,
                      complete=True)
        atomic_json(path, result)
        return result

    def mine(self, pool, step, path):
        self.api.set_decoder_checkpointing(self.eng, False)
        rows = []
        for pair in pool:
            row = dict(pair_id=str(pair["pair_id"]), step=step)
            if self.args.mine_score == "greedy":
                for side in ("original", "edited"):
                    record = self.moved(pair["cache_" + side], pair[side + "_answer"])
                    prediction = self.api.generate_answer(self.eng, record, self.pool,
                                                          max_new_tokens=self.args.max_new_tokens)
                    row[side + "_prediction"] = prediction
                    row[side + "_correct"] = correct(prediction, pair[side + "_answer"])
                    self.forward_counts["mining_answer"] += 1
                row["score"] = int(row["original_correct"]) + int(row["edited_correct"])
                row["currently_blind"] = row["score"] < 2
                row["strict_collapse"] = (row["currently_blind"] and
                                           normalize(row["original_prediction"]) ==
                                           normalize(row["edited_prediction"]))
            else:
                margins = []
                for side, opposite in (("original", "edited"), ("edited", "original")):
                    record = self.moved(pair["cache_" + side], pair[side + "_answer"])
                    logp = self.api.candidate_logprobs(
                        self.eng, record, [pair[side + "_answer"], pair[opposite + "_answer"]], self.pool)
                    margins.append(float((logp[0] - logp[1]).item()))
                    self.forward_counts["mining_answer"] += 2
                row.update(score=sum(margins), margins=margins,
                           score_type="current_pair_answer_log_likelihood_margin_proxy")
            rows.append(row)
            append_json(path, row)
        rng = random.Random(self.args.seed + step)
        rng.shuffle(rows)
        if self.args.mine_score == "greedy":
            rows.sort(key=lambda row: (not row["strict_collapse"], row["score"]))
            failed = [row for row in rows if row["currently_blind"]]
            rows = failed if failed else rows
        else:
            rows.sort(key=lambda row: row["score"])
        chosen = rows[:self.args.pairs]
        return [self.by_id[row["pair_id"]] for row in chosen]

    def train(self, arm, directory, optimizer, fixed=None, replay=None):
        directory.mkdir(parents=True, exist_ok=True)
        # Fail rather than append a second run to an old trajectory.
        for filename in ("history.jsonl", "mining.jsonl"):
            if (directory / filename).exists():
                raise FileExistsError(f"existing trajectory: {directory / filename}; choose a fresh --out")
        history, schedule, losses = [], [], []
        counts_before = self.forward_counts.copy()
        rng = random.Random(self.args.seed)
        active = fixed or self.plan["fit"]
        t0 = time.time()
        for step in range(self.args.steps):
            if replay is not None:
                exposures = replay[step]
            else:
                if arm == "online" and step > 0 and step % self.args.round_steps == 0:
                    active = self.mine(self.plan["fit"], step, directory / "mining.jsonl")
                if arm == "ordinary":
                    selected = [rng.choice(self.plan["fit"]), rng.choice(self.plan["fit"])]
                    exposures = [dict(pair_id=str(pair["pair_id"]), side="original") for pair in selected]
                else:
                    pair = rng.choice(active)
                    exposures = [dict(pair_id=str(pair["pair_id"]), side=side)
                                 for side in ("original", "edited")]
            schedule.append(exposures)
            self.api.set_decoder_checkpointing(self.eng, self.args.checkpointing)
            self.pool.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for exposure in exposures:
                pair = self.by_id[exposure["pair_id"]]
                side = exposure["side"]
                target = pair.get(side + "_teacher_prediction", pair[side + "_answer"])
                record = self.moved(pair["cache_" + side], target)
                # VLMEvalKit's vlm/__init__.py globally disables autograd.
                # Explicitly enable it for every trainable compressor forward.
                with self.torch.enable_grad():
                    opposite = "edited" if side == "original" else "original"
                    negative = (pair.get(opposite + "_teacher_prediction", pair[opposite + "_answer"])
                                if arm not in ("ordinary", "replay_ce") else None)
                    loss = self.train_loss(record, negative) / len(exposures)
                    if not loss.requires_grad:
                        raise RuntimeError("compressor loss lost its autograd graph")
                    if not bool(self.torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite answer training loss")
                    loss.backward()
                step_loss += float(loss.detach())
                self.forward_counts["training_answer"] += 1
                del loss, record
            norm = self.torch.nn.utils.clip_grad_norm_(self.pool.parameters(), 1.0)
            if not bool(self.torch.isfinite(norm)):
                raise FloatingPointError("nonfinite compressor gradients")
            gate_gradient = float(self.pool.alpha.grad.detach()) if self.pool.alpha.grad is not None else None
            optimizer.step()
            entry = dict(step=step + 1, loss=step_loss, gradient_norm=float(norm),
                         gate_gradient=gate_gradient, gate=float(self.pool.gate.detach()),
                         exposures=exposures, elapsed_seconds=time.time() - t0)
            append_json(directory / "history.jsonl", entry)
            losses.append(step_loss)
            history.append(entry)
            if (step + 1) % self.args.round_steps == 0 or step + 1 == self.args.steps:
                self.checkpoint(directory / "checkpoint.pt", step + 1, arm, history)
                atomic_json(directory / "training.json", dict(
                    arm=arm, steps=step + 1, schedule=schedule, initial_sha256=self.initial_hash,
                    objectives=sorted(obj for obj in self.objectives
                                      if obj != "opposing_answer_sequence_margin"
                                      or (self.args.relation_weight > 0 and arm not in ("ordinary", "replay_ce"))),
                    contrastive_active=self.args.relation_weight > 0 and arm not in ("ordinary", "replay_ce"),
                    frozen_decoder_unchanged=self.frozen_ok(),
                    forward_counts={key: value - counts_before[key]
                                    for key, value in self.forward_counts.items()},
                    gate=float(self.pool.gate.detach()),
                    mean_loss=sum(losses) / len(losses), wall_seconds=time.time() - t0))
                print(f"[{arm}] {step + 1}/{self.args.steps} loss={step_loss:.4f} "
                      f"gate={float(self.pool.gate.detach()):.5f}", flush=True)
        if not self.frozen_ok():
            raise RuntimeError("frozen decoder parameters changed")
        if self.decoder_sample_checksum() != self.initial_decoder_sample:
            raise RuntimeError("frozen decoder sampled values changed")
        if not any(entry["gradient_norm"] > 0 for entry in history):
            raise RuntimeError("all compressor training gradients were zero")
        return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining", default=str(DEFAULT_OUT / "mining.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT / "training"))
    parser.add_argument("--stage", choices=("gate", "compare"), default="gate")
    parser.add_argument("--steps", type=int, default=None,
                        help="fixed optimizer updates (default: gate=120, compare=160)")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--pool-pairs", type=int, default=32)
    parser.add_argument("--round-steps", type=int, default=20)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--eval-sources", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--carrier", choices=("pool", "readout", "local_readout"), default="pool",
                        help="generic trainable carrier, not the method contribution")
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--relation-margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--gate-lr", type=float, default=.01)
    parser.add_argument("--kl-weight", type=float, default=1.)
    parser.add_argument("--prepare-teacher-kl", action="store_true",
                        help="cache full-teacher top-64+other answer distributions for every fit image")
    parser.add_argument("--mine-score", choices=("greedy", "margin"), default="greedy")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 120 if args.stage == "gate" else 160
    for name in ("steps", "pairs", "pool_pairs", "round_steps", "max_new_tokens", "rank"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (args.eval_pairs < 0 or args.eval_sources < 0 or args.lr <= 0 or args.gate_lr <= 0
            or args.kl_weight < 0 or args.relation_weight < 0):
        parser.error("evaluation counts/KL weight must be nonnegative and learning rates positive")
    mining = json.loads(Path(args.mining).read_text(encoding="utf-8"))
    if not args.dry_run and mining.get("stage") != "complete":
        raise RuntimeError("training requires mining.json stage='complete'; screening/source evaluation is still running")
    plan = make_plan(mining, args)
    public = public_plan(plan, args)
    print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"{args.stage}_plan.json", public)
    if not plan["fit"] or not plan["gate"]:
        atomic_json(out / "gate_result.json", dict(verdict="NO_FIT_BLIND_PAIRS", plan=public,
                                                  reason="no available teacher-valid fit compression failures"))
        return
    if args.stage == "compare":
        gate_path = out / "gate_result.json"
        if not gate_path.exists() or json.loads(gate_path.read_text()).get("verdict") != "PASS":
            raise RuntimeError("comparison requires a completed PASS gate_result.json at --out")
    experiment = Experiment(args, plan)
    if args.prepare_teacher_kl:
        experiment.prepare_teacher_kl(out / f"{args.stage}_teacher_kl.json")
    identity = experiment.identity_audit(plan["gate"][0])
    atomic_json(out / f"{args.stage}_identity.json", identity)
    baseline = experiment.evaluate_pairs(plan["heldout"], out / f"{args.stage}_heldout_before.json",
                                         check_identity=True)
    source_baseline = experiment.evaluate_sources(plan["sources"], out / f"{args.stage}_sources_before.json")
    if args.stage == "gate":
        before = experiment.evaluate_pairs(plan["gate"], out / "gate_fit_before.json", check_identity=True)
        optimizer = experiment.reset()
        experiment.train("gate", out / "gate", optimizer, fixed=plan["gate"])
        after = experiment.evaluate_pairs(plan["gate"], out / "gate_fit_after.json")
        heldout = experiment.evaluate_pairs(plan["heldout"], out / "gate_heldout_after.json")
        sources = experiment.evaluate_sources(plan["sources"], out / "gate_sources_after.json")
        gain = after["summary"]["both_correct"] - before["summary"]["both_correct"]
        result = dict(verdict="PASS" if gain > 0 else "UNRESOLVED_CARRIER_OR_OPTIMIZATION",
                      carrier=args.carrier, trainable_parameters=experiment.trainable_parameters,
                      criterion="at least one additional training pair both-correct in actual generation",
                      requested_pairs=args.pairs, actual_pairs=len(plan["gate"]),
                      fit_selection_strict_collapses=sum(bool(pair.get("collapse")) for pair in plan["gate"]),
                      fit_selection_broad_only_failures=sum(not bool(pair.get("collapse")) for pair in plan["gate"]),
                      train_both_correct_gain=gain, before=before["summary"], after=after["summary"],
                      fit_collapse_changes=collapse_changes(before["records"], after["records"]),
                      heldout=heldout["summary"], sources=sources,
                      heldout_change=paired_bootstrap(baseline["records"], heldout["records"], args.seed),
                      heldout_collapse_changes=collapse_changes(baseline["records"], heldout["records"]),
                      source_baseline=source_baseline, initial_sha256=experiment.initial_hash,
                      native_identity=identity,
                      decoder_sample_checksum_unchanged=(experiment.decoder_sample_checksum()
                                                         == experiment.initial_decoder_sample),
                      frozen_decoder_unchanged=experiment.frozen_ok(),
                      limitation="overfit gate only; neither generalization nor mining advantage established")
        atomic_json(out / "gate_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    results, online_schedule = {}, None
    # Both hard arms begin with exactly the same current-model search result.
    # Its screening cost is incurred once and explicitly shared; only online
    # refreshes it after training rounds.  No held-out examples enter here.
    experiment.reset()
    initial_hard = experiment.mine(plan["fit"], 0, out / "comparison_initial_mining.jsonl")
    atomic_json(out / "comparison_initial_hard.json", dict(
        pair_ids=[pair["pair_id"] for pair in initial_hard],
        candidate_pairs=len(plan["fit"]), shared_by=["fixed", "online"],
        search_step=0, extra_online_search_steps=list(range(args.round_steps, args.steps, args.round_steps))))
    arms = ["ordinary", "random", "fixed", "online", "replay"]
    if args.relation_weight > 0:
        arms.append("replay_ce")
    for arm in arms:
        optimizer = experiment.reset()
        fixed = initial_hard if arm in ("fixed", "online") else None
        replay = None
        if arm in ("replay", "replay_ce"):
            # Shuffle complete exposures, not pair updates: this exactly keeps
            # the multiset while removing online ordering and paired adjacency.
            flat = [dict(exposure) for update in online_schedule for exposure in update]
            random.Random(args.seed + 3).shuffle(flat)
            replay = [flat[index:index + 2] for index in range(0, len(flat), 2)]
        schedule = experiment.train(arm, out / arm, optimizer, fixed=fixed, replay=replay)
        if arm == "online":
            online_schedule = schedule
        heldout = experiment.evaluate_pairs(plan["heldout"], out / arm / "heldout.json")
        sources = experiment.evaluate_sources(plan["sources"], out / arm / "sources.json")
        results[arm] = dict(heldout=heldout, sources=sources,
                            heldout_change=paired_bootstrap(baseline["records"], heldout["records"], args.seed),
                            heldout_collapse_changes=collapse_changes(baseline["records"], heldout["records"]),
                            initial_sha256=experiment.initial_hash)
        contrasts = {}
        if "online" in results:
            for reference in ("ordinary", "random", "fixed", "replay", "replay_ce"):
                if reference in results:
                    contrasts["online_minus_" + reference] = paired_bootstrap(
                        results[reference]["heldout"]["records"], results["online"]["heldout"]["records"], args.seed)
        atomic_json(out / "comparison.json", dict(
            complete=arm == arms[-1], arms=results, paired_contrasts=contrasts,
            carrier=args.carrier, trainable_parameters=experiment.trainable_parameters,
            config=vars(args), objectives=sorted(experiment.objectives),
            frozen_decoder_unchanged=experiment.frozen_ok(),
            decoder_sample_checksum_unchanged=(experiment.decoder_sample_checksum()
                                               == experiment.initial_decoder_sample),
            native_identity=identity,
            forward_counts=experiment.forward_counts, baseline=baseline,
            source_baseline=source_baseline,
            interpretation="random/fixed/online/replay share objective; replay_ce uses identical data with CE+KL only; negative-answer forwards reported separately"))


if __name__ == "__main__":
    main()
