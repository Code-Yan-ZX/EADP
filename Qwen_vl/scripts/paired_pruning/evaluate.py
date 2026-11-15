#!/usr/bin/env python3
"""Frozen, resumable evaluation of both paired-pruning experiments.

No answer enters prepare(), the scorer, or generation. Checkpoints and input
bytes are frozen before the test manifest is decoded. Timing uses fresh image
preparation per invocation, independently of the cached-feature QA evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
import traceback

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.paired_pruning"

HERE = Path(__file__).resolve().parent
BASELINES = ("full", "b1", "b2", "rres512", "random")
QUERY_ARMS = ("bag_rank", "ordered_rank", "ordered_pair")
SETS_ARMS = ("canonical", "token_marginal", "multimode")
LEARNED_ARMS = QUERY_ARMS + SETS_ARMS
SCHEMA = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def official_anls(prediction, answers):
    """Independent copy of the repository's official DocVQA ANLS convention.

    Edit distance normalizes whitespace and case; its denominator deliberately
    uses the ORIGINAL upper-cased string lengths, matching VLMEvalKit. Exactly
    0.5 similarity passes. This function does not import a model or GPU library.
    """
    if not isinstance(prediction, str) or not isinstance(answers, list) or not answers:
        raise ValueError("ANLS requires a string prediction and nonempty answer aliases")
    similarities = []
    for answer in answers:
        if not isinstance(answer, str):
            raise ValueError("answer aliases must be strings")
        left, right = (" ".join(s.strip().lower().split()) for s in (answer, prediction))
        if len(left) > len(right):
            left, right = right, left
        previous = list(range(len(left) + 1))
        for j, b in enumerate(right, 1):
            current = [j]
            for i, a in enumerate(left, 1):
                current.append(min(current[-1] + 1, previous[i] + 1,
                                   previous[i - 1] + (a != b)))
            previous = current
        denominator = max(len(answer.upper()), len(prediction.upper()))
        similarity = 1.0 - (previous[-1] / denominator if denominator else 0.0)
        similarities.append(similarity if similarity >= 0.5 else 0.0)
    return max(similarities)


def strict_match(prediction, answers):
    """Literal alias equality: no case, punctuation, whitespace or numeric fixes."""
    return float(any(prediction == answer for answer in answers))


def arm_specs(protocol):
    seeds = protocol.get("seeds", [11, 23, 37])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("protocol seeds must be distinct and nonempty")
    return ([{"name": name, "arm": name, "seed": None} for name in BASELINES]
            + [{"name": f"{arm}__seed_{seed}", "arm": arm, "seed": int(seed)}
               for arm in LEARNED_ARMS for seed in seeds])


def read_protocol(run_dir):
    from .artifacts import load_protocol
    protocol = load_protocol(run_dir)
    if (protocol.get("n", 1024), protocol.get("k", 256)) != (1024, 256):
        raise ValueError("this preregistered evaluator requires N1024 -> K256")
    if protocol.get("max_new_tokens", 128) != 128:
        raise ValueError("QA generation cap must remain 128")
    return protocol


def check_training_summary(path, arms, seeds, protocol_hash):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "complete":
        raise ValueError(f"training is not complete: {path}")
    if set(data.get("arms", [])) != set(arms) or set(data.get("seeds", [])) != set(seeds):
        raise ValueError(f"training arms/seeds differ from protocol: {path}")
    if data.get("protocol_sha256") != protocol_hash:
        raise ValueError(f"training used a different protocol: {path}")
    rows = data.get("checkpoints", [])
    pairs = [(r["arm"], int(r["seed"])) for r in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != {(a, s) for a in arms for s in seeds}:
        raise ValueError(f"incomplete/duplicate checkpoint roster: {path}")
    return data


def require_gate(path, protocol_hash):
    gate = json.loads(Path(path).read_text())
    if gate.get("status") != "PASS" or gate.get("protocol_sha256") != protocol_hash:
        raise ValueError("native GPU gate must PASS for this exact protocol")
    return gate


def freeze_inputs(run_dir, timing_blocks=5, timing_warmup=3, gate_path=None):
    """Hash manifest bytes without parsing test answers, then freeze the roster."""
    run_dir = Path(run_dir).resolve()
    protocol = read_protocol(run_dir)
    protocol_hash = sha256(run_dir / "protocol.json")
    seeds = [int(s) for s in protocol.get("seeds", [11, 23, 37])]
    if timing_blocks != protocol.get("timing_blocks", 5) or timing_warmup != protocol.get("timing_warmup", 3):
        raise ValueError("timing arguments must match the frozen protocol")
    gate_path = Path(gate_path or run_dir / "gates/gpu.json").resolve()
    require_gate(gate_path, protocol_hash)
    files = {run_dir / "protocol.json", run_dir / "data/manifest.json", gate_path}
    files.update(Path(path) for path in protocol.get("model_metadata_sha256", {}))
    cache_index_path = run_dir / "cache/index.json"
    cache_index = json.loads(cache_index_path.read_text())
    if not cache_index.get("complete") or cache_index.get("protocol_sha256") != protocol_hash:
        raise ValueError("training cache index is incomplete or belongs to another protocol")
    cache_hash = sha256(cache_index_path)
    files.add(cache_index_path)
    for experiment, arms in (("query", QUERY_ARMS), ("sets", SETS_ARMS)):
        summary_path = run_dir / f"checkpoints/{experiment}_summary.json"
        summary = check_training_summary(summary_path, arms, seeds, protocol_hash)
        if summary.get("cache_index_sha256") != cache_hash:
            raise ValueError("training summaries must use the same frozen cache index")
        files.add(summary_path)
        training_freeze = run_dir / f"checkpoints/{experiment}_train_freeze.json"
        if training_freeze.is_file():
            files.add(training_freeze)
        for row in summary["checkpoints"]:
            checkpoint = run_dir / f"checkpoints/{row['arm']}__seed_{row['seed']}.pt"
            if Path(row["path"]).resolve() != checkpoint.resolve():
                raise ValueError("training summary checkpoint path differs from canonical arm/seed path")
            if sha256(checkpoint) != row["sha256"]:
                raise ValueError(f"checkpoint does not match training summary: {checkpoint}")
            files.add(checkpoint)
    diagnostic_path = run_dir / "diagnostics/mechanisms.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("protocol_sha256") != protocol_hash or diagnostic.get("cache_index_sha256") != cache_hash:
            raise ValueError("mechanism diagnostics used different inputs")
        files.add(diagnostic_path)
    # The runtime may enumerate native model, prompt and selector dependencies.
    # Importing this module must not construct a model or touch CUDA.
    from . import runtime
    sources = set(HERE.glob("*.py"))
    if hasattr(runtime, "source_paths"):
        sources.update(Path(path) for path in runtime.source_paths())
    files.update(sources)
    frozen = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "files": {str(path.resolve()): sha256(path) for path in sorted(files)},
        "local_sources": sorted(str(path.resolve()) for path in HERE.glob("*.py")),
        "arms": arm_specs(protocol),
        "protocol_sha256": protocol_hash,
        "manifest_sha256": sha256(run_dir / "data/manifest.json"),
        "gate_path": str(gate_path),
        "timing": {"warmup": timing_warmup, "blocks": timing_blocks,
                   "caps": [1, 16], "scope": "fresh prepare+encode+select+generate; model load excluded",
                   "fixed_16_note": "timing only: ignore EOS and generate exactly 16 tokens; QA uses native EOS"},
        "test_policy": "all original test questions; all frozen arms/seeds; no oracle selection",
    }
    frozen["freeze_sha256"] = json_hash(frozen)
    path = run_dir / "evaluation/freeze.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise ValueError("frozen inputs, source, checkpoint roster or timing changed; refuse resume")
        return old
    if (run_dir / "evaluation/predictions.sqlite3").exists():
        raise ValueError("predictions exist without freeze; refuse to adopt them")
    frozen["created_utc"] = utc_now()
    atomic_json(path, frozen)
    return frozen


def verify_freeze(run_dir):
    path = Path(run_dir).resolve() / "evaluation/freeze.json"
    frozen = json.loads(path.read_text())
    identity = {k: v for k, v in frozen.items() if k not in ("freeze_sha256", "created_utc")}
    if json_hash(identity) != frozen["freeze_sha256"]:
        raise ValueError("freeze metadata was modified")
    for filename, expected in frozen["files"].items():
        if not Path(filename).is_file() or sha256(filename) != expected:
            raise ValueError(f"frozen file changed or missing: {filename}")
    if sorted(str(p.resolve()) for p in HERE.glob("*.py")) != frozen["local_sources"]:
        raise ValueError("source file roster changed since freeze")
    return frozen


def load_manifest(run_dir):
    """Only call after freeze_inputs; validate full denominators and split isolation."""
    manifest = json.loads((Path(run_dir) / "data/manifest.json").read_text())
    seen_items, seen_parents, seen_pixels, seen_pages = set(), {}, {}, {}
    for split in ("fit", "val", "test", "dev"):
        pages = manifest["splits"][split]
        if not pages:
            raise ValueError(f"empty required split: {split}")
        for page in pages:
            for name in ("page_id", "parent_id", "image_path", "image_sha256", "pixel_sha256", "questions"):
                if name not in page:
                    raise ValueError(f"missing page field {name}")
            for key, lookup in (("parent_id", seen_parents), ("pixel_sha256", seen_pixels),
                                ("page_id", seen_pages)):
                value = str(page[key])
                if value in lookup and lookup[value] != split:
                    raise ValueError(f"cross-split leakage in {key}: {value}")
                lookup[value] = split
            if not page["questions"]:
                raise ValueError("empty question roster")
            for question in page["questions"]:
                item_id = str(question["item_id"])
                if item_id in seen_items:
                    raise ValueError(f"duplicate item_id: {item_id}")
                seen_items.add(item_id)
                if not isinstance(question["question"], str) or not question["question"].strip():
                    raise ValueError("empty/non-string question")
                if not isinstance(question["answers"], list) or not question["answers"]:
                    raise ValueError("every original question requires nonempty answer aliases")
                if not all(isinstance(a, str) for a in question["answers"]):
                    raise ValueError("answer aliases must be strings")
                if not isinstance(question.get("numeric"), bool):
                    raise ValueError("numeric subgroup must be fixed in the manifest")
    return manifest


def items_for_split(manifest, split="test"):
    return [(page, q) for page in manifest["splits"][split] for q in page["questions"]]


def public_page(page):
    return {k: page[k] for k in ("page_id", "parent_id", "image_path", "image_sha256", "pixel_sha256")}


def validate_keep_indices(indices, expected_count, source_count):
    if not isinstance(indices, list) or len(indices) != expected_count:
        raise ValueError("actual keep_idx length does not match the physical token budget")
    if any(type(index) is not int or not 0 <= index < source_count for index in indices):
        raise ValueError("keep_idx contains invalid original indices")
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("keep_idx must contain unique indices in original order")


class ResultStore:
    """One committed SQLite row per question/arm, with immutable replay identity."""
    def __init__(self, path, freeze_hash):
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                item_id TEXT NOT NULL, arm TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (item_id, arm));
            CREATE TABLE IF NOT EXISTS timing (
                record_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, timestamp TEXT, payload TEXT NOT NULL);
        """)
        old = self.connection.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()
        if old is not None and old[0] != freeze_hash:
            self.connection.close()
            raise ValueError("result database belongs to a different freeze")
        if old is None and self.connection.execute("SELECT count(*) FROM predictions").fetchone()[0]:
            self.connection.close()
            raise ValueError("existing predictions lack freeze identity; refuse to adopt")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('freeze_sha256', ?)", (freeze_hash,))
        self.connection.commit()

    def get(self, item_id, arm):
        row = self.connection.execute("SELECT payload FROM predictions WHERE item_id=? AND arm=?",
                                      (str(item_id), arm)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, record):
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        old = self.get(record["item_id"], record["arm"])
        if old is not None:
            if old != record:
                raise ValueError("refuse to replace an existing prediction")
            return
        self.connection.execute("INSERT INTO predictions VALUES (?, ?, ?)",
                                (str(record["item_id"]), record["arm"], serialized))
        self.connection.commit()

    def timing_exists(self, key):
        return self.connection.execute("SELECT 1 FROM timing WHERE record_key=?", (key,)).fetchone() is not None

    def save_timing(self, key, record):
        self.connection.execute("INSERT INTO timing VALUES (?, ?)",
                                (key, json.dumps(record, ensure_ascii=False, allow_nan=False)))
        self.connection.commit()

    def event(self, value):
        self.connection.execute("INSERT INTO events(timestamp, payload) VALUES (?, ?)",
                                (utc_now(), json.dumps(value, ensure_ascii=False)))
        self.connection.commit()

    def close(self):
        self.connection.close()


def _matrix(tensor, name):
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have [tokens, hidden] shape")
    return tensor


def choose_keep(engine, ctx, spec, models, k):
    import torch
    from . import runtime
    if spec["arm"] in ("full", "rres512"):
        return None
    if spec["arm"] in BASELINES:
        return runtime.baseline_keep(engine, ctx, spec["arm"], k=k)
    from .models import select_keep
    model = models[spec["name"]]
    parameter = next(model.parameters())
    streams = [_matrix(ctx["V"], "V")] + [_matrix(t, "DS") for t in ctx["DS"]]
    if len(streams) != 4 or len({tuple(t.shape) for t in streams}) != 1:
        raise ValueError("student requires matching V and all three native DeepStack streams")
    # Match offline cache serialization even if an encoder implementation
    # happens to return a wider activation dtype.
    features = torch.stack(streams, dim=0).unsqueeze(0).to(dtype=torch.bfloat16).to(
        device=parameter.device, dtype=parameter.dtype)
    question = _matrix(ctx["question_embeddings"], "question").unsqueeze(0).to(
        device=parameter.device, dtype=parameter.dtype)
    mask = torch.ones(question.shape[:2], dtype=torch.bool, device=parameter.device)
    with torch.inference_mode():
        output = model(features, question, mask)
        keep = select_keep(output, k=k)[0]
    return keep.sort().values


def generation_record(result, page, question, spec, source_count, expected_count, cap, freeze_hash):
    required = ("text", "token_ids", "truncated", "actual_visual_count", "keep_idx", "audit",
                "ds_lengths", "positions_sha256")
    if any(key not in result for key in required):
        raise ValueError(f"runtime.generate must return {required}")
    validate_keep_indices(result["keep_idx"], expected_count, source_count)
    if result["actual_visual_count"] != expected_count or type(result["truncated"]) is not bool:
        raise ValueError("physical token count/truncation audit failed")
    if (not isinstance(result["audit"], dict) or result["audit"].get("native") is not True
            or result["audit"].get("ds_streams") != 3
            or result["audit"].get("positions_preserved") is not True
            or result["ds_lengths"] != [expected_count] * 3):
        raise ValueError("missing native DeepStack/position audit")
    if not isinstance(result["token_ids"], list) or len(result["token_ids"]) > cap:
        raise ValueError("invalid generated token trace")
    if not isinstance(result["text"], str):
        raise ValueError("runtime returned non-string answer")
    score = official_anls(result["text"], question["answers"])
    return {**result, "item_id": str(question["item_id"]), "page_id": str(page["page_id"]),
            "parent_id": str(page["parent_id"]), "question_id": question.get("question_id"),
            "question": question["question"], "numeric": question["numeric"],
            "arm": spec["name"], "base_arm": spec["arm"], "seed": spec["seed"],
            "anls": score, "strict": strict_match(result["text"], question["answers"]),
            "source_visual_count": source_count, "max_new_tokens": cap,
            "freeze_sha256": freeze_hash, "saved_utc": utc_now()}


def evaluate_qa(engine, manifest, specs, models, protocol, frozen, store):
    from . import runtime
    n, k, cap = protocol.get("n", 1024), protocol.get("k", 256), protocol.get("max_new_tokens", 128)
    for page, question in items_for_split(manifest):
        pending = [s for s in specs if store.get(question["item_id"], s["name"]) is None]
        if not pending:
            continue
        if sha256(page["image_path"]) != page["image_sha256"]:
            raise ValueError(f"input image changed: {page['image_path']}")
        clean_page = public_page(page)
        ctx = None
        for spec in pending:
            try:
                if spec["arm"] == "rres512":
                    with runtime.low_resolution(engine, pixels=512):
                        low_ctx = runtime.prepare(engine, clean_page, question["question"],
                                                  max_question_tokens=protocol.get("max_question_tokens", 256))
                        count = len(_matrix(low_ctx["V"], "V"))
                        if count != k:
                            raise ValueError(f"R-res512 must produce {k} native visual tokens, got {count}")
                        result = runtime.generate(engine, low_ctx, keep=None, max_tokens=cap)
                    del low_ctx
                else:
                    if ctx is None:
                        ctx = runtime.prepare(engine, clean_page, question["question"],
                                              max_question_tokens=protocol.get("max_question_tokens", 256))
                    count = len(_matrix(ctx["V"], "V"))
                    if count != n:
                        raise ValueError(f"native high-resolution token count must be {n}, got {count}")
                    keep = choose_keep(engine, ctx, spec, models, k)
                    result = runtime.generate(engine, ctx, keep=keep, max_tokens=cap)
                expected = n if spec["arm"] == "full" else k
                record = generation_record(result, page, question, spec, count, expected, cap,
                                           frozen["freeze_sha256"])
                runtime_score = float(runtime.anls(result["text"], question["answers"]))
                if not math.isclose(record["anls"], runtime_score, abs_tol=1e-12):
                    raise ValueError("independent ANLS disagrees with runtime official scorer")
                store.save(record)
                print(json.dumps({"item_id": record["item_id"], "arm": spec["name"], "saved": True}), flush=True)
            except Exception:
                store.event({"phase": "qa", "item_id": str(question["item_id"]), "arm": spec["name"],
                             "error": traceback.format_exc()})
                raise
        del ctx


def _synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timing_key(item_id, arm, cap, block):
    return json.dumps([str(item_id), arm, int(cap), int(block)], separators=(",", ":"))


def evaluate_timing(engine, manifest, specs, models, protocol, frozen, store):
    """Paired blocks on DEV only. Every measured call encodes a fresh image."""
    from . import runtime
    timing = frozen["timing"]
    n, k = protocol.get("n", 1024), protocol.get("k", 256)
    # All dev questions and pages are fixed by the manifest, never by results.
    for page, question in items_for_split(manifest, "dev"):
        if sha256(page["image_path"]) != page["image_sha256"]:
            raise ValueError("timing image changed")
        clean_page = public_page(page)

        def one_call(spec, cap):
            manager = runtime.low_resolution(engine, pixels=512) if spec["arm"] == "rres512" else nullcontext()
            with manager:
                _synchronize()
                start = time.perf_counter()
                ctx = runtime.prepare(engine, clean_page, question["question"],
                                      max_question_tokens=protocol.get("max_question_tokens", 256))
                count = len(_matrix(ctx["V"], "V"))
                if count != (k if spec["arm"] == "rres512" else n):
                    raise ValueError("fresh timing input token count mismatch")
                keep = choose_keep(engine, ctx, spec, models, k)
                result = runtime.generate(engine, ctx, keep=keep, max_tokens=cap, ignore_eos=True)
                _synchronize()
                # Runtime captures readiness immediately after stock decoding,
                # before post-generation position/hash/CPU-copy diagnostics.
                ready = result.get("generation_ready_perf_counter", time.perf_counter())
                elapsed_ms = (ready - start) * 1000.0
                validate_keep_indices(result["keep_idx"], n if spec["arm"] == "full" else k, count)
                if len(result["token_ids"]) != cap:
                    raise ValueError("timing generation must produce its exact fixed token count")
            return result, elapsed_ms

        for cap in timing["caps"]:
            # Warmup repeats on every resumed process; not mixed into measured blocks.
            for warmup in range(timing["warmup"]):
                for spec in specs:
                    if all(store.timing_exists(timing_key(question["item_id"], spec["name"], cap, b))
                           for b in range(timing["blocks"])):
                        continue
                    one_call(spec, cap)
            for block in range(timing["blocks"]):
                order = list(specs)
                seed = int(hashlib.sha256(f"{question['item_id']}:{cap}:{block}".encode()).hexdigest()[:16], 16)
                random.Random(seed).shuffle(order)
                for position, spec in enumerate(order):
                    key = timing_key(question["item_id"], spec["name"], cap, block)
                    if store.timing_exists(key):
                        continue
                    try:
                        result, elapsed_ms = one_call(spec, cap)
                        record = {"item_id": str(question["item_id"]), "page_id": str(page["page_id"]),
                                  "parent_id": str(page["parent_id"]), "arm": spec["name"],
                                  "block": block, "order_in_block": position, "cap": cap,
                                  "metric": "fresh_ttft_ms" if cap == 1 else "fresh_16_token_total_ms",
                                  "elapsed_ms": elapsed_ms, "generated_tokens": len(result["token_ids"]),
                                  "token_ids": result["token_ids"], "truncated": result["truncated"],
                                  "keep_idx": result["keep_idx"], "actual_visual_count": result["actual_visual_count"],
                                  "ds_lengths": result["ds_lengths"], "positions_sha256": result["positions_sha256"],
                                  "audit": result["audit"], "freeze_sha256": frozen["freeze_sha256"],
                                  "saved_utc": utc_now()}
                        store.save_timing(key, record)
                        print(json.dumps({"timing": key, "elapsed_ms": elapsed_ms}), flush=True)
                    except Exception:
                        store.event({"phase": "timing", "key": key, "error": traceback.format_exc()})
                        raise


@contextmanager
def evaluation_lock(run_dir):
    directory = Path(run_dir) / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "evaluate.lock").open("w") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another evaluation process owns this run") from error
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "qa", "timing", "freeze"), default="all")
    parser.add_argument("--timing-blocks", type=int, default=5)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args(argv)
    if args.timing_blocks < 1 or args.timing_warmup < 0:
        parser.error("timing-blocks must be positive and timing-warmup nonnegative")
    with evaluation_lock(args.run_dir):
        frozen = freeze_inputs(args.run_dir, args.timing_blocks, args.timing_warmup, args.gate)
        verify_freeze(args.run_dir)
        if args.phase == "freeze":
            print(frozen["freeze_sha256"])
            return
        # Only now decode the test answers. They remain outside inference inputs.
        manifest = load_manifest(args.run_dir)
        protocol = read_protocol(args.run_dir)
        import torch
        from . import runtime
        from .models import load_checkpoint
        engine = runtime.load_engine(model_path=protocol.get("model_path"))
        device = getattr(engine, "device", "cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for spec in frozen["arms"]:
            if spec["seed"] is not None:
                model, _ = load_checkpoint(args.run_dir / f"checkpoints/{spec['name']}.pt", device=device)
                models[spec["name"]] = model.eval()
        store = ResultStore(args.run_dir / "evaluation/predictions.sqlite3", frozen["freeze_sha256"])
        try:
            if args.phase in ("all", "qa"):
                evaluate_qa(engine, manifest, frozen["arms"], models, protocol, frozen, store)
            if args.phase in ("all", "timing"):
                evaluate_timing(engine, manifest, frozen["arms"], models, protocol, frozen, store)
        finally:
            store.close()
        verify_freeze(args.run_dir)
    print("Evaluation phase finished. Run report.py to audit completeness and paired results.")


if __name__ == "__main__":
    main()
