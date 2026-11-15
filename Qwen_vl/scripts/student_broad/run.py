"""Freeze selected students, check native inference, and run transfer benchmarks.

No training, offline teacher, target checkpoint selection, or target tuning.
Existing per-item predictions survive interruptions and are never replaced.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import traceback

from scripts.paired_pruning.artifacts import atomic_json, sha256
from scripts.paired_pruning.evaluate import json_hash, validate_keep_indices

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
DEFAULT_SOURCE = QWEN / "outputs/paired_pruning/run_20261111"
DEFAULT_RUN = QWEN / "outputs/student_broad/run_20261113_v2"
DATASETS = ("RealWorldQA", "ChartQA_TEST", "InfoVQA_VAL", "AI2D_TEST", "MMStar", "HallusionBench", "MME", "POPE", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11")
ARMS = ("full", "b1", "b2", "rres512") + tuple(f"bag_rank__seed_{s}" for s in (11, 23, 37))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timing_roster(manifest, count):
    selected = {}
    for name in DATASETS:
        data = manifest["datasets"][name]
        eligible = set(data.get("decontaminated_item_ids", [i["item_id"] for i in data["items"]]))
        by_image = {}
        for item in sorted(data["items"], key=lambda i: (str(i.get("image_sha256") or ""), str(i["item_id"]))):
            if item["item_id"] in eligible and item.get("has_image", True):
                by_image.setdefault(item["image_sha256"], item)
        if len(by_image) < count:
            raise ValueError(f"Need {count} independent timing images for {name}")
        # Deterministic input-only sample, before the first target prediction.
        selected[name] = [str(i["item_id"]) for i in list(by_image.values())[:count]]
    return selected


def freeze_run(run_dir, source_run, timing_images=8, timing_blocks=3, timing_warmup=2):
    from scripts.paired_pruning.artifacts import load_protocol
    from scripts.paired_pruning.evaluate import verify_freeze
    from scripts.paired_pruning import runtime as old_runtime
    import torch
    import transformers
    from PIL import __version__ as pillow_version
    source_freeze = verify_freeze(source_run)
    original = load_protocol(source_run)
    manifest = read_json(run_dir / "manifest.json")
    from .data import verify_manifest
    verify_manifest(manifest)
    summary_path = source_run / "checkpoints/query_summary.json"
    summary = read_json(summary_path)
    if summary["status"] != "complete":
        raise ValueError("Source training must be complete")
    checkpoints = {}
    for seed in (11, 23, 37):
        arm = f"bag_rank__seed_{seed}"
        row, = [r for r in summary["checkpoints"] if r["arm"] == "bag_rank" and r["seed"] == seed]
        path = (source_run / f"checkpoints/{arm}.pt").resolve()
        if str(path) != row["path"] or sha256(path) != row["sha256"]:
            raise ValueError("Selected student checkpoint provenance differs")
        checkpoints[arm] = dict(path=str(path), sha256=row["sha256"], seed=seed,
                                best_step=row["best_step"], updates=row["updates"])
    # Validate every question without truncation before starting GPU inference.
    tokenizer = transformers.AutoTokenizer.from_pretrained(original["model_path"], local_files_only=True)
    from .runtime import PROMPT_SUFFIX, prompt_text
    lengths = {}
    for dataset in DATASETS:
        items = manifest["datasets"][dataset]["items"]
        if not items or len({str(i["item_id"]) for i in items}) != len(items):
            raise ValueError(f"Empty or duplicate target item roster: {dataset}")
        values = [len(tokenizer.encode(prompt_text(i), add_special_tokens=True)) for i in items]
        if min(values) < 1 or max(values) > 4096:
            raise ValueError(f"{dataset} question length outside selected student's 4096-token input adapter")
        lengths[dataset] = dict(count=len(values), maximum=max(values))
    protocol = dict(schema=1, source_run=str(source_run), source_freeze_sha256=source_freeze["freeze_sha256"],
        source_protocol_sha256=sha256(source_run / "protocol.json"), model_path=original["model_path"],
        model_metadata_sha256=original["model_metadata_sha256"], model_weight_stat=original["model_weight_stat"],
        arms=list(ARMS), seeds=[11, 23, 37], checkpoints=checkpoints, datasets=list(DATASETS),
        n=1024, k=256, max_new_tokens=128, max_question_tokens=4096, prompt_suffix=PROMPT_SUFFIX,
        training="none; reuse frozen selected bag_rank students; no target learned-parameter adaptation",
        checkpoint_selection="source validation objective only; report all three seeds",
        tokenization_audit=lengths, manifest_sha256=sha256(run_dir / "manifest.json"),
        timing=dict(item_ids_by_dataset=timing_roster(manifest, timing_images), images_per_dataset=timing_images,
                    blocks=timing_blocks, warmup=timing_warmup, caps=[1, 16],
                    scope="fresh image preparation+vision+selection+native generation; no image/KV cache; model load excluded",
                    stop="generation_ready_perf_counter before post-generation audit; fixed caps ignore EOS only for timing"),
        metrics="Pinned native dataset metrics; deterministic exact_matching for MCQ; grouped native scores reported separately from row accuracy",
        student_input_adapter="Unordered bag scorer length guard raised to 4096, no parameter or positional encoding changes; original short-input equivalence gate",
        interpretation="public benchmarks have prior project exposure; not untouched blind tests; base pretraining exposure unknown",
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         transformers=transformers.__version__, pillow=pillow_version,
                         cuda=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    atomic_json(run_dir / "protocol.json", protocol, frozen=True)
    paths = set(old_runtime.source_paths()) | set(HERE.glob("*.py")) | set(HERE.glob("*.sh"))
    paths.update(Path(p) for p in source_freeze["files"])
    paths.update((QWEN / "scripts/student_transfer").glob("*.py"))
    from .metrics import source_metadata
    paths.update(Path(meta["path"]) for meta in source_metadata().values())
    paths.update(Path(p) for p in manifest["provenance"]["source_files"])
    paths.update(Path(c["path"]) for c in checkpoints.values())
    paths.update((run_dir / "manifest.json", run_dir / "protocol.json"))
    paths.update(path for path in (run_dir / "pandas_metadata_audit.json",
                                  run_dir / "manifest_pre_pandas_na_policy.json") if path.exists())
    paths.update(Path(item["image_path"]) for data in manifest["datasets"].values()
                 for item in data["items"] if item.get("has_image", True))
    paths.add(ROOT / "docs/student_broad_prereg.md")
    paths.update(QWEN / "VLMEvalKit/vlmeval" / p for p in
                 ("dataset/utils/vqa_eval.py", "dataset/utils/ocrbench.py", "smp/misc.py", "dataset/utils/multiple_choice.py", "dataset/utils/yorn.py", "dataset/image_mcq.py", "dataset/image_yorn.py", "dataset/image_vqa.py"))
    frozen = dict(schema=1, source_run=str(source_run), files={str(p.resolve()): sha256(p) for p in sorted(paths)},
                  local_sources=sorted(str(p.resolve()) for p in HERE.glob("*.py")),
                  arms=list(ARMS), manifest_sha256=protocol["manifest_sha256"],
                  protocol_sha256=sha256(run_dir / "protocol.json"))
    frozen["freeze_sha256"] = json_hash(frozen)
    atomic_json(run_dir / "freeze.json", frozen, frozen=True)
    return frozen, protocol, manifest


def verify_run(run_dir):
    frozen = read_json(run_dir / "freeze.json")
    if json_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"}) != frozen["freeze_sha256"]:
        raise ValueError("Freeze identity changed")
    for path, expected in frozen["files"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen source/input/weight changed: {path}; use a new run directory")
    if sorted(str(p.resolve()) for p in HERE.glob("*.py")) != frozen["local_sources"]:
        raise ValueError("Frozen source roster changed")
    protocol = read_json(run_dir / "protocol.json")
    for path, expected in protocol["model_weight_stat"].items():
        stat = Path(path).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError(f"Frozen base model weights changed: {path}")
    return frozen


class Store:
    def __init__(self, path, freeze_hash):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(dataset TEXT,item_id TEXT,arm TEXT,payload TEXT NOT NULL,
          PRIMARY KEY(dataset,item_id,arm));
        CREATE TABLE IF NOT EXISTS timings(dataset TEXT,item_id TEXT,arm TEXT,cap INTEGER,block INTEGER,payload TEXT NOT NULL,
          PRIMARY KEY(dataset,item_id,arm,cap,block));
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,timestamp TEXT,payload TEXT NOT NULL);
        """)
        old = self.db.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()
        if old and old[0] != freeze_hash:
            raise ValueError("Database freeze differs")
        if not old and sum(self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                           for table in ("predictions", "timings")):
            raise ValueError("Existing records have no freeze identity")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('freeze_sha256',?)", (freeze_hash,))
        self.db.commit()

    def has(self, dataset, item_id, arm, cap=None, block=None):
        table, columns, values = "predictions", "dataset=? AND item_id=? AND arm=?", (dataset, str(item_id), arm)
        if cap is not None:
            table, columns, values = "timings", columns + " AND cap=? AND block=?", values + (cap, block)
        return self.db.execute(f"SELECT 1 FROM {table} WHERE {columns}", values).fetchone() is not None

    def save(self, record, timing=False):
        values = (record["dataset"], str(record["item_id"]), record["arm"])
        if self.has(*values, *( (record["cap"], record["block"]) if timing else () )):
            raise ValueError("Refuse to replace existing record")
        table = "timings" if timing else "predictions"
        if timing:
            values += (record["cap"], record["block"])
        values += (json.dumps(record, ensure_ascii=False, allow_nan=False),)
        self.db.execute(f"INSERT INTO {table} VALUES({','.join('?' for _ in values)})", values)
        self.db.commit()

    def event(self, payload):
        self.db.execute("INSERT INTO events(timestamp,payload) VALUES(?,?)",
                        (now(), json.dumps(payload, ensure_ascii=False)))
        self.db.commit()

    def count(self, table="predictions"):
        if table not in ("predictions", "timings"):
            raise ValueError(table)
        return self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def validate_native(result, arm, count, cap):
    no_image = count == 0
    expected = 0 if no_image else (1024 if arm == "full" else 256)
    if count != (0 if no_image else (256 if arm == "rres512" else 1024)):
        raise ValueError("Actual source visual budget differs")
    validate_keep_indices(result["keep_idx"], expected, count)
    if result["actual_visual_count"] != expected or result["ds_lengths"] != ([] if no_image else [expected]*3):
        raise ValueError("Actual retained visual/DeepStack budget differs")
    audit = result["audit"]
    if not (audit.get("native") and audit.get("positions_preserved") and audit.get("ds_streams") == (0 if no_image else 3)
            and audit.get("answer_in_selection") is False and audit.get("fresh_kv") is True):
        raise ValueError("Native answer-free inference audit failed")
    if not isinstance(result["truncated"], bool) or len(result["token_ids"]) > cap:
        raise ValueError("Invalid token/truncation trace")
    if no_image and not audit.get("text_only_control"):
        raise ValueError("Text-only control audit missing")


def inference(engine, item, arm, models, cap=128, ignore_eos=False, context=None):
    from . import runtime
    manager = runtime.low_resolution(engine, 512) if arm == "rres512" and item.get("has_image", True) else nullcontext()
    with manager:
        context = context if context is not None else runtime.prepare(engine, runtime.public_item(item))
        keep = runtime.choose_keep(engine, context, arm, models)
        result = runtime.generate(engine, context, keep, max_tokens=cap, ignore_eos=ignore_eos)
        count = int(context["prep"]["n_vis"])
        validate_native(result, arm, count, cap)
    return result, context


def load_gate_helpers():
    # The historical gate module imports its sibling oracle by its script
    # name. Mirror the old validated launcher without editing frozen files.
    oracle_directory = str(QWEN / "scripts/multistream_oracle")
    if oracle_directory not in sys.path:
        sys.path.insert(0, oracle_directory)
    from scripts.multistream_oracle.gates import Gates, stock_identity
    return Gates, stock_identity


def official_prompt_row(item):
    """Mirror pandas' absent-hint semantics in the native prompt fixture."""
    row = dict(question=item["question"], **item.get("options", {}))
    if item.get("hint"):
        row["hint"] = item["hint"]
    return row


def gpu_gate(engine, manifest, models, run_dir, frozen):
    """Input/interface checks only; no target accuracy determines continuation."""
    from . import runtime
    import torch
    Gates, stock_identity = load_gate_helpers()
    from types import SimpleNamespace
    import common
    output = run_dir / "gates/gpu.json"
    if output.exists():
        old = read_json(output)
        if old.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise ValueError("Gate freeze differs")
        if old.get("status") == "PASS":
            return old
        number = 1
        while output.with_name(f"gpu.failed_{number}.json").exists():
            number += 1
        output.rename(output.with_name(f"gpu.failed_{number}.json"))
    payload = dict(status="RUNNING", freeze_sha256=frozen["freeze_sha256"], started_utc=now())
    g = Gates()
    try:
        for dataset in DATASETS:
            item = sorted((i for i in manifest["datasets"][dataset]["items"] if i.get("has_image", True)),
                          key=lambda i: str(i["item_id"]))[0]
            clean = runtime.public_item(item)
            proxy = SimpleNamespace(dump_image=lambda row: item["image_path"])
            engine.vlm.set_dump_image(proxy.dump_image)
            row = official_prompt_row(item)
            official = common.build_message(engine.vlm, proxy, dataset, row)
            g.check(dataset+":official_prompt", official == runtime.message_for(clean))
            first = runtime.prepare(engine, clean)
            second = runtime.prepare(engine, clean)
            g.check(dataset+":fresh_vision_equal", torch.equal(first["V"], second["V"])
                    and all(torch.equal(a, b) for a, b in zip(first["DS"], second["DS"])))
            changed = dict(clean, question=item["question"] + " Please respond briefly.")
            other = runtime.prepare(engine, changed)
            g.check(dataset+":question_sequence_and_vision", not torch.equal(first["question_token_ids"], other["question_token_ids"])
                    and torch.equal(first["V"], other["V"])
                    and all(torch.equal(a, b) for a, b in zip(first["DS"], other["DS"])))
            start = len(g.rows)
            stock_identity(g, engine, first["prep"], first["V"], first["DS"], forced_steps=2)
            for row in g.rows[start:]:
                row["name"] = dataset+":"+row["name"]
            for arm in ARMS:
                cached = first if arm != "rres512" else None
                a, _ = inference(engine, item, arm, models, cap=4, ignore_eos=True, context=cached)
                b, _ = inference(engine, item, arm, models, cap=4, ignore_eos=True)
                fields = ("token_ids", "keep_idx", "positions_sha256", "ds_lengths", "decode_positions")
                g.check(dataset+":"+arm+":shared_fresh", all(a[k] == b[k] for k in fields))
            # Exercise native EOS independently of the fixed-token timing path.
            a, _ = inference(engine, item, ARMS[4], models, context=first)
            b, _ = inference(engine, item, ARMS[4], models)
            g.check(dataset+":native_eos", a["token_ids"] == b["token_ids"] and a["truncated"] == b["truncated"])
            del first, second, other, a, b
        # Raising the unused bag input-length guard must leave existing short
        # question behavior exactly unchanged for every selected checkpoint.
        source = read_json(Path(frozen["source_run"]) / "data/manifest.json") if "source_run" in frozen else None
        if source:
            page = source["splits"]["dev"][0]
            from scripts.student_transfer import runtime as previous_runtime
            original_context = previous_runtime.prepare(engine,
                dict(dataset="DocVQA_VAL", item_id=page["questions"][0]["item_id"],
                     image_path=page["image_path"], image_sha256=page["image_sha256"],
                     question=page["questions"][0]["question"]))
            for arm, model in models.items():
                model.config["max_question_tokens"] = 256
                before = runtime.choose_keep(engine, original_context, arm, models)
                runtime.adapt_student(model)
                after = runtime.choose_keep(engine, original_context, arm, models)
                g.check(arm+":short_input_guard_equivalence", torch.equal(before, after))
            del original_context
        controls = [i for i in manifest["datasets"]["HallusionBench"]["items"] if not i.get("has_image", True)]
        if controls:
            item = controls[0]
            engine.vlm.set_dump_image(lambda row: [])
            official = common.build_message(engine.vlm, SimpleNamespace(), "HallusionBench", dict(question=item["question"]))
            g.check("text_only:official_prompt", official == runtime.message_for(runtime.public_item(item)))
            ctx = runtime.prepare(engine, runtime.public_item(item))
            start = len(g.rows)
            stock_identity(g, engine, ctx["prep"], ctx["V"], ctx["DS"], forced_steps=2)
            for row in g.rows[start:]:
                row["name"] = "text_only:"+row["name"]
            predictions = [inference(engine, item, arm, models, cap=4, ignore_eos=True)[0] for arm in ARMS]
            g.check("text_only:all_arms_equal", all(p["token_ids"] == predictions[0]["token_ids"] for p in predictions))
            del ctx, predictions
        g.check("frozen_base_and_students", not engine.model.training
                and all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
                and all(not model.training for model in models.values()))
        payload.update(status="PASS" if all(row["passed"] for row in g.rows) else "FAIL", checks=g.rows,
                       completed_utc=now(), gpu=torch.cuda.get_device_name(), backend=engine.text.config._attn_implementation)
        atomic_json(output, payload)
        if payload["status"] != "PASS":
            raise RuntimeError("Native transfer gate failed")
        return payload
    except Exception:
        payload.update(status="FAIL", checks=g.rows, error=traceback.format_exc())
        atomic_json(output, payload)
        raise


def evaluate_timing(engine, manifest, models, protocol, frozen, store, progress=None):
    from . import runtime
    timing = protocol["timing"]
    for dataset in DATASETS:
        lookup = {str(i["item_id"]): i for i in manifest["datasets"][dataset]["items"]}
        for item_id in timing["item_ids_by_dataset"][dataset]:
            item = lookup[item_id]
            for cap in timing["caps"]:
                pending = [a for a in ARMS if not all(store.has(dataset, item_id, a, cap, b)
                                                      for b in range(timing["blocks"]))]
                for _ in range(timing["warmup"]):
                    for arm in pending:
                        inference(engine, item, arm, models, cap=cap, ignore_eos=True)
                for block in range(timing["blocks"]):
                    order = list(ARMS)
                    key = f"{dataset}:{item_id}:{cap}:{block}"
                    random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)).shuffle(order)
                    for position, arm in enumerate(order):
                        if store.has(dataset, item_id, arm, cap, block):
                            continue
                        runtime.sync()
                        start = time.perf_counter()
                        result, context = inference(engine, item, arm, models, cap=cap, ignore_eos=True)
                        ready = result["generation_ready_perf_counter"]
                        elapsed = (ready-start)*1000
                        if len(result["token_ids"]) != cap or elapsed <= 0:
                            raise ValueError("Fixed-token timing trace invalid")
                        store.save(dict(dataset=dataset, item_id=item_id, parent_id=item["parent_id"], arm=arm, has_image=True,
                            cap=cap, block=block, order_in_block=position, elapsed_ms=elapsed,
                            generated_tokens=len(result["token_ids"]), token_ids=result["token_ids"],
                            source_visual_count=int(context["prep"]["n_vis"]),
                            keep_idx=result["keep_idx"], actual_visual_count=result["actual_visual_count"],
                            ds_lengths=result["ds_lengths"], positions_sha256=result["positions_sha256"], audit=result["audit"],
                            freeze_sha256=frozen["freeze_sha256"], saved_utc=now()), timing=True)
                        del result, context
                    if progress:
                        progress(timings=store.count("timings"))
            print(json.dumps(dict(phase="timing", dataset=dataset, item_id=item_id, measured=store.count("timings"))), flush=True)


def evaluate_qa(engine, manifest, models, protocol, frozen, store, progress=None):
    from . import runtime
    from .metrics import score_item, parse_prediction
    for dataset in DATASETS:
        for item in manifest["datasets"][dataset]["items"]:
            pending = [arm for arm in ARMS if not store.has(dataset, item["item_id"], arm)]
            context = None
            for arm in pending:
                try:
                    if context is None and arm != "rres512":
                        context = runtime.prepare(engine, runtime.public_item(item))
                    result, used = inference(engine, item, arm, models,
                                              context=context if arm != "rres512" else None)
                    score = score_item(dataset, result["text"], item["answers"], item.get("category"), metadata=item)
                    store.save(dict(result, dataset=dataset, item_id=str(item["item_id"]),
                        parent_id=item["parent_id"], arm=arm,
                        seed=int(arm.rsplit("_", 1)[-1]) if "seed_" in arm else None,
                        score=score, parsed_prediction=parse_prediction(dataset, result["text"], metadata=item),
                        has_image=item.get("has_image", True),
                        source_visual_count=int(used["prep"]["n_vis"]), max_new_tokens=128,
                        freeze_sha256=frozen["freeze_sha256"], saved_utc=now()))
                    del result, used
                except Exception:
                    store.event(dict(phase="qa", dataset=dataset, item_id=str(item["item_id"]), arm=arm,
                                     error=traceback.format_exc()))
                    raise
            del context
            if pending and store.count() % 70 == 0:
                count = store.count()
                print(json.dumps(dict(phase="qa", dataset=dataset, item_id=str(item["item_id"]), outputs=count)), flush=True)
                if progress:
                    progress(outputs=count, dataset=dataset, item_id=str(item["item_id"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--phase", choices=("all", "prepare", "freeze", "gate", "timing", "qa", "report"), default="all")
    parser.add_argument("--timing-images", type=int, default=8)
    parser.add_argument("--timing-blocks", type=int, default=3)
    parser.add_argument("--timing-warmup", type=int, default=2)
    args = parser.parse_args(argv)
    if args.timing_images < 1 or args.timing_blocks < 1 or args.timing_warmup < 0:
        parser.error("Invalid timing sizes")
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run_dir / "status.json"
        previous = read_json(state_path) if state_path.exists() else {}
        state = dict(status="running", pid=os.getpid(), started_utc=previous.get("started_utc", now()),
                     current_invocation_started_utc=now(), phase=args.phase)

        def update(**values):
            state.update(values, updated_utc=now())
            atomic_json(state_path, state)

        store = None
        try:
            update(current_stage="data")
            if not (run_dir / "manifest.json").exists():
                from .data import build_manifest
                build_manifest(run_dir, source_run, datasets=DATASETS)
            if args.phase == "prepare":
                update(status="prepared")
                return
            if args.phase == "report":
                verify_run(run_dir)
                from .report import build_report
                result = build_report(run_dir)
                update(status=result["status"], current_stage="report")
                return
            update(current_stage="freeze")
            frozen, protocol, manifest = freeze_run(run_dir, source_run, args.timing_images,
                                                     args.timing_blocks, args.timing_warmup)
            verify_run(run_dir)
            expected = sum(len(d["items"]) for d in manifest["datasets"].values()) * len(ARMS)
            update(expected_outputs=expected, freeze_sha256=frozen["freeze_sha256"])
            if args.phase == "freeze":
                update(status="frozen")
                return
            from . import runtime
            from scripts.paired_pruning.models import load_checkpoint
            import torch
            update(current_stage="model_load")
            engine = runtime.load_engine(protocol["model_path"])
            device = next(engine.model.parameters()).device
            models = {arm: runtime.adapt_student(load_checkpoint(info["path"], device=device)[0])
                      for arm, info in protocol["checkpoints"].items()}
            update(current_stage="gates")
            gpu_gate(engine, manifest, models, run_dir, frozen)
            store = Store(run_dir / "evaluation/predictions.sqlite3", frozen["freeze_sha256"])
            if args.phase in ("all", "timing"):
                update(current_stage="timing")
                evaluate_timing(engine, manifest, models, protocol, frozen, store, update)
            if args.phase in ("all", "qa"):
                update(current_stage="qa", outputs=store.count())
                evaluate_qa(engine, manifest, models, protocol, frozen, store, update)
            if args.phase == "all":
                update(current_stage="report", outputs=store.count(), timings=store.count("timings"))
                verify_run(run_dir)
                from .report import build_report
                result = build_report(run_dir)
                if result["status"] != "complete":
                    raise RuntimeError("Full run ended with an incomplete report")
                update(status="complete", completed_utc=now())
            else:
                update(status="phase_complete", current_stage=args.phase)
        except Exception:
            error = traceback.format_exc()
            update(status="failed", error=error)
            if store:
                store.event(dict(phase=state.get("current_stage"), error=error))
            raise
        finally:
            if store:
                store.db.close()


if __name__ == "__main__":
    main()
