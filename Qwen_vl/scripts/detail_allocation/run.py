"""Freeze and run a finite-bank, budget-matched spatial-detail diagnostic."""
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
import subprocess
import sys
import time
import traceback

from scripts.paired_pruning.artifacts import atomic_json, sha256, load_protocol
from scripts.paired_pruning.evaluate import json_hash, verify_freeze
from scripts.student_broad.run import Store

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
DEFAULT_RUN = QWEN / "outputs/detail_allocation/run_20261114"
DEFAULT_SOURCE = QWEN / "outputs/paired_pruning/run_20261111"
DATASETS = ("InfoVQA_TRAIN", "TextVQA_TRAIN")
BASELINES = ("full", "b1", "rres512", "uniform256") + tuple(f"bag_rank__seed_{s}" for s in (11, 23, 37))
FACTORIAL_ARMS = tuple(f"factorial__rep_{r}__{condition}" for r in (0, 1) for condition in ("00", "10", "01"))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def frozen_files(paths):
    return {str(p.resolve()): sha256(p) for p in sorted(set(map(Path, paths)))}


def freeze_run(run_dir, source_run, n_main=192, n_dev=2):
    from scripts.paired_pruning import runtime as native
    from scripts.student_transfer.metrics import source_metadata
    from .data import verify_manifest
    from .geometry import build_bank, geometry_report
    from .runtime import message_for, public_item
    import transformers
    import torch

    source_freeze = verify_freeze(source_run)
    original = load_protocol(source_run)
    manifest = read_json(run_dir / "manifest.json")
    verify_manifest(manifest)
    if set(manifest["datasets"]) != set(DATASETS):
        raise ValueError("Diagnostic dataset roster differs")
    tokenizer = transformers.AutoTokenizer.from_pretrained(original["model_path"], local_files_only=True)
    lengths, timings = {}, {}
    identities = set()
    for name in DATASETS:
        data = manifest["datasets"][name]
        if len(data["items"]) != n_main or len(data["dev"]) != n_dev:
            raise ValueError("Manifest sizes differ from the declared independent panel")
        items = data["items"] + data["dev"]
        for item in items:
            if item["dataset"] != name or item["item_id"] in identities:
                raise ValueError("Duplicate/misassigned diagnostic question")
            identities.add(item["item_id"])
        values = [len(tokenizer.encode(message_for(public_item(i))[-1]["value"], add_special_tokens=True)) for i in items]
        if not min(values) > 0 or max(values) > 256:
            raise ValueError("No silent question truncation or new student adaptation is allowed")
        lengths[name] = dict(count=len(values), maximum=max(values))
        timings[name] = [str(item["item_id"]) for item in data["dev"]]
    summary = read_json(source_run / "checkpoints/query_summary.json")
    if summary["status"] != "complete":
        raise ValueError("Source students are not complete")
    checkpoints = {}
    for seed in (11, 23, 37):
        arm = f"bag_rank__seed_{seed}"
        row, = [r for r in summary["checkpoints"] if r["arm"] == "bag_rank" and r["seed"] == seed]
        path = (source_run / f"checkpoints/{arm}.pt").resolve()
        if str(path) != row["path"] or sha256(path) != row["sha256"]:
            raise ValueError("Source validation-selected student differs")
        checkpoints[arm] = dict(path=str(path), sha256=row["sha256"], seed=seed, best_step=row["best_step"])
    bank = build_bank()
    singles = [a for a in bank if a["family"] == "single"]
    duals = [a for a in bank if a["family"] == "dual"]
    arms = list(BASELINES) + [a["action_id"] for a in bank] + list(FACTORIAL_ARMS)
    timing_arms = list(BASELINES) + [singles[0]["action_id"], duals[0]["action_id"]]
    protocol = dict(schema=1, experiment="matched_connected_vs_separated_native_detail",
        source_run=str(source_run), source_freeze_sha256=source_freeze["freeze_sha256"],
        model_path=original["model_path"], model_metadata_sha256=original["model_metadata_sha256"],
        model_weight_stat=original["model_weight_stat"], checkpoints=checkpoints,
        datasets=list(DATASETS), n_main=n_main, n_dev=n_dev, n=1024, k=256,
        max_new_tokens=128, max_question_tokens=256, seeds=[11, 23, 37],
        baseline_arms=list(BASELINES), arms=arms, action_bank=bank, geometry=geometry_report(),
        training="none; no new teacher labels, optimization, checkpoint choice or target parameter adaptation",
        oracle="Each fixed candidate generates without answers; post-hoc maximum QA score over 12 actions per family is an answer-dependent finite-bank diagnostic, never a deployment result",
        factorial="Choose the highest QA-scoring dual, ties lexicographically by action_id; run both deterministic donor replicates on every main item; controls explicitly mark answer-dependent selection",
        metrics={"InfoVQA_TRAIN": "official ANLS", "TextVQA_TRAIN": "official ten-vote VQA consensus"},
        decision=dict(success=.8, failure=.5, minimum_gap_items_per_dataset=12,
            minimum_pooled_delta=.03, strict_total=8, strict_per_dataset=2,
            go="both dataset deltas >0; equal-dataset pooled delta >=.03 and CI lower >0; >=8 strict cases with >=2 per dataset; diagnosis only",
            hold="fewer than 12 full-success/R-res-failure items in either dataset",
            stop="STOP_THIS_GEOMETRY; does not disprove general multi-region perception"),
        timing=dict(arms=timing_arms, item_ids_by_dataset=timings, caps=[1, 16], blocks=3, warmup=2,
            scope="fresh preparation+full/low-resolution vision+fixed selector+native generation; excludes model load and post-generation audit",
            interpretation="Fixed-mask microbenchmarks, NOT oracle-search latency or a learned route's deployment speed; high-resolution arms still pay the complete 1024-token vision encoder"),
        tokenization_audit=lengths, manifest_sha256=sha256(run_dir / "manifest.json"),
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
            transformers=transformers.__version__, cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    atomic_json(run_dir / "protocol.json", protocol, frozen=True)
    paths = set(native.source_paths()) | set(HERE.glob("*.py")) | set(HERE.glob("*.sh"))
    paths.update(Path(p) for p in source_freeze["files"])
    paths.update((QWEN / "scripts/student_transfer").glob("*.py"))
    paths.update((QWEN / "scripts/student_broad").glob("*.py"))
    paths.update(Path(v["path"]) for v in source_metadata().values())
    paths.update(Path(p) for p in manifest["provenance"]["source_files"])
    paths.update(Path(c["path"]) for c in checkpoints.values())
    paths.update((run_dir / "manifest.json", run_dir / "protocol.json", ROOT / "docs/detail_allocation_prereg.md"))
    if (run_dir / "manifest_panel96_preflight.json").exists():
        paths.add(run_dir / "manifest_panel96_preflight.json")
    paths.update(Path(i["image_path"]) for d in manifest["datasets"].values() for s in ("items", "dev") for i in d[s])
    frozen = dict(schema=1, source_run=str(source_run), files=frozen_files(paths),
        local_sources=sorted(str(p.resolve()) for p in HERE.glob("*.py")),
        arms=arms, protocol_sha256=sha256(run_dir / "protocol.json"),
        manifest_sha256=protocol["manifest_sha256"])
    frozen["freeze_sha256"] = json_hash(frozen)
    atomic_json(run_dir / "freeze.json", frozen, frozen=True)
    return frozen, protocol, manifest


def verify_run(run_dir):
    frozen = read_json(run_dir / "freeze.json")
    if json_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"}) != frozen["freeze_sha256"]:
        raise ValueError("Freeze identity changed")
    for path, expected in frozen["files"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input/source/checkpoint changed: {path}; use a new run directory")
    if frozen["local_sources"] != sorted(str(p.resolve()) for p in HERE.glob("*.py")):
        raise ValueError("Local source roster changed")
    protocol = read_json(run_dir / "protocol.json")
    for path, expected in protocol["model_weight_stat"].items():
        stat = Path(path).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError("Frozen base weights changed")
    return frozen


def gpu_snapshot(device=0):
    value = subprocess.check_output(["nvidia-smi", f"--id={device}",
        "--query-gpu=memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    free, used, total = [int(v.strip()) for v in value.split(",")]
    rows = subprocess.check_output(["nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True)
    processes = []
    for row in rows.splitlines():
        if not row.strip() or row.startswith("No running"):
            continue
        pid, name = row.split(",", 1)
        if int(pid.strip()) != os.getpid():
            processes.append(dict(pid=int(pid.strip()), name=name.strip()))
    return dict(free_mib=free, used_mib=used, total_mib=total, other_compute_processes=processes)


def wait_for_gpu(update, *, minimum_free_mib=24576, poll_seconds=30,
                 timeout_seconds=86400, stable_polls=2, stage_after="model_load"):
    started, stable = time.monotonic(), 0
    while True:
        snapshot = gpu_snapshot()
        okay = snapshot["free_mib"] >= minimum_free_mib and not snapshot["other_compute_processes"]
        stable = stable + 1 if okay else 0
        update(status="waiting_gpu", current_stage="waiting_gpu", gpu=snapshot,
               gpu_wait_elapsed_seconds=round(time.monotonic() - started, 1),
               gpu_wait_reason="Need enough free device memory and no other reported compute process; never terminate other tasks")
        if stable >= stable_polls:
            update(status="running", current_stage=stage_after, gpu=snapshot)
            return snapshot
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("GPU remained busy; prepared/frozen artifacts are preserved for same-command resume")
        time.sleep(poll_seconds)


def load_gate_helpers():
    directory = str(QWEN / "scripts/multistream_oracle")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from scripts.multistream_oracle.gates import Gates, stock_identity
    return Gates, stock_identity


def uniform_geometry():
    from .geometry import uniform256_indices
    return dict(action_id="uniform256", family="uniform", keep_idx=uniform256_indices())


def gpu_gate(engine, manifest, models, protocol, frozen, run_dir):
    from . import runtime
    from .geometry import factorial_controls
    import torch
    import common
    from types import SimpleNamespace
    Gates, stock_identity = load_gate_helpers()
    output = run_dir / "gates/gpu.json"
    if output.exists():
        previous = read_json(output)
        if previous.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise ValueError("Gate freeze differs")
        if previous.get("status") == "PASS":
            return previous
        number = 1
        while output.with_name(f"gpu.failed_{number}.json").exists():
            number += 1
        output.rename(output.with_name(f"gpu.failed_{number}.json"))
    g = Gates()
    payload = dict(status="RUNNING", freeze_sha256=frozen["freeze_sha256"], started_utc=now())
    try:
        lookup = {a["action_id"]: a for a in protocol["action_bank"]}
        uniform = uniform_geometry()
        first_dual = next(a for a in protocol["action_bank"] if a["family"] == "dual")
        for dataset in DATASETS:
            item = manifest["datasets"][dataset]["dev"][0]
            clean = runtime.public_item(item)
            engine.vlm.set_dump_image(lambda row: item["image_path"])
            official = common.build_message(engine.vlm, SimpleNamespace(),
                "InfoVQA_VAL" if dataset == "InfoVQA_TRAIN" else "TextVQA_VAL", dict(question=item["question"]))
            g.check(dataset+":official_prompt", official == runtime.message_for(clean))
            first = runtime.prepare(engine, clean)
            second = runtime.prepare(engine, clean)
            g.check(dataset+":fresh_features_equal", torch.equal(first["V"], second["V"])
                    and all(torch.equal(a, b) for a, b in zip(first["DS"], second["DS"])))
            other = runtime.prepare(engine, dict(clean, question=clean["question"] + " Please respond briefly."))
            g.check(dataset+":question_changes_only_query", not torch.equal(first["question_token_ids"], other["question_token_ids"])
                    and torch.equal(first["V"], other["V"]))
            positions = engine.positions_full(first["prep"])[0][:, 0,
                first["prep"]["img_start"]:first["prep"]["img_end"]]
            rows = torch.arange(32, device=positions.device).repeat_interleave(32)
            cols = torch.arange(32, device=positions.device).repeat(32)
            g.check(dataset+":physical_32x32_raster", positions.shape == (3, 1024)
                    and torch.equal(positions[1] - positions[1, 0], rows)
                    and torch.equal(positions[2] - positions[2, 0], cols))
            offset = len(g.rows)
            stock_identity(g, engine, first["prep"], first["V"], first["DS"], forced_steps=2)
            for row in g.rows[offset:]:
                row["name"] = dataset+":"+row["name"]
            for arm in list(BASELINES) + list(lookup):
                geometry = uniform if arm == "uniform256" else lookup.get(arm)
                a, _ = runtime.inference(engine, item, arm, models, geometry=geometry, cap=2,
                    ignore_eos=True, context=None if arm == "rres512" else first)
                b, _ = runtime.inference(engine, item, arm, models, geometry=geometry, cap=2, ignore_eos=True)
                fields = ("token_ids", "keep_idx", "positions_sha256", "decode_positions", "ds_lengths")
                g.check(dataset+":"+arm+":shared_fresh", all(a[k] == b[k] for k in fields))
            controls = factorial_controls(first_dual)
            for control in controls:
                a, _ = runtime.inference(engine, item, "factorial_gate", models, geometry=control,
                    cap=2, ignore_eos=True, context=first, answer_dependent=True)
                g.check(dataset+":"+control["action_id"]+":honest_oracle_audit",
                    a["audit"]["answer_in_selection"] is True and a["audit"]["generation_answer_free"] is True)
            a, _ = runtime.inference(engine, item, next(iter(models)), models, context=first)
            b, _ = runtime.inference(engine, item, next(iter(models)), models)
            g.check(dataset+":native_eos", a["token_ids"] == b["token_ids"] and a["truncated"] == b["truncated"])
            del first, second, other, a, b, positions
        g.check("base_and_students_frozen", not engine.model.training
            and all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
            and all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in models.values()))
        payload.update(status="PASS" if all(row["passed"] for row in g.rows) else "FAIL", checks=g.rows,
            completed_utc=now(), gpu=torch.cuda.get_device_name(), backend=engine.text.config._attn_implementation)
        atomic_json(output, payload)
        if payload["status"] != "PASS":
            raise RuntimeError("Native diagnostic gates did not pass")
        return payload
    except Exception:
        payload.update(status="FAIL", checks=g.rows, error=traceback.format_exc())
        atomic_json(output, payload)
        raise


def get_prediction(store, dataset, item_id, arm):
    row = store.db.execute("SELECT payload FROM predictions WHERE dataset=? AND item_id=? AND arm=?",
                           (dataset, str(item_id), arm)).fetchone()
    return json.loads(row[0]) if row else None


def choose_dual(store, item, bank):
    from .runtime import score_item
    candidates = []
    for action in bank:
        if action["family"] != "dual":
            continue
        record = get_prediction(store, item["dataset"], item["item_id"], action["action_id"])
        if record is None:
            raise ValueError("Every dual candidate must finish before factorial selection")
        score = score_item(item["dataset"], record["text"], item["answers"])
        if abs(record["score"] - score) > 1e-12:
            raise ValueError("Saved candidate score differs from original-answer scoring")
        candidates.append((-score, action["action_id"], action))
    return min(candidates, key=lambda x: (x[0], x[1]))[2]


def evaluate_qa(engine, manifest, models, protocol, frozen, store, update=None):
    from . import runtime
    from .geometry import factorial_controls
    bank = protocol["action_bank"]
    lookup = {a["action_id"]: a for a in bank}
    uniform = uniform_geometry()
    for dataset in DATASETS:
        for number, item in enumerate(manifest["datasets"][dataset]["items"], 1):
            context = None
            def record(arm, geometry=None, selected=None):
                nonlocal context
                old = get_prediction(store, dataset, item["item_id"], arm)
                if old:
                    if old.get("selected_dual_action") != selected:
                        raise ValueError("Resumed conditional winner differs")
                    return
                if context is None and arm != "rres512":
                    context = runtime.prepare(engine, runtime.public_item(item))
                result, used = runtime.inference(engine, item, arm, models, geometry=geometry,
                    context=None if arm == "rres512" else context, answer_dependent=selected is not None)
                score = runtime.score_item(dataset, result["text"], item["answers"])
                store.save(dict(result, dataset=dataset, item_id=str(item["item_id"]), parent_id=item["parent_id"],
                    arm=arm, score=score, max_new_tokens=128, has_image=True,
                    selected_dual_action=selected, geometry=geometry,
                    freeze_sha256=frozen["freeze_sha256"], saved_utc=now()))
                del result, used
            try:
                for arm in list(BASELINES) + list(lookup):
                    record(arm, uniform if arm == "uniform256" else lookup.get(arm))
                winner = choose_dual(store, item, bank)
                for control in factorial_controls(winner):
                    arm = f"factorial__rep_{control['replicate']}__{control['condition']}"
                    record(arm, control, winner["action_id"])
            except Exception:
                store.event(dict(phase="qa", dataset=dataset, item_id=item["item_id"], error=traceback.format_exc()))
                raise
            finally:
                del context
            if number % 5 == 0 or number == len(manifest["datasets"][dataset]["items"]):
                count = store.count()
                print(json.dumps(dict(phase="qa", dataset=dataset, items_done=number, outputs=count)), flush=True)
                if update:
                    update(outputs=count, dataset=dataset, dataset_items_done=number)


def evaluate_timing(engine, manifest, models, protocol, frozen, store, update=None, resource_check=None):
    from . import runtime
    lookup = {a["action_id"]: a for a in protocol["action_bank"]}
    uniform = uniform_geometry()
    config = protocol["timing"]
    for dataset in DATASETS:
        items = {str(i["item_id"]): i for i in manifest["datasets"][dataset]["dev"]}
        for item_id in config["item_ids_by_dataset"][dataset]:
            item = items[item_id]
            for cap in config["caps"]:
                pending = [a for a in config["arms"] if not all(store.has(dataset, item_id, a, cap, b) for b in range(config["blocks"]))]
                if pending and resource_check:
                    resource_check()
                for _ in range(config["warmup"]):
                    for arm in pending:
                        geometry = uniform if arm == "uniform256" else lookup.get(arm)
                        runtime.inference(engine, item, arm, models, geometry=geometry, cap=cap, ignore_eos=True)
                for block in range(config["blocks"]):
                    order = list(config["arms"])
                    random.Random(int(hashlib.sha256(f"{dataset}:{item_id}:{cap}:{block}".encode()).hexdigest()[:16], 16)).shuffle(order)
                    for position, arm in enumerate(order):
                        if store.has(dataset, item_id, arm, cap, block):
                            continue
                        geometry = uniform if arm == "uniform256" else lookup.get(arm)
                        runtime.sync()
                        start = time.perf_counter()
                        result, context = runtime.inference(engine, item, arm, models, geometry=geometry, cap=cap, ignore_eos=True)
                        elapsed = (result["generation_ready_perf_counter"] - start) * 1000
                        if len(result["token_ids"]) != cap or elapsed <= 0:
                            raise ValueError("Invalid fixed-token timing trace")
                        store.save(dict(result, dataset=dataset, item_id=str(item_id), parent_id=item["parent_id"],
                            arm=arm, cap=cap, block=block, order_in_block=position, elapsed_ms=elapsed,
                            generated_tokens=len(result["token_ids"]), geometry=geometry, has_image=True,
                            freeze_sha256=frozen["freeze_sha256"], saved_utc=now()), timing=True)
                        del result, context
                    if update:
                        update(timings=store.count("timings"))
            print(json.dumps(dict(phase="timing", dataset=dataset, item_id=item_id, measured=store.count("timings"))), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--phase", choices=("all", "prepare", "freeze", "gate", "timing", "qa", "report"), default="all")
    parser.add_argument("--n-main", type=int, default=192)
    parser.add_argument("--n-dev", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=24576)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=86400)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        parser.error("This single-A40 protocol supports physical GPU0 only; set CUDA_VISIBLE_DEVICES=0")
    if min(args.n_main, args.n_dev, args.min_free_mib, args.gpu_poll_seconds, args.gpu_wait_timeout_seconds) <= 0:
        parser.error("Panel and resource limits must be positive")
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run_dir / "status.json"
        previous = read_json(path) if path.exists() else {}
        state = dict(status="running", pid=os.getpid(), started_utc=previous.get("started_utc", now()),
            current_invocation_started_utc=now(), phase=args.phase)
        def update(**values):
            state.update(values, updated_utc=now())
            atomic_json(path, state)
        store = None
        try:
            update(current_stage="data")
            if not (run_dir / "manifest.json").exists():
                from .data import build_manifest
                build_manifest(run_dir, n_main=args.n_main, n_dev=args.n_dev)
            if args.phase == "prepare":
                update(status="prepared")
                return
            if args.phase == "report":
                verify_run(run_dir)
                from .report import build_report
                report = build_report(run_dir)
                update(status=report["status"], current_stage="report")
                return
            update(current_stage="freeze")
            frozen, protocol, manifest = freeze_run(run_dir, source_run, args.n_main, args.n_dev)
            verify_run(run_dir)
            expected = sum(len(d["items"]) for d in manifest["datasets"].values()) * len(protocol["arms"])
            update(expected_outputs=expected, expected_timings=2 * args.n_dev * len(protocol["timing"]["arms"]) * 2 * 3,
                   freeze_sha256=frozen["freeze_sha256"])
            if args.phase == "freeze":
                update(status="frozen")
                return
            wait_for_gpu(update, minimum_free_mib=args.min_free_mib, poll_seconds=args.gpu_poll_seconds,
                timeout_seconds=args.gpu_wait_timeout_seconds)
            from . import runtime
            from scripts.paired_pruning.models import load_checkpoint
            engine = runtime.load_engine(protocol["model_path"])
            device = next(engine.model.parameters()).device
            models = {a: load_checkpoint(c["path"], device=device)[0].eval().requires_grad_(False)
                      for a, c in protocol["checkpoints"].items()}
            update(current_stage="gates")
            gpu_gate(engine, manifest, models, protocol, frozen, run_dir)
            store = Store(run_dir / "evaluation/predictions.sqlite3", frozen["freeze_sha256"])
            if args.phase in ("all", "timing"):
                update(current_stage="timing")
                def resource_check():
                    snapshot = gpu_snapshot()
                    if snapshot["other_compute_processes"] or snapshot["free_mib"] < 1024:
                        wait_for_gpu(update, minimum_free_mib=1024, poll_seconds=args.gpu_poll_seconds,
                            timeout_seconds=args.gpu_wait_timeout_seconds, stage_after="timing")
                evaluate_timing(engine, manifest, models, protocol, frozen, store, update, resource_check)
            if args.phase in ("all", "qa"):
                update(current_stage="qa", outputs=store.count())
                evaluate_qa(engine, manifest, models, protocol, frozen, store, update)
            if args.phase == "all":
                update(current_stage="report", outputs=store.count(), timings=store.count("timings"))
                verify_run(run_dir)
                from .report import build_report
                report = build_report(run_dir)
                if report["status"] != "complete":
                    raise RuntimeError("Diagnostic finished with an incomplete report")
                update(status="complete", completed_utc=now(), decision=report.get("decision"))
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
