"""Sequential launcher; the user starts the long GPU experiment explicitly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from .artifacts import atomic_json, sha256

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
TRAIN = Path("/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/m3_train_data")
VAL_TSV = Path("/media/disk2/YZX/LMUData/DocVQA_VAL.tsv")
STAGES = ("cpu-tests", "data", "freeze", "gates", "cache", "diagnostics",
          "train-query", "train-sets", "evaluate", "report")


def now():
    return datetime.now(timezone.utc).isoformat()


def model_snapshot(path=None):
    requested = path or os.environ.get("QWEN3_VL_8B_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    direct = Path(requested).expanduser()
    if direct.is_dir():
        return direct.resolve()
    if requested != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("Use an existing local model directory; automatic downloads are disabled")
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home()/".cache/huggingface")))/"hub")))
    model = hub / "models--Qwen--Qwen3-VL-8B-Instruct"
    revision = (model / "refs/main").read_text().strip()
    result = model / "snapshots" / revision
    if not result.is_dir():
        raise FileNotFoundError("Cached Qwen3-VL-8B snapshot is incomplete")
    return result.resolve()


def preflight(args):
    import torch
    import transformers
    import pyarrow
    import PIL
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"Validated native runtime requires transformers 4.57.6; got {transformers.__version__}. "
                           "Use /home/dell/miniconda3/envs/qwen3vl_clean/bin/python via PYTHON.")
    for package in ("editdistance", "numpy", "pandas", "qwen_vl_utils", "accelerate", "pytest"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"Missing package {package} in {sys.executable}")
    shards = sorted((args.train_root / "train_fetch_001/shards").glob("*.parquet"))
    if not shards or not args.val_tsv.is_file():
        raise FileNotFoundError("Missing local DocVQA TRAIN shards or original VAL exclusion TSV")
    snapshot = model_snapshot(args.model_path)
    index = snapshot / "model.safetensors.index.json"
    raw = json.loads(index.read_text())
    weights = sorted(set(raw["weight_map"].values()))
    missing = [name for name in weights if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local model weight shards: {missing}")
    available = shutil.disk_usage(args.run_dir.parent if args.run_dir.parent.exists() else QWEN).free
    estimate = (args.fit_pages + args.val_pages) * 4 * 1024 * 4096 * 2 + (3 << 30)
    if available < estimate:
        raise RuntimeError(f"Need at least {estimate/(1<<30):.1f} GiB for feature caches/checkpoints")
    return dict(python=sys.executable, python_version=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__, pyarrow=pyarrow.__version__,
        pillow=PIL.__version__, cuda=torch.version.cuda, model_path=str(snapshot),
        train_shards=len(shards), free_disk_gib=round(available/(1<<30),1),
        estimated_minimum_cache_gib=round(estimate/(1<<30),1),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        gpu_model_loaded=False)


def freeze(args, environment):
    from .runtime import source_paths, PROMPT_SUFFIX
    from transformers import AutoTokenizer
    manifest = args.run_dir / "data/manifest.json"
    data = json.loads(manifest.read_text())
    requested_counts = dict(fit=args.fit_pages, val=args.val_pages, test=args.test_pages, dev=1)
    if {role:len(rows) for role,rows in data["splits"].items()} != requested_counts:
        raise RuntimeError("Data split sizes do not match launcher arguments")
    snapshot = Path(environment["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    question_lengths, teacher_lengths = [], []
    for role, pages in data["splits"].items():
        for page in pages:
            for question in page["questions"]:
                length = len(tokenizer.encode(question["question"]+PROMPT_SUFFIX, add_special_tokens=True))
                if not 0 < length <= 256:
                    raise ValueError(f"{question['item_id']} needs {length} question tokens; no silent truncation")
                question_lengths.append(length)
                # Check teacher interface limits before loading the GPU model;
                # held-out test answers are not tokenized into teacher labels.
                if role in ("fit", "val", "dev"):
                    answer_length = len(tokenizer.encode(question["answers"][0], add_special_tokens=False))
                    if not 0 < answer_length <= 128:
                        raise ValueError(f"{question['item_id']} first answer needs {answer_length} tokens; offline teacher limit 128")
                    teacher_lengths.append(answer_length)
    metadata = {str(path):sha256(path) for path in snapshot.iterdir()
                if path.is_file() and path.suffix in (".json", ".txt", ".jinja")}
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {str((snapshot/name).resolve()):dict(size=(snapshot/name).stat().st_size,
        mtime_ns=(snapshot/name).stat().st_mtime_ns) for name in sorted(set(weight_map.values()))}
    value = dict(schema_version=1, name="paired_pruning_20261111", seed=args.seed,
        n=1024, k=256, steps=args.steps, seeds=args.seeds, bank_size=args.bank_size,
        max_new_tokens=128, max_question_tokens=256, success_min_anls=.9,
        tokenization_audit=dict(question_count=len(question_lengths), max_question_tokens=max(question_lengths),
                                teacher_answer_count=len(teacher_lengths), max_teacher_answer_tokens=max(teacher_lengths)),
        success_tolerance=.02, cross_max_jaccard=.8, prompt_suffix=PROMPT_SUFFIX,
        model_path=str(snapshot), model_metadata_sha256=metadata, model_weight_stat=weights,
        model_config=dict(feature_dim=4096, streams=4, hidden_dim=64, num_heads=4,
                          mode_heads=4, max_question_tokens=256),
        model_weight_verification="snapshot revision + metadata SHA256 + resolved weight size/mtime; not a full weight-byte hash",
        query_arms=["bag_rank", "ordered_rank", "ordered_pair"],
        set_arms=["canonical", "token_marginal", "multimode"],
        timing_blocks=args.timing_blocks, timing_warmup=args.timing_warmup,
        training_device=args.device, environment={key:environment[key] for key in
            ("python", "python_version", "torch", "transformers", "pyarrow", "pillow", "cuda", "cuda_visible_devices")},
        manifest_sha256=sha256(manifest),
        source_sha256={str(path):sha256(path) for path in source_paths()},
        mechanism_policy="report_only; run both experiments and all seeds; do not tune on test",
        scope="DocVQA TRAIN held-out parent documents; exploratory, not official benchmark test",
        split_pages=requested_counts, questions_per_page=args.questions_per_page)
    atomic_json(args.run_dir / "protocol.json", value, frozen=True)
    print("Protocol frozen:", args.run_dir / "protocol.json", flush=True)


def stage_command(stage, args):
    base = [sys.executable, "-u", "-m"]
    package = "scripts.paired_pruning."
    common = ["--run-dir", str(args.run_dir)]
    if stage == "cpu-tests":
        return [sys.executable, "-m", "pytest", "-q", str(HERE)]
    if stage == "data":
        return base + [package+"data", "--out", str(args.run_dir/"data"),
            "--train-root", str(args.train_root), "--val-tsv", str(args.val_tsv), "--seed", str(args.seed),
            "--fit", str(args.fit_pages), "--val", str(args.val_pages), "--test", str(args.test_pages),
            "--dev", "1", "--questions-per-page", str(args.questions_per_page)]
    if stage.startswith("train-"):
        return base+[package+"train"]+common+["--experiment", stage.removeprefix("train-"),
            "--device", args.device, "--steps", str(args.steps), "--seeds", *map(str,args.seeds)]
    module = {"cache":"build_cache", "evaluate":"evaluate", "report":"report",
              "gates":"gates", "diagnostics":"diagnostics"}[stage]
    command = base+[package+module]+common
    if stage == "evaluate":
        command += ["--timing-blocks", str(args.timing_blocks), "--timing-warmup", str(args.timing_warmup)]
    return command


def run_stage(stage, args):
    command = stage_command(stage, args)
    path = args.run_dir / "logs" / (stage + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] {stage}: {' '.join(command)}", flush=True)
    with path.open("a", buffering=1) as stream:
        stream.write(f"\n[{now()}] START {stage}\n")
        process = subprocess.Popen(command, cwd=QWEN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        stream.write(f"[{now()}] EXIT {code}\n")
    if code:
        raise RuntimeError(f"Stage {stage} exited {code}; inspect {path}. Re-run the same command to resume.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=QWEN/"outputs/paired_pruning/run_20261111")
    p.add_argument("--train-root", type=Path, default=TRAIN)
    p.add_argument("--val-tsv", type=Path, default=VAL_TSV)
    p.add_argument("--model-path", type=str)
    p.add_argument("--fit-pages", type=int, default=128)
    p.add_argument("--val-pages", type=int, default=24)
    p.add_argument("--test-pages", type=int, default=64)
    p.add_argument("--questions-per-page", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026111101)
    p.add_argument("--seeds", type=int, nargs="+", default=[11,23,37])
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--bank-size", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing-blocks", type=int, default=5)
    p.add_argument("--timing-warmup", type=int, default=3)
    p.add_argument("--check-only", action="store_true", help="CPU/environment/local-input preflight, no model load or GPU work")
    p.add_argument("--only", choices=STAGES, help="Run one stage in an already configured run directory")
    return p


def main():
    args = parser().parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.train_root = args.train_root.expanduser().resolve()
    args.val_tsv = args.val_tsv.expanduser().resolve()
    if min(args.fit_pages,args.val_pages,args.test_pages,args.steps,args.timing_blocks) <= 0 or args.timing_warmup < 0:
        raise ValueError("Page counts, steps and timing blocks must be positive")
    if args.questions_per_page < 2 or not 8 <= args.bank_size <= 64:
        raise ValueError("Need >=2 questions/page and 8..64 bank masks")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be nonempty and distinct")
    environment = preflight(args)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir/"launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another launcher owns this run directory")
        status = dict(status="running", pid=os.getpid(), started_utc=now(), stages=[])
        atomic_json(args.run_dir/"status.json", status)
        try:
            stages = (args.only,) if args.only else STAGES
            for stage in stages:
                status.update(current_stage=stage, updated_utc=now())
                atomic_json(args.run_dir/"status.json", status)
                started = time.perf_counter()
                if stage == "freeze":
                    freeze(args, environment)
                else:
                    run_stage(stage, args)
                status["stages"].append(dict(stage=stage, wall_seconds=time.perf_counter()-started))
            status.update(status="complete" if not args.only else "stage_complete", completed_utc=now())
            atomic_json(args.run_dir/"status.json", status)
            print("Finished. Run directory:", args.run_dir, flush=True)
        except BaseException:
            status.update(status="failed", failed_utc=now(), error=traceback.format_exc())
            atomic_json(args.run_dir/"status.json", status)
            raise


if __name__ == "__main__":
    main()
