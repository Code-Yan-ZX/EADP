#!/usr/bin/env python3
"""Run the fresh author's complete old EN MMBench protocol, preserving all old runs."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AUTHOR = ROOT / "_reproduction/author_eadp"
PYTHON = "/home/dell/miniconda3/envs/llava_pruner/bin/python"
MODEL = Path("/media/disk2/YZX/doct/FastV/llava-v1.5-7b")
DATA = OUT / "data/mmbench_dev_20230712.tsv"
PRED = OUT / "v15_EADP_K128_oldEN_beta1_alpha05.jsonl"
URL = "https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields_after_comm = stat.rsplit(") ", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields_after_comm[19]),
            "session_id": os.getsid(pid), "process_group": os.getpgid(pid),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def command():
    return [PYTHON, str(OUT / "observe_native.py"), str(OUT / "observed"),
            str(AUTHOR / "LLaVA/llava/eval/model_vqa_mmbench.py"), "--model-path", str(MODEL),
            "--question-file", str(DATA), "--answers-file", str(PRED),
            "--visual_token_num", "128", "--beta", "1.0", "--alpha", "0.5",
            "--single-pred-prompt", "--temperature", "0", "--conv-mode", "vicuna_v1"]


def load_rows():
    csv.field_size_limit(2**31 - 1)
    with DATA.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def prepare():
    from PIL import Image
    rows = load_rows()
    assert len(rows) == 4377
    assert len({r["index"] for r in rows}) == 4377
    groups = Counter(int(r["index"]) % 1000000 for r in rows)
    assert len(groups) == 1176
    decoded, image_records = {}, []
    for row in rows:
        encoded = row["image"]
        assert len(encoded) > 100, "Fresh author loader requires embedded image on every old row"
        payload = base64.b64decode(encoded, validate=True)
        payload_sha = hashlib.sha256(payload).hexdigest()
        if payload_sha not in decoded:
            image = Image.open(io.BytesIO(payload))
            image.load()
            rgb = image.convert("RGB")
            decoded[payload_sha] = {"size": list(image.size), "mode": image.mode,
                                    "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        assert row["answer"] in "ABCD" and len(row["answer"]) == 1
        image_records.append({"index": int(row["index"]), "payload_sha256": payload_sha,
                              **decoded[payload_sha]})
    save("image_manifest.json", image_records)
    source_names = ["LLaVA/scripts/v1_5/eval/mmbench.sh", "LLaVA/EVAL.md",
                    "LLaVA/llava/eval/model_vqa_mmbench.py", "LLaVA/llava/model/llava_arch.py",
                    "LLaVA/llava/model/language_model/llava_llama.py", "LLaVA/llava/model/builder.py",
                    "LLaVA/llava/model/multimodal_encoder/clip_encoder.py", "LLaVA/llava/mm_utils.py",
                    "LLaVA/llava/conversation.py", "LLaVA/llava/constants.py",
                    "LLaVA/scripts/convert_mmbench_for_submission.py"]
    files = [AUTHOR / n for n in source_names] + [OUT / "observe_native.py", Path(__file__)] + [MODEL / n for n in
            ("config.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json")]
    manifest = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Fresh public release script defaults; complete EN Dev 20230712 generation",
        "author_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AUTHOR, text=True).strip(),
        "dataset": {"path": str(DATA), "url": URL, "sha256": sha(DATA), "n_rows": len(rows),
                    "n_circular_groups": len(groups), "unique_image_payloads": len(decoded),
                    "all_images_decoded": True, "group_rotation_counts": dict(Counter(groups.values()))},
        "image_manifest_sha256": sha(OUT / "image_manifest.json"),
        "command": command(), "cwd": str(AUTHOR / "LLaVA"),
        "parameters": {"K": 128, "alpha": .5, "beta": 1, "temperature": 0,
                       "max_new_tokens": 1024, "conv_mode": "vicuna_v1", "single_pred_prompt": True,
                       "lang": "en", "all_rounds": False, "dtype": "float16"},
        "source_files": [{"path": str(p), "sha256": sha(p)} for p in files],
        "fresh_vs_local_source_comparison": [{"relative_path": name,
                "fresh_sha256": sha(AUTHOR / name),
                "old_local_sha256": sha(Path("/media/disk2/YZX/research/EADP_amp") / name),
                "identical": sha(AUTHOR / name) == sha(Path("/media/disk2/YZX/research/EADP_amp") / name)}
                for name in source_names],
        "limits": ["Fresh authors' public script configuration is not proof of unpublished paper argv.",
                   "Both dataset revision and beta differ from old local V11/beta2 run; this first run tests the released recipe as a whole.",
                   "Model and CLIP weights reuse local cached checkpoints; no new checkpoint download is claimed.",
                   "No old prediction reuse or subset projection; all 4377 rows must generate fresh answers.",
                   "Author CUDA stream implementation is used unchanged; no synchronization patch is installed.",
                   "Local exact-letter circular scoring is diagnostic; author evaluation-server text matching is not asserted equivalent."]}
    save("protocol.json", manifest)
    print(json.dumps(manifest["dataset"], ensure_ascii=False), flush=True)


def score():
    rows = load_rows()
    answers = [json.loads(line) for line in PRED.read_text().splitlines() if line.strip()]
    expected = [int(row["index"]) for row in rows]
    actual = [int(answer["question_id"]) for answer in answers]
    if actual != expected:
        raise ValueError("Native prediction order/IDs differ from the entire old benchmark")
    groups, unparsed, row_correct = {}, [], 0
    for row, answer in zip(rows, answers):
        text = answer["text"].strip()
        match = re.search(r"\b([A-E])\b", text)
        letter = match.group(1) if match else None
        correct = letter == row["answer"]
        row_correct += correct
        groups.setdefault(int(row["index"]) % 1000000, []).append(correct)
        if letter is None:
            unparsed.append({"index": int(row["index"]), "text": text})
    correct_groups = sum(all(v) for v in groups.values())
    value = 100 * correct_groups / len(groups)
    save("local_circular_score.json", {
        "metric": "Local exact-letter circular accuracy (%); unparsed answers counted wrong",
        "n_rows": len(rows), "n_groups": len(groups), "row_correct": row_correct,
        "row_accuracy_percent": 100 * row_correct / len(rows), "circular_correct": correct_groups,
        "value": value, "paper_reported": 62.7, "local_minus_paper": value - 62.7,
        "unparsed_count": len(unparsed), "unparsed": unparsed,
        "dataset_sha256": sha(DATA), "prediction_sha256": sha(PRED),
        "scorer_source_sha256": sha(__file__), "all_question_ids_and_order_match": True,
        "limitation": "Paper used external evaluation-server answer matching; exact local parser is not a verified identical server evaluator."})
    print(json.dumps({"n_rows": len(rows), "n_groups": len(groups), "circular_correct": correct_groups,
                      "value": value, "unparsed_count": len(unparsed)}), flush=True)


def convert_submission():
    upload = OUT / "answers_upload"
    upload.mkdir(exist_ok=True)
    convert_command = [PYTHON, str(AUTHOR / "LLaVA/scripts/convert_mmbench_for_submission.py"),
                       "--annotation-file", str(DATA), "--result-dir", str(OUT),
                       "--upload-dir", str(upload), "--experiment", PRED.stem]
    with (OUT / "conversion.log").open("w") as log:
        completed = subprocess.run(convert_command, stdout=log, stderr=subprocess.STDOUT)
    save("conversion.json", {"command": convert_command, "exit_code": completed.returncode,
                             "output": str(upload / (PRED.stem + ".xlsx")),
                             "submitted_to_external_server": False})
    if completed.returncode:
        raise RuntimeError("Fresh author's submission conversion failed; raw prediction is preserved")


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["dataset"]["sha256"] == sha(DATA)
    for item in protocol["source_files"]:
        assert item["sha256"] == sha(item["path"]), item["path"]
    if PRED.exists() and PRED.stat().st_size:
        raise ValueError("Existing new-run output preserved; choose a distinct run directory to retry")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("USE_LLAVA_ARCH_"):
            env.pop(key)
    env.update(PYTHONPATH=str(AUTHOR / "LLaVA"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    state = {"status": "running", "command": command(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "unspecified"),
             "runner_process": process_identity(os.getpid()),
             "author_code_unchanged": True, "protocol_sha256": sha(OUT / "protocol.json")}
    save("state.json", state)
    start = time.monotonic()
    with (OUT / "generation.log").open("w") as log:
        child = subprocess.Popen(command(), cwd=AUTHOR / "LLaVA", env=env, stdout=log, stderr=subprocess.STDOUT)
        state["native_process"] = process_identity(child.pid)
        save("state.json", state)
        returncode = child.wait()
    state.update(status="generation_complete" if returncode == 0 else "failed", exit_code=returncode,
                 duration_seconds=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat())
    if PRED.exists():
        state["prediction_sha256"] = sha(PRED)
        state["n_answers"] = sum(1 for line in PRED.open() if line.strip())
    save("state.json", state)
    if returncode:
        raise SystemExit(returncode)
    try:
        score()
        convert_submission()
    except Exception as error:
        state.update(status="scoring_or_conversion_failed", error=str(error))
        save("state.json", state)
        raise
    state.update(status="complete", score_sha256=sha(OUT / "local_circular_score.json"),
                 conversion_record_sha256=sha(OUT / "conversion.json"))
    save("state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.action]()
