"""Protect answer isolation, EOS semantics, and durable evaluation identity."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import sys

import pytest
import torch

from . import run, runtime


def item_fixture():
    return dict(dataset="TextVQA_VAL", item_id="42", parent_id="page42", question="What number?",
                image_path="/unused.png", image_sha256="a"*64, answers=["2"]*10, category=None)


def generated(count, cap, ignore_eos):
    indices = list(range(count))
    return dict(text="2", token_ids=[2]*(cap if ignore_eos else 1), truncated=bool(ignore_eos),
        keep_idx=indices, actual_visual_count=count, ds_lengths=[count]*3,
        positions_sha256="f"*64, decode_positions=[],
        generation_ready_perf_counter=__import__("time").perf_counter(),
        audit=dict(native=True, ds_streams=3, positions_preserved=True,
                   answer_in_selection=False, fresh_kv=True))


def test_qa_resume_and_no_answer_in_inference(tmp_path):
    item = item_fixture()
    manifest = {"datasets": {dataset: {"items": [dict(item, dataset=dataset,
        answers=["2"]*10 if dataset == "TextVQA_VAL" else ["2"],
        category="Regular Text Recognition" if dataset == "OCRBench" else None)]}
        for dataset in run.DATASETS}}
    seen = []

    def prepare(engine, public):
        assert "answers" not in public and "parent_id" not in public
        assert set(public) == {"dataset", "item_id", "image_path", "image_sha256", "question"}
        return dict(prep={"n_vis": engine.count}, V=torch.zeros(engine.count, 2))

    @contextmanager
    def low(engine, pixels=512):
        old, engine.count = engine.count, 256
        try:
            yield
        finally:
            engine.count = old

    def generate(engine, context, keep, max_tokens, ignore_eos):
        seen.append((max_tokens, ignore_eos))
        return generated(context["prep"]["n_vis"] if keep is None else len(keep), max_tokens, ignore_eos)

    engine = SimpleNamespace(count=1024)
    store = run.Store(tmp_path / "predictions.sqlite3", "freeze")
    protocol = dict(n=1024, k=256, max_new_tokens=128, arms=list(run.ARMS), seeds=[11,23,37],
                    timing=dict(caps=[1,16], blocks=1, warmup=0,
                                item_ids_by_dataset={d:["42"] for d in run.DATASETS}))
    with patch.object(runtime, "prepare", side_effect=prepare), \
         patch.object(runtime, "low_resolution", side_effect=low), \
         patch.object(runtime, "choose_keep", side_effect=lambda eng,ctx,arm,models: None if arm in ("full","rres512") else list(range(256))), \
         patch.object(runtime, "generate", side_effect=generate), \
         patch.object(runtime, "sync", return_value=None):
        run.evaluate_qa(engine, manifest, {}, {}, {"freeze_sha256": "freeze"}, store)
        assert store.count() == len(run.ARMS)*len(run.DATASETS)
        assert all(cap == 128 and eos is False for cap, eos in seen)
        assert all(json.loads(row[0])["score"] == 1 for row in store.db.execute("SELECT payload FROM predictions"))
        count = len(seen)
        run.evaluate_qa(engine, manifest, {}, {}, {"freeze_sha256": "freeze"}, store)
        assert len(seen) == count
        run.evaluate_timing(engine, manifest, {}, protocol, {"freeze_sha256":"freeze"}, store)
        assert store.count("timings") == 2*len(run.ARMS)*len(run.DATASETS)
        assert all(eos is True and cap in (1,16) for cap,eos in seen[count:])
        count = len(seen)
        run.evaluate_timing(engine, manifest, {}, protocol, {"freeze_sha256":"freeze"}, store)
        assert len(seen) == count
    store.db.close()
    for name, value in (("manifest.json",manifest), ("protocol.json",protocol),
                        ("freeze.json",dict(freeze_sha256="freeze", files={}))):
        (tmp_path/name).write_text(json.dumps(value))
    # Consume actual evaluator records, rather than only hand-built report
    # fixtures; catches field drift between timing/QA producers and reports.
    (tmp_path/"evaluation").mkdir()
    (tmp_path/"predictions.sqlite3").rename(tmp_path/"evaluation/predictions.sqlite3")
    from .report import build_report
    report = build_report(tmp_path, draws=32)
    assert report["status"] == "complete"
    with pytest.raises(ValueError, match="freeze"):
        run.Store(tmp_path / "evaluation/predictions.sqlite3", "different")


def test_timing_uses_independent_images_and_input_only_roster():
    items = [dict(item_fixture(), item_id=str(i), image_sha256=str(i//2), answers=[str(i)]) for i in range(8)]
    manifest = {"datasets": {d: {"items": items, "decontaminated_item_ids": [str(i) for i in range(2,8)]}
                             for d in run.DATASETS}}
    first = run.timing_roster(manifest, 3)
    for item in items:
        item["answers"] = ["changed secret label"]
    assert run.timing_roster(manifest, 3) == first
    assert all(len(ids) == 3 and "0" not in ids and "1" not in ids for ids in first.values())


def test_prepare_rejects_answer_fields_before_opening_image():
    with pytest.raises(ValueError, match="public"):
        runtime.prepare(None, item_fixture())


def test_native_audit_rejects_misaligned_stream_budget():
    result = generated(256, 128, False)
    run.validate_native(result, "bag_rank__seed_11", 1024, 128)
    result["ds_lengths"] = [256, 256, 1024]
    with pytest.raises(ValueError, match="DeepStack"):
        run.validate_native(result, "bag_rank__seed_11", 1024, 128)


def test_native_gate_helpers_import_before_model_loading():
    code = ("import torch; from scripts.student_transfer.run import load_gate_helpers; "
            "g,s=load_gate_helpers(); assert callable(s); assert not torch.cuda.is_initialized()")
    result = subprocess.run([sys.executable, "-c", code], cwd=run.QWEN,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
