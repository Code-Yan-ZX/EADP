"""Resource safety and actual runner/report integration without a GPU."""
from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.paired_pruning.artifacts import sha256
from . import run, runtime
from .geometry import build_bank


def item(dataset, identity):
    return dict(dataset=dataset, item_id=dataset+"_"+identity, parent_id=dataset+"_page_"+identity,
        image_path="/unused.png", image_sha256="a"*64, question="Is the answer yes?",
        answers=["yes"]*(10 if dataset == "TextVQA_TRAIN" else 1))


def result(arm, keep, *, answer="yes", cap=128, ignore_eos=False, dependent=False):
    source = 256 if arm == "rres512" else 1024
    tokens = [1] * (cap if ignore_eos else 1)
    decode = list(range(2000, 2000+len(tokens)-1))
    return dict(text=answer, token_ids=tokens, truncated=bool(ignore_eos), keep_idx=keep,
        actual_visual_count=len(keep), source_visual_count=source, ds_lengths=[len(keep)]*3,
        positions_sha256="b"*64, gradients_enabled=False, decode_positions=decode,
        prefill_length=len(keep)+10, cache_length=len(keep)+10+len(decode),
        selector_answer_dependent=dependent,
        selection_scope="offline_oracle_control" if dependent else "fixed_input_rule",
        generation_ready_perf_counter=time.perf_counter(),
        audit=dict(native=True, ds_streams=3, positions_preserved=True, physical_visual_count=len(keep),
            fresh_kv=True, answer_in_selection=dependent, generation_answer_free=True))


def test_resource_wait_requires_real_free_memory_and_no_other_compute():
    low = dict(free_mib=18000, used_mib=27000, total_mib=46068, other_compute_processes=[])
    occupied = dict(free_mib=44000, used_mib=1000, total_mib=46068,
                    other_compute_processes=[dict(pid=123, name="other-user-task")])
    ready = dict(free_mib=44000, used_mib=1000, total_mib=46068, other_compute_processes=[])
    updates = []
    with patch.object(run, "gpu_snapshot", side_effect=[low, occupied, ready, ready]), patch.object(run.time, "sleep") as sleep:
        observed = run.wait_for_gpu(lambda **values: updates.append(values), poll_seconds=.001)
    assert observed == ready and sleep.call_count == 3
    assert updates[-1]["status"] == "running" and updates[-1]["current_stage"] == "model_load"
    assert all(row["status"] == "waiting_gpu" for row in updates[:-1])
    assert updates[1]["gpu"]["other_compute_processes"][0]["pid"] == 123


def test_resource_wait_timeout_preserves_wait_state_without_loading():
    busy = dict(free_mib=100, used_mib=45000, total_mib=46068, other_compute_processes=[])
    values = []
    with patch.object(run, "gpu_snapshot", return_value=busy), patch.object(run.time, "monotonic", side_effect=[0., 2., 3.]):
        with pytest.raises(TimeoutError):
            run.wait_for_gpu(lambda **row: values.append(row), timeout_seconds=1)
    assert values[-1]["status"] == "waiting_gpu"


def test_wrapper_marks_conditional_selection_and_rejects_rres_feature_reuse():
    i = item("InfoVQA_TRAIN", "1")
    context = dict(V=torch.zeros(1024, 2), prep=dict(n_vis=1024))
    geometry = build_bank()[0]
    engine = SimpleNamespace()
    def native_generate(engine, context, keep, cap, ignore_eos):
        return result("fixed", keep.tolist(), cap=cap, ignore_eos=ignore_eos)
    with patch.object(runtime.transfer, "generate", side_effect=native_generate):
        row, _ = runtime.inference(engine, i, "conditional", {}, geometry=geometry,
                                    context=context, answer_dependent=True)
    assert row["selector_answer_dependent"] is True
    assert row["audit"]["answer_in_selection"] is True
    assert row["audit"]["generation_answer_free"] is True
    assert row["selection_scope"] == "offline_oracle_control"
    with pytest.raises(ValueError, match="own genuinely low-resolution"):
        runtime.inference(engine, i, "rres512", {}, context=context)
    with pytest.raises(ValueError, match="only public"):
        runtime.prepare(None, i)


def test_dual_choice_uses_actual_scores_and_deterministic_tie_break(tmp_path):
    bank = build_bank()
    i = item("InfoVQA_TRAIN", "one")
    store = run.Store(tmp_path/"outputs.sqlite3", "c"*64)
    for action in bank:
        if action["family"] == "dual":
            store.save(dict(dataset=i["dataset"], item_id=i["item_id"], arm=action["action_id"], text="yes", score=1.))
    chosen = run.choose_dual(store, i, bank)
    assert chosen["action_id"] == min(a["action_id"] for a in bank if a["family"] == "dual")
    key = (i["dataset"], i["item_id"], chosen["action_id"])
    row = run.get_prediction(store, *key)
    row["score"] = .1
    store.db.execute("UPDATE predictions SET payload=? WHERE dataset=? AND item_id=? AND arm=?", (json.dumps(row), *key))
    store.db.commit()
    with pytest.raises(ValueError, match="differs"):
        run.choose_dual(store, i, bank)
    store.db.close()


def test_actual_runner_to_report_and_resume_with_real_geometry(tmp_path):
    from .report import build_report
    bank = build_bank()
    fixed = {a["action_id"]: a for a in bank}
    winner = min(a["action_id"] for a in bank if a["family"] == "dual")
    manifest = dict(datasets={name: dict(items=[item(name, "main")],
        dev=[item(name, "dev0"), item(name, "dev1")]) for name in run.DATASETS})
    timing_arms = [*run.BASELINES, bank[0]["action_id"], next(a["action_id"] for a in bank if a["family"] == "dual")]
    protocol = dict(n=1024, k=256, max_new_tokens=128, action_bank=bank,
        arms=[*run.BASELINES, *fixed, *run.FACTORIAL_ARMS], baseline_arms=list(run.BASELINES),
        timing=dict(arms=timing_arms, caps=[1, 16], warmup=1, blocks=3,
                    item_ids_by_dataset={name: [i["item_id"] for i in d["dev"]] for name, d in manifest["datasets"].items()}))
    for name, value in (("manifest.json", manifest), ("protocol.json", protocol)):
        (tmp_path/name).write_text(json.dumps(value))
    source = tmp_path/"frozen_source.py"
    source.write_text("# synthetic interface fixture\n")
    frozen = dict(files={str(p): sha256(p) for p in [source, tmp_path/"manifest.json", tmp_path/"protocol.json"]})
    frozen["freeze_sha256"] = run.json_hash(frozen)
    (tmp_path/"freeze.json").write_text(json.dumps(frozen))
    store = run.Store(tmp_path/"evaluation/predictions.sqlite3", frozen["freeze_sha256"])
    calls = []
    def prepare(engine, clean):
        assert set(clean) == {"dataset", "item_id", "question", "image_path", "image_sha256"}
        return dict(prep=dict(n_vis=1024), V=torch.empty(0))
    def inference(engine, raw, arm, models, *, geometry=None, cap=128, ignore_eos=False, context=None, answer_dependent=False):
        calls.append((arm, context is None, ignore_eos))
        keep = geometry["keep_idx"] if geometry else list(range(1024 if arm == "full" else 256))
        row = result(arm, keep, answer="yes" if arm in ("full", winner) else "no",
                     cap=cap, ignore_eos=ignore_eos, dependent=answer_dependent)
        return row, dict(prep=dict(n_vis=row["source_visual_count"]))
    with patch.object(runtime, "prepare", side_effect=prepare), patch.object(runtime, "inference", side_effect=inference), patch.object(runtime, "sync"):
        run.evaluate_qa(None, manifest, {}, protocol, frozen, store)
        run.evaluate_timing(None, manifest, {}, protocol, frozen, store)
        assert store.count() == 74 and store.count("timings") == 216
        assert all(fresh for arm, fresh, timing in calls if timing)
        before = len(calls)
        run.evaluate_qa(None, manifest, {}, protocol, frozen, store)
        run.evaluate_timing(None, manifest, {}, protocol, frozen, store)
        assert len(calls) == before
    store.db.close()
    report = build_report(tmp_path, draws=32)
    assert report["status"] == "complete" and report["decision"] == "HOLD_INSUFFICIENT_GAP"
    assert report["n_observed"] == 74 and report["timing"]["n_observed"] == 216
    assert all(d["n_strict"] == 1 for d in report["datasets"].values())
    assert report["n_error_events"] == 0
