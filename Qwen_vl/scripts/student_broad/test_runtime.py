"""Native task prompts, immutable weights, text controls and durable records."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import sys

import pytest
import torch

from . import run, runtime


def fixture(dataset="ChartQA_TEST", number="0", image=True):
    return dict(dataset=dataset, item_id=dataset+"_"+number, parent_id="parent"+number,
        source_id="0_0_0_0_0_"+number, question="What is the value?" if dataset not in runtime.YORN else "Is it blue?",
        image_path="/unused.png" if image else None, image_sha256="a"*64 if image else None,
        has_image=image, answers=["Yes"] if dataset in runtime.YORN else ["2"], options={}, hint=None,
        category="test", metadata={"index":"0_0_0_0_0_"+number,"l2-category":"test"})


def generated(count, cap, ignore_eos, text="2"):
    return dict(text=text, token_ids=[2]*(cap if ignore_eos else 1), truncated=bool(ignore_eos),
        keep_idx=list(range(count)), actual_visual_count=count, ds_lengths=[count]*3 if count else [],
        positions_sha256="f"*64, decode_positions=[],
        generation_ready_perf_counter=__import__("time").perf_counter(),
        audit=dict(native=True, ds_streams=3 if count else 0, positions_preserved=True,
                   physical_visual_count=count, answer_in_selection=False, fresh_kv=True,
                   text_only_control=count == 0))


def test_task_prompts_match_original_qwen_methods_with_options_and_hint():
    path = run.QWEN / "VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py"
    spec = importlib.util.spec_from_file_location("isolated_qwen_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.Qwen3VLPromptMixin
    for dataset in run.DATASETS:
        item = fixture(dataset)
        if dataset in runtime.MCQ:
            item.update(hint="Consider the image", options={"A":"first", "B":"second"})
            method = cls._build_mcq_prompt
        elif dataset in runtime.YORN:
            method = cls._build_yorn_prompt
        else:
            method = cls._build_vqa_prompt
        proxy = SimpleNamespace(dump_image=lambda row,dataset: item["image_path"])
        row = run.official_prompt_row(item)
        assert method(proxy, row, dataset) == runtime.message_for(runtime.public_item(item))
        if dataset in runtime.MCQ:
            for missing_hint in (None, ""):
                item["hint"] = missing_hint
                row = run.official_prompt_row(item)
                assert "hint" not in row
                assert method(proxy, row, dataset) == runtime.message_for(runtime.public_item(item))
    item = fixture("HallusionBench", image=False)
    proxy = SimpleNamespace(dump_image=lambda row,dataset: [])
    assert cls._build_yorn_prompt(proxy,dict(question=item["question"]),"HallusionBench") == runtime.message_for(runtime.public_item(item))


def test_bag_length_adapter_changes_no_parameters_or_short_input_scores():
    from scripts.paired_pruning.models import PruningScorer
    torch.set_num_threads(1)
    model = PruningScorer(dict(feature_dim=8, streams=4, hidden_dim=8, num_heads=2,
                               mode_heads=1, max_question_tokens=256, ordered=False)).eval()
    features = torch.randn(1,4,1024,8)
    question = torch.randn(1,12,8)
    mask = torch.ones(1,12,dtype=torch.bool)
    with torch.no_grad():
        before = model(features,question,mask)["scores"]
        weights = {k:v.clone() for k,v in model.state_dict().items()}
        runtime.adapt_student(model)
        after = model(features,question,mask)["scores"]
        long = model(features,torch.randn(1,300,8),torch.ones(1,300,dtype=torch.bool))["scores"]
    assert torch.equal(before,after)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    assert long.shape == (1,1,1024) and torch.isfinite(long).all()
    model.config["ordered"] = True
    with pytest.raises(ValueError,match="unordered"):
        runtime.adapt_student(model)


def test_answer_boundary_and_no_image_budget():
    with pytest.raises(ValueError,match="public"):
        runtime.prepare(None,fixture())
    public = runtime.public_item(fixture("HallusionBench", image=False))
    assert "answers" not in public and "metadata" not in public and "category" not in public
    result = generated(0,128,False,"yes")
    run.validate_native(result,"rres512",0,128)
    result["ds_lengths"] = [0,0,0]
    with pytest.raises(ValueError,match="DeepStack"):
        run.validate_native(result,"rres512",0,128)


def test_actual_qa_timing_report_and_resume_include_text_controls(tmp_path):
    datasets = ("ChartQA_TEST", "HallusionBench")
    items = {"ChartQA_TEST":[fixture()], "HallusionBench":[fixture("HallusionBench"),fixture("HallusionBench","1",False)]}
    manifest = {"datasets":{d:dict(items=items[d],decontaminated_item_ids=[i["item_id"] for i in items[d]]) for d in datasets}}
    protocol = dict(n=1024,k=256,max_new_tokens=128,arms=list(run.ARMS),seeds=[11,23,37],
        timing=dict(caps=[1,16],blocks=1,warmup=0,item_ids_by_dataset={d:[items[d][0]["item_id"]] for d in datasets}))
    engine = SimpleNamespace(count=1024)
    seen = []

    def prepare(engine,public):
        assert set(public) == runtime.PUBLIC_FIELDS
        n = engine.count if public["has_image"] else 0
        return dict(prep=dict(n_vis=n), V=torch.zeros(n,2),
                    fake_text="yes" if public["dataset"] in runtime.YORN else "2")

    @contextmanager
    def low(engine,pixels=512):
        old,engine.count = engine.count,256
        try:
            yield
        finally:
            engine.count = old

    def generate(engine,ctx,keep,max_tokens,ignore_eos):
        seen.append((max_tokens,ignore_eos))
        count = ctx["prep"]["n_vis"] if keep is None else len(keep)
        return generated(count,max_tokens,ignore_eos,ctx["fake_text"])

    frozen = {"files":{}}
    frozen["freeze_sha256"] = run.json_hash(frozen)
    store = run.Store(tmp_path/"evaluation/predictions.sqlite3",frozen["freeze_sha256"])
    with patch.object(run,"DATASETS",datasets), patch.object(runtime,"prepare",side_effect=prepare), \
         patch.object(runtime,"low_resolution",side_effect=low), patch.object(runtime,"sync",return_value=None), \
         patch.object(runtime,"choose_keep",side_effect=lambda eng,ctx,arm,models: None if not ctx["prep"]["n_vis"] or arm in ("full","rres512") else list(range(256))), \
         patch.object(runtime,"generate",side_effect=generate):
        run.evaluate_qa(engine,manifest,{},protocol,frozen,store)
        assert store.count() == 21 and all(not eos for _,eos in seen)
        count = len(seen)
        run.evaluate_qa(engine,manifest,{},protocol,frozen,store)
        assert len(seen) == count
        run.evaluate_timing(engine,manifest,{},protocol,frozen,store)
        assert store.count("timings") == 28 and all(eos for _,eos in seen[count:])
        count = len(seen)
        run.evaluate_timing(engine,manifest,{},protocol,frozen,store)
        assert len(seen) == count
    store.db.close()
    for name,value in (("manifest.json",manifest),("protocol.json",protocol),("freeze.json",frozen)):
        (tmp_path/name).write_text(json.dumps(value))
    from .report import build_report
    report = build_report(tmp_path,draws=32)
    assert report["status"] == "complete"
    with pytest.raises(ValueError,match="freeze"):
        run.Store(tmp_path/"evaluation/predictions.sqlite3","changed")


def test_timing_roster_excludes_no_image_controls_without_using_answers():
    dataset = dict(items=[fixture("HallusionBench","0",False),fixture("HallusionBench","1",True)])
    manifest = {"datasets":{d:dataset for d in run.DATASETS}}
    roster = run.timing_roster(manifest,1)
    for i in dataset["items"]:
        i["answers"] = ["secret changed"]
    assert run.timing_roster(manifest,1) == roster
    assert all(ids == ["HallusionBench_1"] for ids in roster.values())


def test_native_gate_helper_import_without_cuda_or_model():
    code = ("import torch; from scripts.student_broad.run import load_gate_helpers; "
            "g,s=load_gate_helpers(); assert callable(s); assert not torch.cuda.is_initialized()")
    result = subprocess.run([sys.executable,"-c",code],cwd=run.QWEN,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr
