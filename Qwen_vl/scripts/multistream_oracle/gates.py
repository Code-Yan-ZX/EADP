"""Development-only multistream oracle gates; CPU/GPU attempts are preserved.

Do not run alongside another GPU job. OOM is recorded as a setup/teacher
failure, with no automatic objective, resolution or checkpointing change.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]
OUT=QWEN/"outputs/multistream_oracle"
for directory in (HERE.parent/"discovery",QWEN):
    sys.path.insert(0,str(directory))
import common
from model.native_qwen3 import NativeEngine
from oracle import (STREAM_NAMES,answer_token_ids,score_full_streams,teacher_native_agreement,
    aggregate_scores,generate_selection,frozen_decoder,tensor_sha256)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()


def serial(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(serial(payload),ensure_ascii=False,indent=2)+"\n")
    temporary.replace(path)


class Gates:
    def __init__(self):self.rows=[]
    def check(self,name,passed,**metrics):self.rows.append(dict(name=name,passed=bool(passed),**metrics))
    def run(self,name,call):
        try:return call()
        except Exception as exc:
            self.check(name,False,error=repr(exc),traceback=traceback.format_exc())
            return None


def cpu_checks(g):
    result=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(HERE),"-p","test_*.py","-v"],capture_output=True,text=True)
    g.check("tiny_native_cpu_suite",result.returncode==0,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)


def fingerprint(model):
    """All parameter version counters plus deterministic sampled value hash."""
    digest=hashlib.sha256();versions=[]
    for name,parameter in model.named_parameters():
        versions.append((name,parameter._version))
        flat=parameter.detach().reshape(-1)
        indices=torch.tensor([i*(flat.numel()-1)//max(1,min(37,flat.numel())-1)
            for i in range(min(37,flat.numel()))],device=flat.device)
        values=flat[indices].float().cpu().numpy()
        digest.update(name.encode());digest.update(values.tobytes())
    return dict(sampled_value_sha256=digest.hexdigest(),versions=versions)


@torch.inference_mode()
def stock_identity(g,eng,prep,V,DS,forced_steps=4):
    reference=eng.model(**prep["inputs"],use_cache=True)
    state=eng.prefill(prep,V,DS,None,full_lm_head=True)
    difference=float((state.logits.float()-reference.logits[:,-1].float()).abs().max())
    g.check("stock_native_full_prefill",torch.equal(state.logits,reference.logits[:,-1]),max_abs=difference,full_shape_lm_head=True)
    current=reference.logits[:,-1]
    for index in range(forced_steps):
        token=current.argmax(-1).reshape(1,1);cp=state.lk+index
        mask=torch.ones(1,cp+1,dtype=torch.long,device=V.device)
        reference=eng.model(input_ids=token,attention_mask=mask,past_key_values=reference.past_key_values,
            cache_position=torch.tensor([cp],device=V.device),use_cache=True)
        current=reference.logits[:,-1]
        native=eng.text(input_ids=token,attention_mask=mask,past_key_values=state.cache,
            position_ids=torch.full((3,1,1),cp+state.rope_delta,dtype=torch.long,device=V.device),
            cache_position=torch.tensor([cp],device=V.device),use_cache=True)
        native_logits=eng.model.lm_head(native.last_hidden_state)[:,-1]
        g.check(f"stock_native_forced_decode_{index}",torch.equal(native_logits,current),
            max_abs=float((native_logits.float()-current.float()).abs().max()))


def native_agreement(g,eng,prep,V,DS,answer,args):
    for enabled in (True,False):
        metrics=teacher_native_agreement(eng,prep,V,DS,answer,deepstack=enabled,
            append_eos=False,checkpointing=args.checkpointing)
        cache=metrics["cache"]
        expected_positions=list(range(cache["expected_answer_start"],cache["expected_answer_start"]+metrics["target_count"]-1))
        passed=(metrics["same_shape"]["max_abs"]<=args.logit_atol
            and metrics["cached"]["max_abs"]<=args.logit_atol
            and metrics["same_shape_nll_abs_difference"]<=args.nll_atol
            and metrics["cached_nll_abs_difference"]<=args.nll_atol
            and cache["cache_length"]==cache["expected_cache_length"] and cache["actual_visual_count"]==1024
            and cache["answer_input_positions"]==expected_positions
            and cache["ds_lengths"]==([1024]*3 if enabled else None))
        g.check("teacher_cached_gold_agreement_"+("ds_on" if enabled else "ds_off"),passed,
            logit_atol=args.logit_atol,nll_atol=args.nll_atol,**metrics)


def backward_teacher(g,eng,prep,V,DS,answer,args):
    before=fingerprint(eng.model)
    modes=[module.training for module in eng.model.modules()]
    requires=[parameter.requires_grad for parameter in eng.model.parameters()]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(V.device);started=time.perf_counter()
    teacher=score_full_streams(eng,prep,V,DS,answer,K=256,append_eos=False,checkpointing=args.checkpointing)
    torch.cuda.synchronize()
    timing=dict(wall_seconds=time.perf_counter()-started,**teacher["timings"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(V.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(V.device))
    after=fingerprint(eng.model)
    g.check("frozen_parameter_and_mode_restoration",before==after and modes==[m.training for m in eng.model.modules()]
        and requires==[p.requires_grad for p in eng.model.parameters()] and all(p.grad is None for p in eng.model.parameters()),
        sampled_before=before["sampled_value_sha256"],sampled_after=after["sampled_value_sha256"],
        all_parameter_versions_unchanged=before["versions"]==after["versions"])
    target=answer_token_ids(eng.tok,answer,V.device,append_eos=False)
    g.check("complete_original_gold_no_added_eos",torch.equal(target.cpu(),teacher["gold_token_ids"])
        and teacher["target_token_count"]==target.numel()
        and teacher["teacher_input_length"]==prep["ids"].shape[1]+target.numel()-1
        and teacher["prediction_positions"].tolist()==list(range(prep["ids"].shape[1]-1,prep["ids"].shape[1]-1+target.numel())),
        original_target=answer,target_ids=target.cpu(),first_target=int(target[0]),last_target=int(target[-1]),
        teacher_input_length=teacher["teacher_input_length"],prediction_positions=teacher["prediction_positions"],
        target_count=target.numel(),append_eos=False)
    gradient_norms=teacher["stream_statistics"]["gradient_l2_per_token"]
    g.check("four_stream_finite_gradients",teacher["finite_gradients"] and gradient_norms.shape==(4,1024)
        and bool(torch.isfinite(gradient_norms).all()) and bool((gradient_norms.sum(-1)>0).all()),
        global_gradient_l2=teacher["stream_statistics"]["gradient_l2"],
        zero_gradient_tokens=(gradient_norms==0).sum(-1),timing_and_memory=timing)
    recomputed,keeps=aggregate_scores(teacher["magnitude_per_stream"],teacher["signed_per_stream"],256)
    for name in recomputed:
        keep=teacher["keep_indices"][name]
        g.check(name+"_scores_and_stable_K256",torch.equal(recomputed[name],teacher["scores"][name])
            and torch.equal(keeps[name],keep) and keep.numel()==256 and keep.unique().numel()==256
            and int(keep.min())>=0 and int(keep.max())<1024 and bool((keep[1:]>keep[:-1]).all()),
            keep=keep,score_distribution=teacher["score_distributions"][name])
    # The full maps are a development artifact, not QA-selected labels.
    path=args.output.with_name(args.output.stem+"_dev_teacher_"+str(time.time_ns())+".pt")
    torch.save(teacher,path)
    g.check("development_teacher_artifact",True,path=str(path.resolve()),sha256=sha(path),stream_names=list(STREAM_NAMES))
    return teacher


@torch.inference_mode()
def selected_generation(g,eng,prep,V,DS,teacher):
    full_positions=eng.positions_full(prep)[0]
    for name,keep in teacher["keep_indices"].items():
        keep=keep.to(V.device)
        selected_hashes=dict(V=tensor_sha256(V[keep]),DS=[tensor_sha256(value[keep]) for value in DS],
            positions=tensor_sha256(full_positions[:,:,eng.keep_sequence(prep,keep)]))
        modes=[]
        for enabled in (True,False):
            captured={};ds_calls=[]
            own_ds="_deepstack_process" in eng.text.__dict__;original_ds=eng.text._deepstack_process
            def ds_spy(hidden,visual_mask,value):
                ds_calls.append(tensor_sha256(value));return original_ds(hidden,visual_mask,value)
            eng.text._deepstack_process=ds_spy
            def pre_hook(module,args,kwargs):
                if kwargs.get("inputs_embeds") is not None:
                    embeddings=kwargs["inputs_embeds"]
                    sequence=eng.keep_sequence(prep,keep)
                    visual=(sequence>=prep["img_start"])&(sequence<prep["img_end"])
                    captured["embedding_visual_sha256"]=tensor_sha256(embeddings[0,visual])
                    captured["input_length"]=embeddings.shape[1]
            def post_hook(module,args,output):captured["cache"]=output.past_key_values
            hooks=[eng.text.register_forward_pre_hook(pre_hook,with_kwargs=True),eng.text.register_forward_hook(post_hook)]
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(V.device);started=time.perf_counter()
            try:
                result=generate_selection(eng,prep,V,DS,keep,enabled,max_tokens=4,ignore_eos=True)
                torch.cuda.synchronize()
            finally:
                for hook in hooks:hook.remove()
                if own_ds:eng.text._deepstack_process=original_ds
                else:del eng.text._deepstack_process
            elapsed=time.perf_counter()-started
            all_cache_lengths=[int(layer.keys.shape[-2]) for layer in captured["cache"].layers]
            expected_layers=eng.text.config.num_hidden_layers*(1+len(result["decode_positions"]))
            passed=(result["actual_visual_count"]==256 and result["ds_lengths"]==([256]*3 if enabled else None)
                and ds_calls==(selected_hashes["DS"] if enabled else [])
                and captured["embedding_visual_sha256"]==selected_hashes["V"]
                and result["positions_sha256"]==selected_hashes["positions"]
                and result["prefill_length"]==captured["input_length"]
                and all(length==result["cache_length"] for length in all_cache_lengths)
                and eng._layer_calls==expected_layers)
            g.check(name+("_ds_on" if enabled else "_ds_off")+"_physical_native_generation",passed,
                actual_visual_count=result["actual_visual_count"],native_state_n_vis=result["native_state_n_vis"],
                input_length=captured["input_length"],ds_injection_calls=len(ds_calls),all_cache_lengths=all_cache_lengths,
                layer_calls=eng._layer_calls,expected_layer_calls=expected_layers,selected_hashes=selected_hashes,
                decode_positions=result["decode_positions"],elapsed_seconds=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(V.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(V.device),
                generation_content_not_saved=True)
            modes.append(result)
        g.check(name+"_on_off_same_positions_cache",modes[0]["positions_sha256"]==modes[1]["positions_sha256"]
            and modes[0]["prefill_length"]==modes[1]["prefill_length"]
            and modes[0]["decode_positions"]==modes[1]["decode_positions"])


def exception_restoration(g,eng):
    before=fingerprint(eng.model);flags=[p.requires_grad for p in eng.model.parameters()]
    modes=[m.training for m in eng.model.modules()]
    hooks=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in eng.model.modules()]
    try:
        with frozen_decoder(eng,checkpointing=True):raise RuntimeError("intentional restoration test")
    except RuntimeError:pass
    g.check("exception_restores_model_flags_and_hooks",before==fingerprint(eng.model)
        and flags==[p.requires_grad for p in eng.model.parameters()] and modes==[m.training for m in eng.model.modules()]
        and hooks==[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in eng.model.modules()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest",type=Path,default=OUT/"data/development_manifest.json")
    parser.add_argument("--output",type=Path,default=OUT/"gates.json")
    parser.add_argument("--cpu-only",action="store_true")
    parser.add_argument("--checkpointing",action="store_true")
    parser.add_argument("--logit-atol",type=float,default=.5)
    parser.add_argument("--nll-atol",type=float,default=.03)
    args=parser.parse_args();torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    g=Gates();started=time.time();environment=dict(torch=torch.__version__,cuda=torch.version.cuda)
    try:
        g.run("cpu_exception",lambda:cpu_checks(g))
        if not args.cpu_only and all(row["passed"] for row in g.rows):
            def setup():
                manifest=json.loads(args.development_manifest.read_text())
                items=manifest.get("items",manifest.get("records"))
                if manifest.get("role")!="development" or len(items)!=1:raise ValueError("expected one frozen development item")
                item=items[0]
                if sha(item["image_path"])!=item["image_sha256"]:raise ValueError("development image changed")
                model=common.load_model(common.BASELINE_MODEL,max_new_tokens=32);eng=NativeEngine(model)
                message=[dict(type="image",value=item["image_path"]),dict(type="text",value=item["question"]+"\nAnswer with the value only.")]
                prep=eng.prepare(message,"DocVQA_VAL")
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                V,DS=eng.encode(prep);torch.cuda.synchronize()
                environment.update(device=str(V.device),dtype=str(V.dtype),backend=eng.text.config._attn_implementation,
                    gpu=torch.cuda.get_device_name(),item_id=item["item_id"],manifest_sha256=sha(args.development_manifest),
                    feature_seconds=time.perf_counter()-start,feature_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    feature_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                target=answer_token_ids(eng.tok,item["answers"][0],V.device,append_eos=False)
                g.check("native_full_feature_shapes",V.shape[0]==prep["n_vis"]==1024 and len(DS)==3
                    and all(value.shape==V.shape for value in DS) and eng.inner.visual.spatial_merge_size==2,
                    V_shape=list(V.shape),DS_shapes=[list(value.shape) for value in DS],target_ids=target)
                with torch.inference_mode():
                    stock=eng.inner.visual(prep["pv"],grid_thw=prep["gthw"])
                    stock_v,stock_ds=stock
                    g.check("stock_vision_four_stream_bit_identity",torch.equal(V,stock_v) and all(torch.equal(a,b) for a,b in zip(DS,stock_ds)),
                        grid=prep["gthw"],merge_group_rows=4,merged_count=V.shape[0])
                return eng,prep,V,DS,item["answers"][0]
            ready=g.run("model_or_sample_setup_exception",setup)
            if ready is not None:
                eng,prep,V,DS,answer=ready
                g.run("stock_identity_exception",lambda:stock_identity(g,eng,prep,V,DS))
                g.run("teacher_native_agreement_exception",lambda:native_agreement(g,eng,prep,V,DS,answer,args))
                teacher=g.run("teacher_backward_exception",lambda:backward_teacher(g,eng,prep,V,DS,answer,args))
                if teacher is not None:g.run("selected_generation_exception",lambda:selected_generation(g,eng,prep,V,DS,teacher))
                g.run("restoration_exception",lambda:exception_restoration(g,eng))
    except BaseException as exc:
        g.check("unhandled_execution_exception",False,error=repr(exc),traceback=traceback.format_exc())
    sources=sorted(HERE.glob("*.py"))+[Path(common.__file__),QWEN/"model/native_qwen3.py",
        QWEN/"model/e0_selectors.py",QWEN/"model/pruner.py",
        QWEN/"VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py"]
    passed=bool(g.rows) and all(row["passed"] for row in g.rows)
    payload=dict(pass_all=passed,all_passed=passed,cpu_only=args.cpu_only,checkpointing=args.checkpointing,
        oracle_sha256=sha(HERE/"oracle.py"),source_sha256={str(path.resolve()):sha(path) for path in sources},
        checks=g.rows,environment=environment,logit_atol=args.logit_atol,nll_atol=args.nll_atol,
        elapsed_seconds=time.time()-started,completed_at=time.time(),
        limitations=["Offline gold-answer first-order proxies, not deployable methods or strict accuracy upper bounds.",
            "DS-off state.n_vis is known engine metadata zero; physical embeddings, K, positions and cache are verified separately.",
            "Batched answer-prefix vs cached forcing can differ by bf16 GEMM/kernel shape; raw errors and fixed tolerances are preserved."])
    attempt=args.output.with_name(args.output.stem+"_attempt_"+str(time.time_ns())+args.output.suffix)
    save(attempt,payload);save(args.output,payload)
    print(json.dumps(dict(pass_all=passed,n_checks=len(g.rows),failed=[dict(name=row["name"],error=row.get("error")) for row in g.rows if not row["passed"]],
        output=str(args.output),attempt=str(attempt)),indent=2))
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
