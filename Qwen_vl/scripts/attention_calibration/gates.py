"""Development-only native/calibration gates; never reads primary QA answers.

Run this only while holding the experiment's GPU slot. Every attempt is saved
separately, including failures. A failed RESTORE group does not silently become
a successful count study: group_pass records those distinct capabilities.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

from calibration import (AttentionFrame, AttentionLayout, BiasSpec, VisualMassObserver,
                         attention_calibration, build_logit_bias, combine_attention_mask)
from experiment_common import HERE, QWEN, OUT, common, NativeEngine, prepare_sample, save_json, sha256
from rope_bias import distance_log_bias
from runtime import native_axis_map
import method


class Gates:
    def __init__(self):
        self.rows=[]

    def check(self,group,name,passed,**metrics):
        self.rows.append(dict(group=group,name=name,passed=bool(passed),**metrics))

    def run(self,group,name,call):
        try:
            call()
        except Exception as exc:
            self.check(group,name,False,error=repr(exc),traceback=traceback.format_exc())


def tensor_hash(tensor):
    import hashlib
    x=tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


def error_metrics(actual,reference):
    difference=(actual.float()-reference.float()).abs()
    maximum=float(difference.max())
    return dict(max_abs=maximum,relative_to_reference_max=maximum/max(float(reference.float().abs().max()),1e-12),
                finite=bool(torch.isfinite(actual).all()),bit_exact=bool(torch.equal(actual,reference)))


def cpu_group(g):
    run=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(HERE),"-p","test_*.py","-v"],
        text=True,capture_output=True)
    g.check("cpu","all_cpu_suites",run.returncode==0,returncode=run.returncode,
            stdout=run.stdout,stderr=run.stderr)


@torch.inference_mode()
def forced_text_steps(eng,state,tokens):
    outputs=[]
    for index,token in enumerate(tokens):
        cp=state.lk+index
        output=eng.text(input_ids=torch.tensor([[token]],device=state.logits.device),
            attention_mask=torch.ones(1,cp+1,dtype=torch.long,device=state.logits.device),
            position_ids=torch.full((3,1,1),cp+state.rope_delta,dtype=torch.long,device=state.logits.device),
            past_key_values=state.cache,cache_position=torch.tensor([cp],device=state.logits.device),use_cache=True)
        outputs.append(eng.model.lm_head(output.last_hidden_state)[:,-1].clone())
    return outputs


@torch.inference_mode()
def native_identity(g,eng,sample,steps):
    prep,V,DS=sample["prep"],sample["V"],sample["DS"]
    stock=eng.model(**prep["inputs"],use_cache=True)
    native=eng.prefill(prep,V,DS,None,full_lm_head=True)
    metrics=error_metrics(native.logits,stock.logits[:,-1])
    g.check("native","stock_keepall_prefill",metrics["bit_exact"],**metrics,full_shape_lm_head=True)
    tokens=[]; stock_logits=[]; current=stock.logits[:,-1]
    for index in range(steps):
        token=int(current.argmax(-1));tokens.append(token)
        cp=native.lk+index
        stock=eng.model(input_ids=torch.tensor([[token]],device=current.device),
            attention_mask=torch.ones(1,cp+1,dtype=torch.long,device=current.device),
            past_key_values=stock.past_key_values,cache_position=torch.tensor([cp],device=current.device),
            use_cache=True)
        current=stock.logits[:,-1]
        stock_logits.append(current.clone())
    native_logits=forced_text_steps(eng,native,tokens)
    for index,(actual,reference) in enumerate(zip(native_logits,stock_logits)):
        metrics=error_metrics(actual,reference)
        g.check("native",f"stock_keepall_forced_decode_{index}",metrics["bit_exact"],**metrics)
    g.check("native","stock_keepall_cache",native.cache.get_seq_length()==native.lk+steps,
            observed=int(native.cache.get_seq_length()),expected=native.lk+steps)


@torch.inference_mode()
def gpu_operator_group(g,eng):
    attn=eng.text.layers[0].self_attn
    device=next(eng.model.parameters()).device;dtype=next(eng.model.parameters()).dtype
    generator=torch.Generator(device=device).manual_seed(2026110908)
    heads=attn.config.num_attention_heads;kv=heads//attn.num_key_value_groups;dim=attn.head_dim
    for label,qlen,klen in (("prefill",5,5),("decode",1,7),("cached_multiquery",2,7)):
        q=torch.randn(1,heads,qlen,dim,device=device,dtype=dtype,generator=generator)
        k=torch.randn(1,kv,klen,dim,device=device,dtype=dtype,generator=generator)
        v=torch.randn(1,kv,klen,dim,device=device,dtype=dtype,generator=generator)
        cp=torch.arange(klen-qlen,klen,device=device)
        visual=torch.zeros(klen,dtype=torch.bool,device=device);visual[1:4]=True
        weights=torch.ones(klen,device=device);weights[1:4]=torch.tensor([2.,4.,8.],device=device)
        coordinates=torch.arange(klen,device=device).expand(3,-1)
        frame=AttentionFrame(0,heads,cp,coordinates[:,cp],coordinates,visual[cp],visual,weights,dtype,device)
        head=torch.linspace(-.2,.3,heads)
        spec=BiasSpec(log_count_scale=1,visual_bias=math.log(4),layer_head_bias={0:head})
        mask=combine_attention_mask(None,build_logit_bias(frame,spec),cp,klen,dtype)
        actual=sdpa_attention_forward(attn,q,k,v,mask,scaling=attn.scaling,dropout=0.)[0].transpose(1,2)
        # Independent mask and explicit mathematics, with no calibration helper.
        desired=weights.float().log()[None,None,None,:]+(head.to(device)[None,:,None,None]+math.log(4))*visual[None,None,None,:]
        desired=desired.expand(1,heads,qlen,klen).clone()
        allowed=torch.arange(klen,device=device)[None,:]<=cp[:,None]
        desired.masked_fill_(~allowed[None,None],float("-inf"))
        keys=repeat_kv(k,attn.num_key_value_groups);values=repeat_kv(v,attn.num_key_value_groups)
        # Same backend, explicit mask independently assembled and quantized.
        backend=F.scaled_dot_product_attention(q,keys,values,attn_mask=desired.to(dtype),scale=attn.scaling)
        backend_metrics=error_metrics(actual,backend)
        g.check("operator",f"{label}_same_backend_reference",backend_metrics["max_abs"]<=1e-5,
                **backend_metrics,dtype=str(dtype),mask_shape=list(mask.shape))
        reference=((q.float()@keys.float().transpose(-2,-1))*attn.scaling+desired.to(dtype).float()).softmax(-1)@values.float()
        metrics=error_metrics(actual,reference)
        tolerance=max(1e-5,8*torch.finfo(dtype).eps*max(1.,float(reference.abs().max())))
        g.check("operator",f"{label}_float32_explicit_reference",metrics["max_abs"]<=tolerance,
                **metrics,tolerance=tolerance,tolerance_rule="max(1e-5,8*dtype_eps*max(1,max_abs_reference))")
        future=~allowed
        g.check("operator",f"{label}_future_excluded",bool(torch.isneginf(mask.expand(1,heads,qlen,klen)[0,:,future]).all()),
                future_entries=int(future.sum())*heads)


@torch.inference_mode()
def rope_group(g,eng,sample):
    rotary=eng.text.rotary_emb;device=rotary.inv_freq.device
    axes=native_axis_map(rotary);frequency=rotary.inv_freq.float()
    scaling=float(rotary.attention_scaling)
    g.check("restore","rotary_scaling_is_pure_phase",abs(scaling-1.)<=1e-7,
            attention_scaling=scaling,rope_type=str(getattr(rotary,"rope_type",None)),
            mrope_section=list(rotary.mrope_section),axis_counts=[int((axes==i).sum()) for i in range(3)])
    layout=AttentionLayout.from_engine(eng,sample["prep"],sample["vz"][0])
    real=layout.positions_3d.to(device)
    synthetic=torch.tensor([[0,1,7,31],[3,8,17,33],[2,9,12,40]],device=device)
    decode=torch.arange(real.shape[1],real.shape[1]+4,device=device)+layout.rope_delta
    for label,positions in (("synthetic",synthetic),("real_prefill",real),("real_decode",decode.expand(3,-1))):
        raw=positions[:,None,:,None].float()*frequency[None,None,None,:]
        installed=rotary.apply_interleaved_mrope(raw.clone(),rotary.mrope_section)[0]
        direct=positions.transpose(0,1).float()[:,axes]*frequency
        phase_error=float((installed-direct).abs().max())
        g.check("restore",f"{label}_installed_phase",phase_error<=1e-5,max_abs=phase_error,tolerance=1e-5)
        cos,sin=rotary(torch.zeros(1,positions.shape[1],1,device=device,dtype=torch.float32),positions[:,None])
        native_error=max(float((cos[0,:,:frequency.numel()]-direct.cos()*scaling).abs().max()),
                         float((sin[0,:,:frequency.numel()]-direct.sin()*scaling).abs().max()))
        g.check("restore",f"{label}_native_cos_sin",native_error<=1e-5,max_abs=native_error,tolerance=1e-5)
        # Bound memory for real prefill pairs; compare all real keys to a fixed
        # independent set of real/decode queries rather than a QKFx tensor.
        qpos=positions[:,torch.linspace(0,positions.shape[1]-1,min(11,positions.shape[1]),device=device).long()]
        qraw=qpos[:,None,:,None].float()*frequency[None,None,None,:]
        qphase=rotary.apply_interleaved_mrope(qraw.clone(),rotary.mrope_section)[0]
        expected=(qphase[:,None]-installed[None]).cos().mean(-1)
        observed=2-distance_log_bias(qpos,positions,frequency,axes).exp()
        distance_error=float((observed-expected).abs().max())
        g.check("restore",f"{label}_distance_from_native_phase",distance_error<=1e-5,
                max_abs=distance_error,tolerance=1e-5,query_count=qpos.shape[1],key_count=positions.shape[1])
    small=synthetic
    d=2-distance_log_bias(small,small,frequency,axes).exp()
    g.check("restore","distance_diagonal_symmetry_range",bool(torch.allclose(d.diag(),torch.ones(4,device=device),atol=1e-6)
        and torch.allclose(d,d.T,atol=1e-6) and (d>=-1-1e-6).all() and (d<=1+1e-6).all()),
        diagonal_max_error=float((d.diag()-1).abs().max()),symmetry_max_error=float((d-d.T).abs().max()),
        minimum=float(d.min()),maximum=float(d.max()))
    one=torch.tensor([0,4,13,47],device=device)
    uniform=one.expand(3,-1)
    expected=(frequency[None,None,:]*(one[:,None,None]-one[None,:,None])).cos().mean(-1)
    got=2-distance_log_bias(uniform,uniform,frequency,axes).exp()
    g.check("restore","uniform_positions_reduce_to_1d",float((expected-got).abs().max())<=1e-5,
            max_abs=float((expected-got).abs().max()))
    # Official Qwen2.5's contiguous-axis weighted formula, synthetic only.
    sections=[24,20,20];f=torch.linspace(.01,1.,64,device=device)
    contiguous=torch.repeat_interleave(torch.arange(3,device=device),torch.tensor(sections,device=device))
    delta=small[:,:,None]-small[:,None,:]
    by_axis=sum((delta[a,:,:,None].float()*f[sum(sections[:a]):sum(sections[:a+1])]).cos().mean(-1)*sections[a]/sum(sections) for a in range(3))
    actual=2-distance_log_bias(small,small,f,contiguous).exp()
    g.check("restore","qwen25_contiguous_weighted_formula",float((by_axis-actual).abs().max())<=1e-5,
            max_abs=float((by_axis-actual).abs().max()),not_native_qwen3_axis_map=True)


@torch.inference_mode()
def carrier_group(g,eng,sample,steps,mask_atol):
    prep,V,DS=sample["prep"],sample["V"],sample["DS"]
    for label,keep,Vs,DSs,weights in (
        ("vz",sample["vz"][0],sample["vz"][1],sample["vz"][2],sample["count"]),
        ("b2",sample["b2"],V[sample["b2"]],[d[sample["b2"]] for d in DS],torch.ones(256,device=V.device))):
        layout=AttentionLayout.from_engine(eng,prep,keep,weights)
        expected_hashes=dict(keep=tensor_hash(keep),V=tensor_hash(Vs),DS=[tensor_hash(d) for d in DSs],
                             positions=tensor_hash(layout.positions_3d))
        if label=="vz": expected_hashes["membership"]=tensor_hash(sample["vz"][3]["tensors"]["cluster_id"])
        good=keep.numel()==256 and keep.unique().numel()==256 and bool((keep[1:]>keep[:-1]).all()) and int(keep.min())>=0 and int(keep.max())<1024
        g.check("carrier",f"{label}_selection_streams",good and Vs.shape[0]==256 and [d.shape[0] for d in DSs]==[256]*3,
                K=int(keep.numel()),V_length=Vs.shape[0],DS_lengths=[d.shape[0] for d in DSs],hashes=expected_hashes)
        reference=eng.prefill(prep,V,DS,keep,V_sel=Vs,DS_sel=DSs)
        tokens=[int(reference.logits.argmax(-1))]*steps
        ref_logits=[reference.logits.clone()]+forced_text_steps(eng,reference,tokens)
        experiments=(("baseline","vz_maskzero","count","effective","bias_const","restore_prefill","restore_continuous")
            if label=="vz" else ("b2_baseline","b2_maskzero","b2_bias_const","b2_restore_prefill","b2_restore_continuous"))
        for name in experiments:
            group="restore" if "restore" in name else "carrier"
            # Exercise the EXACT deployment specification and carrier, rather
            # than maintaining a parallel collection of hand-built gate specs.
            spec=method.specification(eng,name)
            actual_keep,actual_vs,actual_ds,actual_weights=method.carrier(sample,name)
            effective_vs=V[actual_keep] if actual_vs is None else actual_vs
            effective_ds=[d[actual_keep] for d in DS] if actual_ds is None else actual_ds
            expected_weights=sample["effective"] if name=="effective" else weights
            wired_same=(torch.equal(actual_keep,keep) and torch.equal(effective_vs,Vs)
                and all(torch.equal(a,b) for a,b in zip(effective_ds,DSs))
                and len(effective_ds)==3 and torch.equal(actual_weights,expected_weights))
            g.check(group,f"{name}_actual_carrier_wiring",wired_same,
                source="method.carrier",key_weights_sha256=tensor_hash(actual_weights),
                V_argument_is_none=actual_vs is None,DS_argument_is_none=actual_ds is None)
            is_baseline=name.endswith("baseline")
            is_zero="maskzero" in name
            is_restore="restore" in name
            is_constant="bias_const" in name
            expected_log_count=1. if name in ("count","effective") or is_restore else 0.
            expected_constant=math.log(4) if is_constant else 0.
            spec_valid=(spec.log_count_scale==expected_log_count and spec.visual_bias==expected_constant
                and spec.force_mask==(not is_baseline) and spec.prefill_only==name.endswith("prefill")
                and callable(spec.pair_bias)==is_restore and not spec.layer_head_bias and spec.query_scope=="all")
            g.check(group,f"{name}_actual_specification",spec_valid,source="method.specification",
                log_count_scale=spec.log_count_scale,visual_bias=spec.visual_bias,force_mask=spec.force_mask,
                prefill_only=spec.prefill_only,pair_bias_type=type(spec.pair_bias).__name__)
            used=AttentionLayout.from_engine(eng,prep,actual_keep,actual_weights)
            observer=VisualMassObserver(layers=[0,eng.text.config.num_hidden_layers-1])
            trace=[];handles=[];ds_calls=[]
            original_ds=eng.text._deepstack_process
            original_ds_is_instance_attribute="_deepstack_process" in eng.text.__dict__
            def ds_spy(hidden,visual_mask,deepstack):
                ds_calls.append(dict(length=deepstack.shape[0],hash=tensor_hash(deepstack)))
                return original_ds(hidden,visual_mask,deepstack)
            eng.text._deepstack_process=ds_spy
            try:
                with attention_calibration(eng.text,used,spec,observer):
                    # Registered AFTER calibration: sees the actual mask passed
                    # to unchanged installed self-attention, including decode.
                    for layer in eng.text.layers:
                        signature=inspect.signature(layer.self_attn.forward)
                        def trace_hook(module,args,kwargs,signature=signature):
                            values=signature.bind_partial(*args,**kwargs).arguments
                            h=values["hidden_states"];q=h.shape[1]
                            cache=values.get("past_key_values");past=0 if cache is None else cache.get_seq_length(module.layer_idx)
                            cp=values.get("cache_position",torch.arange(past,past+q,device=h.device));n=past+q
                            vis,w,pos=used.arrays(n,h.device)
                            frame=AttentionFrame(module.layer_idx,module.config.num_attention_heads,cp,pos[:,cp],pos,vis[cp],vis,w,h.dtype,h.device)
                            expected=torch.zeros(1,1,1,n,device=h.device) if spec.prefill_only and not frame.is_prefill else build_logit_bias(frame,spec)
                            mask=values.get("attention_mask")
                            # Check actual allowed logit offsets on the last
                            # query, which sees every current key.
                            last_expected=expected[..., -1, :].expand(1,module.config.num_attention_heads,n).to(h.dtype)
                            if mask is None:
                                last_actual=torch.zeros_like(last_expected)
                                shape=None
                            else:
                                last_actual=mask[..., -1, :n].expand(1,module.config.num_attention_heads,n)
                                shape=list(mask.shape)
                            delta=float((last_actual.float()-last_expected.float()).abs().max())
                            trace.append(dict(layer=module.layer_idx,stage="prefill" if frame.is_prefill else "decode",
                                cache_position_last=int(cp[-1]),mask_shape=shape,bias_shape=list(expected.shape),
                                max_allowed_error=delta,finite_allowed=bool(torch.isfinite(last_actual).all()),
                                visual_nonzero=int((last_expected[...,vis]!=0).sum()),text_nonzero=int((last_expected[...,~vis]!=0).sum()),
                                correction_nonzero=int((last_expected!=0).sum())))
                        handles.append(layer.self_attn.register_forward_pre_hook(trace_hook,with_kwargs=True))
                    state=eng.prefill(prep,V,DS,actual_keep,V_sel=actual_vs,DS_sel=actual_ds)
                    actual_logits=[state.logits.clone()]+forced_text_steps(eng,state,tokens)
            finally:
                if original_ds_is_instance_attribute:
                    eng.text._deepstack_process=original_ds
                else:
                    del eng.text._deepstack_process
                for handle in handles:handle.remove()
            metrics=[error_metrics(a,b) for a,b in zip(actual_logits,ref_logits)]
            if is_baseline:
                g.check("carrier",f"{name}_native_bit_exact",all(m["bit_exact"] for m in metrics),logits=metrics)
            elif is_zero:
                greedy=[int(a.argmax(-1))!=int(b.argmax(-1)) for a,b in zip(actual_logits,ref_logits)]
                g.check("carrier",f"{name}_native_tolerance",all(m["max_abs"]<=mask_atol for m in metrics),
                        logits=metrics,absolute_tolerance=mask_atol,greedy_changed=greedy,
                        distinction="mask control can select a different SDPA kernel; not bit-identity")
            g.check(group,f"{name}_trace",len(trace)==eng.text.config.num_hidden_layers*(steps+1)
                and all(t["finite_allowed"] and t["max_allowed_error"]<=1e-6 for t in trace),trace=trace)
            decode_trace=[t for t in trace if t["stage"]=="decode"]
            should_zero=is_baseline or is_zero or name.endswith("prefill")
            g.check(group,f"{name}_decode_support",all((t["correction_nonzero"]==0) if should_zero else (t["correction_nonzero"]>0) for t in decode_trace),
                    expected_zero=should_zero,decode_calls=len(decode_trace))
            if spec.force_mask:
                g.check(group,f"{name}_decode_explicit_mask_control",all(t["mask_shape"] is not None for t in decode_trace),
                        exact_zero_mathematical_bias=should_zero,mask_shapes=[t["mask_shape"] for t in decode_trace])
            g.check(group,f"{name}_native_streams_cache",state.ds_lengths==[256]*3 and len(ds_calls)==3
                    and [x["hash"] for x in ds_calls]==expected_hashes["DS"] and state.cache.get_seq_length()==state.lk+steps
                    and torch.equal(state.positions_prefill[:,0].cpu(),used.positions_3d),
                    ds_calls=ds_calls,cache_length=int(state.cache.get_seq_length()),expected_cache_length=state.lk+steps,
                    decode_start=state.lk+state.rope_delta,expected_decode_start=int(eng.positions_full(prep)[0].max())+1)
            after=dict(keep=tensor_hash(keep),V=tensor_hash(Vs),DS=[tensor_hash(d) for d in DSs],positions=tensor_hash(layout.positions_3d))
            if label=="vz":after["membership"]=tensor_hash(sample["vz"][3]["tensors"]["cluster_id"])
            g.check(group,f"{name}_inputs_immutable",after==expected_hashes,hashes=after)
            # Independently compare the public runner entry point with the
            # already-audited primitive sequence. Forced fixed-length decode
            # avoids EOS/truncation changing the number of executed calls.
            greedy_tokens=max(2,min(steps+1,5))
            with attention_calibration(eng.text,used,method.specification(eng,name)):
                direct=eng.prefill(prep,V,DS,actual_keep,V_sel=actual_vs,DS_sel=actual_ds)
                direct_ids,direct_text=eng.decode(direct,greedy_tokens,ignore_eos=True)
            wired=method.predict_calibrated(eng,sample,name,max_tokens=greedy_tokens,ignore_eos=True)
            g.check(group,f"{name}_predict_calibrated_wiring",wired["token_ids"]==direct_ids and wired["text"]==direct_text
                and wired["n_vis"]==256 and wired["ds_lengths"]==[256]*3
                and wired["cache_length"]==direct.cache.get_seq_length()
                and wired["decode_positions"]==direct.decode_positions,
                source="method.predict_calibrated",forced_generation_tokens=greedy_tokens,
                exact_token_sequence_match=wired["token_ids"]==direct_ids,
                cache_length=wired["cache_length"],decode_positions=wired["decode_positions"],
                content_not_saved=True)
        # Directly inspect zero size correction and exact log4 support.
        vis,w,pos=layout.arrays(layout.visual_mask.numel(),V.device);cp=torch.arange(vis.numel(),device=V.device)
        frame=AttentionFrame(0,eng.text.config.num_attention_heads,cp,pos,pos,vis,vis,torch.ones_like(w),V.dtype,V.device)
        unit_bias=build_logit_bias(frame,BiasSpec(log_count_scale=1))
        constant=build_logit_bias(frame,BiasSpec(visual_bias=math.log(4)))
        g.check("carrier",f"{label}_unit_sizes_zero_and_log4",bool((unit_bias==0).all()) and bool((constant[...,~vis]==0).all())
                and bool((constant[...,vis]==torch.tensor(math.log(4),device=V.device,dtype=torch.float32)).all()),
                unit_sizes_max_abs=float(unit_bias.abs().max()))
    counts=sample["count"];effective=sample["effective"];diag=sample["vz"][3]["tensors"]
    dominant=torch.isin(sample["vz"][0],diag["dominant_idx"])
    g.check("carrier","actual_count_effective_bounds",int(counts.sum())==1024 and bool(((effective>=1)&(effective<=counts+1e-5)).all())
            and bool((counts[dominant]==1).all()) and bool((effective[dominant]==1).all()),
            count_sum=float(counts.sum()),count_min=float(counts.min()),count_max=float(counts.max()),
            effective_min=float(effective.min()),effective_max=float(effective.max()),exponent=8)
    hooks_before=[len(layer.self_attn._forward_pre_hooks) for layer in eng.text.layers]
    try:
        with attention_calibration(eng.text,AttentionLayout.from_engine(eng,prep,sample["b2"]),BiasSpec(visual_bias=1)):
            raise RuntimeError("intentional gate exception")
    except RuntimeError:pass
    g.check("carrier","exception_restores_hooks",hooks_before==[len(layer.self_attn._forward_pre_hooks) for layer in eng.text.layers]
            and not hasattr(eng.text,"_attention_calibration_active"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest",type=Path,default=OUT/"data/development_manifest.json")
    parser.add_argument("--output",type=Path,default=OUT/"gates.json")
    parser.add_argument("--forced-steps",type=int,default=4)
    parser.add_argument("--maskzero-logit-atol",type=float,default=.5)
    parser.add_argument("--cpu-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    gates=Gates();started=time.time()
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda)
    try:
        gates.run("cpu","cpu_suite_exception",lambda:cpu_group(gates))
        if not args.cpu_only and all(r["passed"] for r in gates.rows):
            ready={}
            def setup():
                manifest=json.loads(args.development_manifest.read_text())
                if manifest.get("role")!="development" or len(manifest["items"])!=1:
                    raise ValueError("gates require the single frozen outside-panel development item")
                environment.update(development_item_id=manifest["items"][0]["item_id"],
                    manifest_sha256=sha256(args.development_manifest))
                model=common.load_model(common.BASELINE_MODEL,max_new_tokens=32)
                eng=NativeEngine(model)
                sample=prepare_sample(eng,manifest["items"][0])
                environment.update(device=str(next(eng.model.parameters()).device),dtype=str(next(eng.model.parameters()).dtype),
                    attention_backend=eng.text.config._attn_implementation,gpu_name=torch.cuda.get_device_name())
                ready.update(engine=eng,sample=sample)
                gates.check("setup","native_development_sample_ready",True)
            gates.run("setup","model_or_sample_setup_exception",setup)
            if ready and all(r["passed"] for r in gates.rows if r["group"]=="setup"):
                eng,sample=ready["engine"],ready["sample"]
                gates.run("native","native_identity_exception",lambda:native_identity(gates,eng,sample,args.forced_steps))
                gates.run("operator","gpu_operator_exception",lambda:gpu_operator_group(gates,eng))
                gates.run("restore","rope_exception",lambda:rope_group(gates,eng,sample))
                gates.run("carrier","carrier_exception",lambda:carrier_group(gates,eng,sample,args.forced_steps,args.maskzero_logit_atol))
    except BaseException as exc:
        # Even cancellation or an unexpected setup error leaves a reproducible
        # failure artifact instead of only a missing or stale gates.json.
        gates.check("setup","unhandled_execution_exception",False,error=repr(exc),traceback=traceback.format_exc())
    groups={group:all(row["passed"] for row in gates.rows if row["group"]==group) for group in {r["group"] for r in gates.rows}}
    required=["cpu"] if args.cpu_only else ["cpu","setup","native","operator","carrier","restore"]
    external_sources=[QWEN/"model/native_qwen3.py",QWEN/"model/e0_selectors.py",QWEN/"model/pruner.py",
        HERE.parent/"binding_oracle/runtime.py",HERE.parent/"binding_oracle/binding_selectors.py",
        QWEN/"VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py"]
    passed_all=all(groups.get(group,False) for group in required)
    payload=dict(all_passed=passed_all,pass_all=passed_all,group_pass=groups,
        calibration_sha256=sha256(HERE/"calibration.py"),
        completed_at=time.time(),elapsed_seconds=time.time()-started,cpu_only=args.cpu_only,environment=environment,
        source_sha256={str(p.resolve()):sha256(p) for p in sorted(HERE.glob("*.py"))+external_sources},
        forced_decode_steps=args.forced_steps,maskzero_logit_atol=args.maskzero_logit_atol,checks=gates.rows,
        limitations=["Development interface gates are not downstream QA or a trained-method result.",
            "Maskzero is separately controlled because explicit masks may change GPU SDPA kernels.",
            "Observer uses same-local-trajectory Q/K float32 diagnostic recomputation, never a dense teacher upper bound."])
    attempt=args.output.with_name(args.output.stem+"_attempt_"+str(time.time_ns())+args.output.suffix)
    save_json(attempt,payload);save_json(args.output,payload)
    print(json.dumps(dict(all_passed=payload["all_passed"],group_pass=groups,n_checks=len(gates.rows),
        failed=[dict(group=r["group"],name=r["name"],error=r.get("error")) for r in gates.rows if not r["passed"]],
        output=str(args.output),attempt=str(attempt)),indent=2))
    return 0 if payload["all_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
