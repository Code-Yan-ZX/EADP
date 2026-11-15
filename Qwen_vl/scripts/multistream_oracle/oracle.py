"""Offline, answer-conditioned, full-native four-stream gradient diagnostics.

No decoder is copied, no model parameter is trained, and no gradient is used
during evaluation generation. Both score families are first-order proxies at
the dense DS-on teacher, not attainable methods or accuracy upper bounds:

  magnitude_s[i] = sum_d abs(dL/dF_s[i,d] * F_s[i,d])
  signed_s[i]    = -sum_d dL/dF_s[i,d] * F_s[i,d]

The signed joint score is the first-order NLL change under a common amplitude
gate's 1 -> 0 transition for all four streams at one original token position.
Physical deletion additionally changes sequence length and softmax, so this
derivative does NOT equal the actual deletion effect. Primary joint scores sum
raw stream scores, with no clamping or per-stream normalization. Normalized
quantities are exported only as diagnostics.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import time
from typing import Optional

import torch
import torch.nn.functional as F


STREAM_NAMES=("V","DS0","DS1","DS2")
SCORE_NAMES=("magnitude_v","magnitude_joint","signed_v","signed_joint")


def tensor_sha256(tensor):
    raw=tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def answer_token_ids(tokenizer, answer, device=None, append_eos=False, max_answer_tokens=128):
    """Full original answer string, no BOS, no EOS unless explicitly requested.

    An overlong answer raises instead of silently truncating the teacher.
    Aliases are not selected by model likelihood: the caller supplies the
    already-frozen original alias (the experiment uses answers[0]).
    """
    if not isinstance(answer,str):
        raise TypeError("gold answer must be the original string")
    values=tokenizer.encode(answer,add_special_tokens=False)
    if isinstance(values,torch.Tensor): values=values.reshape(-1).tolist()
    values=list(map(int,values))
    if append_eos:
        eos=tokenizer.eos_token_id
        if eos is None: raise ValueError("append_eos requires a tokenizer EOS ID")
        if not values or values[-1]!=int(eos): values.append(int(eos))
    if not values: raise ValueError("gold answer tokenizes to an empty sequence")
    if len(values)>max_answer_tokens:
        raise ValueError(f"full gold requires {len(values)} tokens, above limit {max_answer_tokens}; no truncation performed")
    return torch.tensor(values,dtype=torch.long,device=device)


def checked_keep(keep_idx,n_vis,device):
    if keep_idx is None: return torch.arange(n_vis,device=device)
    keep=torch.as_tensor(keep_idx,device=device,dtype=torch.long).detach().clone()
    if keep.ndim!=1 or not keep.numel() or keep.unique().numel()!=keep.numel():
        raise ValueError("keep indices must be a nonempty unique vector")
    if int(keep.min())<0 or int(keep.max())>=n_vis or not bool((keep[1:]>keep[:-1]).all()):
        raise ValueError("keep indices must be in-range and raster-ascending")
    return keep


@dataclass
class GoldBatch:
    inputs: dict
    targets: torch.Tensor
    prediction_positions: torch.Tensor
    keep_idx: torch.Tensor
    keep_sequence: torch.Tensor
    prompt_length: int
    full_prompt_length: int
    visual_slots: torch.Tensor


def build_gold_batch(eng,prep,V,DS,targets,keep_idx=None,deepstack=True):
    """Build differentiable inputs for the STOCK native text forward.

    Input suffix is targets[:-1]; logit row prompt_len-1 predicts targets[0].
    All answer-prefix positions begin at the ORIGINAL full prompt's maximum
    3D position + 1, even when visual tokens are removed. Causality is handled
    by the installed text model with its ordinary all-one 2D input mask.
    """
    device=V.device
    ids=prep["ids"].detach().clone().to(device)
    if ids.ndim!=2 or ids.shape[0]!=1: raise ValueError("batch one only")
    n_vis=int(prep["n_vis"]);start,end=int(prep["img_start"]),int(prep["img_end"])
    if V.ndim!=2 or V.shape[0]!=n_vis or end-start!=n_vis:
        raise ValueError("expected one complete contiguous visual span")
    if not (0<=start<end<=ids.shape[1]): raise ValueError("invalid visual span")
    if DS is None or len(DS)!=3 or any(value.shape!=V.shape for value in DS):
        raise ValueError("native oracle requires exactly three aligned DS streams")
    targets=torch.as_tensor(targets,device=device,dtype=torch.long).detach().clone()
    if targets.ndim!=1 or not targets.numel(): raise ValueError("empty or nonvector targets")
    keep=checked_keep(keep_idx,n_vis,device)
    sequence=torch.cat((torch.arange(start,device=device),start+keep,torch.arange(end,ids.shape[1],device=device)))
    prompt_len=int(sequence.numel());full_len=ids.shape[1]
    all_ids=torch.cat((ids,targets[:-1].view(1,-1)),dim=1)
    input_sequence=torch.cat((sequence,torch.arange(full_len,all_ids.shape[1],device=device)))
    # Frozen text embeddings are detached. index_copy preserves the gradient
    # ONLY to the visual leaf; three DS leaves are injected by stock forward.
    embeddings=eng.inner.get_input_embeddings()(all_ids)[0,input_sequence].detach()
    visual_slots=(input_sequence>=start)&(input_sequence<end)
    embeddings=embeddings.index_copy(0,torch.where(visual_slots)[0],V[keep].to(embeddings.dtype))
    full_positions=eng.positions_full(prep)[0].detach().clone().to(device)
    if tuple(full_positions.shape)!=(3,1,full_len): raise ValueError("bad native full positions")
    appended=(full_positions.max()+1+torch.arange(targets.numel()-1,device=device)).view(1,1,-1).expand(3,1,-1)
    positions=torch.cat((full_positions[:,:,sequence],appended),dim=-1)
    inputs=dict(inputs_embeds=embeddings[None],
        attention_mask=torch.ones(1,embeddings.shape[0],dtype=torch.long,device=device),
        position_ids=positions,visual_pos_masks=visual_slots[None],
        deepstack_visual_embeds=[value[keep] for value in DS] if deepstack else None,
        use_cache=False)
    prediction_positions=torch.arange(prompt_len-1,prompt_len-1+targets.numel(),device=device)
    return GoldBatch(inputs,targets,prediction_positions,keep,sequence,prompt_len,full_len,visual_slots)


def forward_gold(eng,batch):
    """Return complete answer logits [T,vocab] and mean answer-only NLL."""
    output=eng.text(**batch.inputs)
    hidden=output.last_hidden_state[:,batch.prediction_positions]
    logits=eng.model.lm_head(hidden)[0]
    loss=F.cross_entropy(logits.float(),batch.targets,reduction="mean")
    return logits,loss


@contextmanager
def frozen_decoder(eng,checkpointing=False):
    """Freeze parameters and restore every changed flag after success/failure.

    Non-reentrant stock checkpointing is opt-in. It requires decoder training
    flags to activate, so nonzero attention dropout is rejected. This is a
    memory control, not a different score or a silent response to an OOM.
    """
    parameters=list(eng.model.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("oracle requires no pre-existing model parameter gradients")
    requires=[parameter.requires_grad for parameter in parameters]
    modules=list(eng.model.modules())
    training=[module.training for module in modules]
    attributes={module:{name:(hasattr(module,name),getattr(module,name,None))
        for name in ("gradient_checkpointing","_gradient_checkpointing_func")} for module in modules}
    try:
        eng.model.requires_grad_(False);eng.model.eval()
        if checkpointing:
            if float(getattr(eng.text.config,"attention_dropout",0.))!=0:
                raise ValueError("checkpointing requires zero attention dropout")
            eng.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            eng.text.train()
        else:
            eng.text.gradient_checkpointing_disable()
        yield
        if any(parameter.grad is not None for parameter in parameters):
            raise AssertionError("a frozen parameter accumulated a gradient")
    finally:
        for parameter,flag in zip(parameters,requires): parameter.requires_grad_(flag)
        for module,flag in zip(modules,training): module.training=flag
        for module,values in attributes.items():
            for name,(present,value) in values.items():
                if present:setattr(module,name,value)
                elif hasattr(module,name):delattr(module,name)


def rank_keep(scores,K):
    """Stable score-descending ranking, raster tie break, raster-ordered output."""
    if scores.ndim!=1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be a finite vector")
    if not 1<=K<=scores.numel(): raise ValueError("invalid K")
    return torch.argsort(scores,descending=True,stable=True)[:K].sort().values


def aggregate_scores(magnitude,signed,K):
    """Return four predeclared raw rankings; normalized mass is diagnostic only."""
    if magnitude.ndim!=2 or magnitude.shape[0]!=4 or signed.shape!=magnitude.shape:
        raise ValueError("expected [four_streams,N] scores")
    scores=dict(magnitude_v=magnitude[0],magnitude_joint=magnitude.sum(0),
        signed_v=signed[0],signed_joint=signed.sum(0))
    return scores,{name:rank_keep(value,K) for name,value in scores.items()}


def stream_statistics(magnitude,signed,activation_l2,gradient_l2):
    mass=magnitude.sum(-1)
    normalized=torch.where(mass[:,None]>0,magnitude/mass[:,None].clamp_min(torch.finfo(torch.float32).tiny),torch.zeros_like(magnitude))
    total=mass.sum()
    share=torch.where(total>0,mass/total.clamp_min(torch.finfo(torch.float32).tiny),torch.zeros_like(mass))
    return dict(stream_names=list(STREAM_NAMES),magnitude_mass=mass,magnitude_mass_share=share,
        magnitude_normalized_per_stream=normalized,signed_sum=signed.sum(-1),
        signed_positive_count=(signed>0).sum(-1),signed_negative_count=(signed<0).sum(-1),
        activation_l2=activation_l2,gradient_l2=gradient_l2)


def score_distribution(values):
    values=values.detach().float().cpu()
    squared=values.square().sum()
    return dict(minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),
        zero_count=int((values==0).sum()),negative_count=int((values<0).sum()),
        finite=bool(torch.isfinite(values).all()),sum_abs=float(values.abs().sum()),
        quantile_levels=[0.,.01,.1,.25,.5,.75,.9,.99,1.],
        quantiles=torch.quantile(values,torch.tensor([0.,.01,.1,.25,.5,.75,.9,.99,1.])),
        participation=None if float(squared)==0 else float(values.abs().sum().square()/squared))


def score_full_streams(eng,prep,V,DS,answer,K=256,append_eos=False,
                       checkpointing=False,expected_visual_tokens=1024,max_answer_tokens=128):
    """One full DS-on dense teacher backward produces all four rankings.

    Works even if the caller uses no_grad or inference_mode: leaf cloning and
    the complete differentiable operation happen inside explicit enable_grad.
    The caller must coordinate GPU ownership; this function never retries OOM
    with a shorter image, a shorter gold answer, or a changed objective.
    All returned tensors are detached CPU artifacts.
    """
    if expected_visual_tokens is not None and int(prep["n_vis"])!=expected_visual_tokens:
        raise ValueError("unexpected dense visual-token count")
    if DS is None or len(DS)!=3: raise ValueError("three native DS streams required")
    def sync():
        if V.device.type=="cuda":torch.cuda.synchronize(V.device)
    sync();started=time.perf_counter()
    with torch.inference_mode(False),torch.enable_grad(),frozen_decoder(eng,checkpointing):
        versions=[parameter._version for parameter in eng.model.parameters()]
        leaves=[value.detach().clone().requires_grad_(True) for value in (V,*DS)]
        if any(not bool(torch.isfinite(value).all()) for value in leaves):
            raise ValueError("nonfinite dense visual activation")
        for value in leaves:value.retain_grad()
        sync();tokenization_started=time.perf_counter()
        targets=answer_token_ids(eng.tok,answer,V.device,append_eos,max_answer_tokens)
        sync();gold_tokenization_seconds=time.perf_counter()-tokenization_started
        batch=build_gold_batch(eng,prep,leaves[0],leaves[1:],targets,deepstack=True)
        sync();forward_started=time.perf_counter()
        logits,loss=forward_gold(eng,batch)
        sync();forward_seconds=time.perf_counter()-forward_started
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite teacher logits or loss")
        backward_started=time.perf_counter();loss.backward();sync()
        backward_seconds=time.perf_counter()-backward_started
        if any(value.grad is None for value in leaves):
            raise AssertionError("a native visual stream received no gradient")
        sync();scores_started=time.perf_counter()
        gradients=[value.grad.detach().float() for value in leaves]
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("nonfinite input gradient")
        products=[gradient*activation.detach().float() for gradient,activation in zip(gradients,leaves)]
        magnitude=torch.stack([value.abs().sum(-1) for value in products]).detach().cpu()
        signed=torch.stack([-value.sum(-1) for value in products]).detach().cpu()
        scores,indices=aggregate_scores(magnitude,signed,K)
        statistics=stream_statistics(magnitude,signed,
            torch.tensor([float(value.detach().float().norm()) for value in leaves]),
            torch.tensor([float(value.norm()) for value in gradients]))
        statistics["activation_l2_per_token"]=torch.stack([value.detach().float().norm(dim=-1).cpu() for value in leaves])
        statistics["gradient_l2_per_token"]=torch.stack([value.norm(dim=-1).cpu() for value in gradients])
        statistics["activation_l2_distributions"]={
            name:score_distribution(values) for name,values in
            zip(STREAM_NAMES,statistics["activation_l2_per_token"])}
        statistics["gradient_l2_distributions"]={
            name:score_distribution(values) for name,values in
            zip(STREAM_NAMES,statistics["gradient_l2_per_token"])}
        versions_unchanged=versions==[parameter._version for parameter in eng.model.parameters()]
        if not versions_unchanged:raise AssertionError("model parameter version changed")
        overlaps={}
        for family in ("magnitude","signed"):
            a,b=indices[family+"_v"],indices[family+"_joint"]
            common=int(torch.isin(a,b).sum())
            overlaps[family]=dict(common=common,changed=K-common,jaccard=common/(2*K-common))
        distributions={name:score_distribution(value) for name,value in scores.items()}
        sync();scores_and_ranking_seconds=time.perf_counter()-scores_started
        return dict(n_visual_tokens=int(V.shape[0]),K=K,teacher_deepstack=True,
            teacher_objective="mean_original_gold_token_NLL"+("_plus_explicit_EOS" if append_eos else "_without_added_EOS"),
            append_eos=append_eos,checkpointing=checkpointing,gold_token_ids=targets.detach().cpu(),
            target_token_count=int(targets.numel()),prediction_positions=batch.prediction_positions.detach().cpu(),
            full_prompt_length=batch.full_prompt_length,teacher_input_length=batch.inputs["inputs_embeds"].shape[1],
            mean_answer_nll=float(loss.detach()),per_target_nll=F.cross_entropy(logits.detach().float(),targets,reduction="none").cpu(),
            magnitude_per_stream=magnitude,signed_per_stream=signed,scores=scores,keep_indices=indices,
            stream_statistics=statistics,selection_overlap=overlaps,
            input_sha256=dict(V=tensor_sha256(V),DS=[tensor_sha256(value) for value in DS],
                native_positions=tensor_sha256(batch.inputs["position_ids"])),
            score_distributions=distributions,
            parameter_gradients_none=all(parameter.grad is None for parameter in eng.model.parameters()),
            parameter_versions_unchanged=versions_unchanged,
            timings=dict(gold_tokenization_seconds=gold_tokenization_seconds,
                forward_seconds=forward_seconds,backward_seconds=backward_seconds,
                scores_and_ranking_seconds=scores_and_ranking_seconds,
                total_label_seconds=time.perf_counter()-started),
            finite_gradients=True,proxy_not_actual_deletion_effect=True,deployable=False)


@torch.inference_mode()
def forced_gold_logits(eng,prep,V,DS,targets,deepstack=True):
    """Independent cached native prompt + every required gold-prefix token."""
    targets=targets.to(V.device)
    state=eng.prefill(prep,V,DS,None,deepstack=deepstack)
    rows=[state.logits.reshape(1,-1)]
    positions=[]
    for index,token in enumerate(targets[:-1]):
        cp=state.lk+index
        position=cp+state.rope_delta
        output=eng.text(input_ids=token.reshape(1,1),
            attention_mask=torch.ones(1,cp+1,dtype=torch.long,device=V.device),
            position_ids=torch.full((3,1,1),position,dtype=torch.long,device=V.device),
            past_key_values=state.cache,cache_position=torch.tensor([cp],device=V.device),use_cache=True)
        rows.append(eng.model.lm_head(output.last_hidden_state[:,-1:]).reshape(1,-1))
        positions.append(position)
    logits=torch.cat(rows,dim=0)
    return logits,dict(cache_length=int(state.cache.get_seq_length()),
        expected_cache_length=state.lk+targets.numel()-1,
        actual_visual_count=state.keep_idx.numel(),native_state_n_vis=state.n_vis,
        answer_input_positions=positions,expected_answer_start=int(eng.positions_full(prep)[0].max())+1,
        ds_lengths=state.ds_lengths,keep_idx=state.keep_idx.detach().cpu())


def teacher_native_agreement(eng,prep,V,DS,answer,deepstack=True,append_eos=False,checkpointing=False):
    """Report differentiable/same-shape no-grad/cached gold-prefix differences.

    The batched gold-prefix and cached path have different GEMM shapes on GPU;
    the caller sets a predeclared dtype tolerance rather than asserting hidden
    bit identity. The same-shape comparison is recorded separately.
    """
    with torch.inference_mode(False),torch.enable_grad(),frozen_decoder(eng,checkpointing):
        leaves=[value.detach().clone().requires_grad_(True) for value in (V,*DS)]
        targets=answer_token_ids(eng.tok,answer,V.device,append_eos)
        batch=build_gold_batch(eng,prep,leaves[0],leaves[1:],targets,deepstack=deepstack)
        actual,actual_loss=forward_gold(eng,batch)
        with torch.no_grad():same_shape,same_loss=forward_gold(eng,batch)
        # Detach before releasing the differentiable graph.
        actual=actual.detach().float().cpu();same_shape=same_shape.detach().float().cpu()
        actual_loss=float(actual_loss.detach());same_loss=float(same_loss)
        positions=batch.inputs["position_ids"].detach().cpu()
        target_copy=targets.detach().cpu()
    cached,cache_info=forced_gold_logits(eng,prep,V,DS,target_copy,deepstack)
    cached=cached.float().cpu()
    def compare(other):
        difference=(actual-other).abs()
        return dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()),
            relative_to_reference_max=float(difference.max())/max(float(other.abs().max()),1e-12),
            per_target_max_abs=difference.max(-1).values,
            per_target_argmax_match=actual.argmax(-1)==other.argmax(-1),
            bit_exact=bool(torch.equal(actual,other)))
    cached_loss=float(F.cross_entropy(cached,target_copy,reduction="mean"))
    return dict(deepstack=deepstack,target_ids=target_copy,target_count=target_copy.numel(),
        mean_nll=actual_loss,same_shape_mean_nll=same_loss,cached_mean_nll=cached_loss,
        same_shape_nll_abs_difference=abs(actual_loss-same_loss),cached_nll_abs_difference=abs(actual_loss-cached_loss),
        same_shape=compare(same_shape),cached=compare(cached),cache=cache_info,
        answer_positions_3d=positions[:,:,-(target_copy.numel()-1):] if target_copy.numel()>1 else positions[:,:,:0])


@torch.inference_mode()
def generate_selection(eng,prep,V,DS,keep_idx=None,deepstack=True,max_tokens=32,ignore_eos=False):
    """Ordinary NativeEngine generation; no gradients, model wrappers or masks.

    DS-off changes the evaluation model intentionally. The current shared
    NativeEngine reports state.n_vis=0 when DS is disabled (it derives this
    metadata from its optional DS mask). Physical K is checked using keep_idx,
    input/cache lengths and retained positions instead of hiding that fact.
    """
    if eng.model.training: raise ValueError("generation requires model.eval()")
    if torch.is_grad_enabled():raise AssertionError("generation must be gradient-free")
    keep=checked_keep(keep_idx,int(prep["n_vis"]),V.device)
    state=eng.prefill(prep,V,DS,keep,deepstack=deepstack)
    ids,text=eng.decode(state,max_tokens,ignore_eos=ignore_eos)
    expected_sequence=eng.keep_sequence(prep,keep)
    full_positions=eng.positions_full(prep)[0]
    if not torch.equal(state.keep_idx,keep) or not torch.equal(state.keep_seq,expected_sequence):
        raise AssertionError("native retained sequence differs")
    if not torch.equal(state.positions_prefill,full_positions[:,:,expected_sequence]):
        raise AssertionError("native 3D positions changed")
    if state.ds_lengths!=([keep.numel()]*3 if deepstack else None):
        raise AssertionError("native DS stream lengths changed")
    if state.cache.get_seq_length()!=state.lk+len(state.decode_positions):
        raise AssertionError("generation cache bookkeeping failed")
    if state.decode_positions and state.decode_positions[0]!=int(full_positions.max())+1:
        raise AssertionError("decode position drift")
    return dict(text=text,token_ids=ids,truncated=len(ids)>=max_tokens,deepstack=deepstack,
        actual_visual_count=keep.numel(),native_state_n_vis=state.n_vis,ds_lengths=state.ds_lengths,
        keep_idx=keep.cpu(),positions_sha256=tensor_sha256(state.positions_prefill),
        prefill_length=state.lk,cache_length=int(state.cache.get_seq_length()),decode_positions=state.decode_positions,
        gradients_enabled=False)


@torch.inference_mode()
def evaluate_rankings(eng,prep,V,DS,oracle,max_tokens=32,include_random=True,random_seed=2026110913):
    """Eight diagnostic arms plus dense on/off and optional fixed-seed random.

    Scores are not recomputed for DS-off evaluation: that contrast deliberately
    tests the SAME teacher-on selected sets under a changed injection policy.
    """
    if oracle["n_visual_tokens"]!=V.shape[0]:raise ValueError("oracle/image token count mismatch")
    if oracle["input_sha256"]["V"]!=tensor_sha256(V) or oracle["input_sha256"]["DS"]!=[tensor_sha256(value) for value in DS]:
        raise ValueError("oracle features differ from evaluation features")
    selections={name:oracle["keep_indices"][name] for name in SCORE_NAMES}
    selections["full"]=torch.arange(V.shape[0])
    if include_random:
        selections["random"]=torch.randperm(V.shape[0],generator=torch.Generator().manual_seed(random_seed))[:oracle["K"]].sort().values
    result={}
    for name,keep in selections.items():
        for enabled in (True,False):
            arm=name+("_ds_on" if enabled else "_ds_off")
            result[arm]=generate_selection(eng,prep,V,DS,keep,enabled,max_tokens)
    return result
