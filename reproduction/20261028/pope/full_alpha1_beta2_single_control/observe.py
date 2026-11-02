"""Keep native references, serialize only after the original generate returns."""
import inspect
import json
from pathlib import Path
import common as c


def install(model, tokenizer, questions, anchors, loaded):
    import torch
    original={name:getattr(model,name) for name in ['generate','encode_images','greed_select']}
    for name,fn in original.items():
        path=inspect.getfile(inspect.unwrap(fn.__func__))
        assert c.sha(path)==loaded['source_sha256'][path]
    refs={};position=0;anchor_checks=[]
    path=c.OUT/'runtime_observations.jsonl';stream=path.open('x')
    def encode(*args,**kwargs):
        refs['encode_calls']=refs.get('encode_calls',0)+1
        refs['encode_guidance']=kwargs.get('texts');refs['split_sizes']=kwargs.get('split_sizes')
        refs['actual_parameters']=[model.alpha,model.beta,model.visual_token_num]
        value=original['encode_images'](*args,**kwargs);refs['encoded']=value;return value
    def greedy(*args,**kwargs):
        refs['greedy_calls']=refs.get('greedy_calls',0)+1
        refs['importance']=args[0];refs['greedy_kwargs']=kwargs
        value=original['greed_select'](*args,**kwargs);refs['greedy']=value;return value
    def generate(*args,**kwargs):
        nonlocal position
        refs.clear();input_ids=args[0] if args else kwargs.get('inputs',kwargs.get('input_ids'))
        result=original['generate'](*args,**kwargs)
        # No content reads, synchronization, clones, changes of numerical
        # operations, or CPU conversions were performed by the hooks above.
        features,mask=refs['encoded'];ordered,quotas=refs['greedy'];output,retained=result
        question=questions[position]
        tokens=input_ids.detach().cpu().tolist()[0];out=output.detach().cpu().tolist()[0]
        quotas_list=quotas.detach().cpu().tolist();quota=quotas_list[0]
        record=dict(position=position,question_id=question['question_id'],image=question['image'],
            full_prompt_sha256=c.text_sha(question['text']),actual_guidance_sha256=c.text_sha(kwargs['texts']),
            actual_alpha_beta_K=refs['actual_parameters'],calls=dict(encode=refs['encode_calls'],greedy=refs['greedy_calls']),
            input_token_ids=tokens,input_ids_sha256=c.int64_sha(tokens),input_ids_dtype=str(input_ids.dtype),input_ids_shape=list(input_ids.shape),
            quotas=quotas_list,mask_shape=list(mask.shape),effective_mask_indices=torch.nonzero(mask[0],as_tuple=False).flatten().detach().cpu().tolist(),
            effective_ordered_prefix=ordered[0,:quota].detach().cpu().tolist(),
            projected_feature_shape=list(features.shape),projected_feature_dtype=str(features.dtype),features_finite=bool(torch.isfinite(features).all().item()),
            importance_shape=list(refs['importance'].shape),importance_finite=bool(torch.isfinite(refs['importance']).all().item()),
            returned_retained_tokens=int(retained),output_token_ids=out,text=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip(),
            new_tokens=len(out)-1,EOS_present=2 in out,generation_horizon=128,post_generation_tensor_reads_only=True)
        # Preserve every native output before checking a panel anchor or raising.
        stream.write(json.dumps(record)+'\n');stream.flush()
        c.check_observation(record,question,position)
        assert refs['encode_guidance']==kwargs['texts']==c.guidance(question['text']) and refs['split_sizes'] is None
        assert refs['greedy_kwargs']==dict(K=1,strategy='importance_based')
        assert {k:kwargs[k] for k in ['do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache']}==dict(
            do_sample=False,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)
        key=(str(question['question_id']),question['text'])
        if key in anchors:
            c.check_anchor(record,anchors[key]);anchor_checks.append(question['question_id'])
        position+=1
        return result
    model.encode_images=encode;model.greed_select=greedy;model.generate=generate
    def finish():
        stream.close();assert position==8910 and len(anchor_checks)==len(set(anchor_checks))==16
        return dict(n_runtime_records=position,panel_beta2_anchor_checks=16,
            raw_native_outputs_saved_before_anchor_checks=True,post_generation_tensor_reads_only=True)
    return finish
