"""One full-visual synthetic-target gradient diagnostic; never scores answers."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import numpy as np
from local_imports import local
contract=local('contract');tensor_meta=contract.tensor_meta;check_actual_inputs=contract.check_actual_inputs
check_prefix_positions=contract.check_prefix_positions;compare_logits=contract.compare_logits

def _native_logits(torch,ref,device):
    with np.load(ref['arrays']['path'],allow_pickle=False) as arrays:
        data=arrays['prefill_logits'].copy()
    meta=ref['prefill_logits']
    if meta['dtype']!='torch.bfloat16' or data.dtype!=np.uint8:raise ValueError('native logits format invalid')
    native=torch.from_numpy(data).view(torch.bfloat16).reshape(meta['shape'])
    if tensor_meta(native)!=meta:raise ValueError('native logits bytes do not match accepted metadata')
    return native.to(device)

def run(torch,wrapper,row,ref,forced_ref,teacher,saliency,plan,on_event,archive):
    if torch.is_inference_mode_enabled():raise ValueError('autograd cannot run inside inference_mode')
    model=wrapper.model;model.eval();model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('model not fully frozen/eval')
    clock=time.monotonic();on_event('processing_started')
    with torch.no_grad():
        messages=wrapper._build_messages(row['message'],dataset=row['task'])
        inputs=wrapper._processor_inputs(messages)
        ids=inputs['input_ids'];mask=(ids[0]==model.config.image_token_id).nonzero(as_tuple=True)[0]
        if ids.shape[0]!=1 or mask.numel()!=1024 or not torch.equal(mask,torch.arange(mask[0],mask[0]+1024,device=ids.device)):
            raise ValueError('processor lacks a contiguous 1024 image-token segment')
        start=int(mask[0]);main_tuple,streams=model.model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
        if len(main_tuple)!=1:raise ValueError('expected one image feature group')
        main=main_tuple[0].detach();streams=[x.detach() for x in streams]
        positions,delta=model.model.get_rope_index(ids,inputs['image_grid_thw'],None,attention_mask=inputs['attention_mask'])
        check_actual_inputs(torch,inputs,main,streams,positions,delta,ref)
        prompt=model.get_input_embeddings()(ids).detach().clone();prompt[:,start:start+1024]=main
        if tensor_meta(prompt)!=ref['prefill_embeds'] or tensor_meta(inputs['attention_mask'])!=ref['prefill_mask']:
            raise ValueError('full actual prefill differs from accepted native')
        on_event('native_prefill_started')
        visual_mask=torch.zeros_like(inputs['attention_mask'],dtype=torch.bool);visual_mask[:,start:start+1024]=True
        if visual_mask.detach().cpu().tolist()!=ref['injected_mask']:
            raise ValueError('actual full DS injection mask differs from native')
        o=model.model.language_model(inputs_embeds=prompt.clone(),position_ids=positions,
            attention_mask=inputs['attention_mask'],visual_pos_masks=visual_mask,deepstack_visual_embeds=streams,use_cache=False)
        native_actual=model.lm_head(o.last_hidden_state[:,-1,:]).detach();del o
        native_ref=_native_logits(torch,ref,native_actual.device)
        native_comparison=compare_logits(native_actual,native_ref,plan['native_first_logits_max_abs_tolerance'])
    on_event('backward_started')
    captured={};length=ids.shape[1];targets=torch.tensor([plan['target_token_ids']],device=ids.device,dtype=torch.long)
    with ExitStack() as stack:
        def prefill(_m,_args,kw):
            if captured.get('prefill_seen'):raise ValueError('teacher called language model more than once')
            if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise ValueError('teacher forward lacks autograd')
            if kw.get('use_cache') is not False or len(kw.get('deepstack_visual_embeds',[]))!=3:
                raise ValueError('teacher does not use the required three uncached streams')
            actual_ds=kw['deepstack_visual_embeds']
            if any(not x.requires_grad or not x.is_leaf for x in actual_ds):raise ValueError('DS inputs are not gradient leaves')
            if not torch.equal(kw['visual_pos_masks'][:,:length],visual_mask) or kw['visual_pos_masks'][:,length:].any():
                raise ValueError('teacher visual mask mismatch')
            if not torch.equal(kw['position_ids'][:,:,:length],positions):raise ValueError('teacher prompt position mismatch')
            check_prefix_positions(kw['position_ids'],length,int(delta.item()),forced_ref,len(plan['target_token_ids'])-1)
            if not torch.equal(kw['inputs_embeds'][:,:length],prompt):raise ValueError('teacher prefill embeddings mismatch')
            captured.update(prefill_seen=True,streams=list(actual_ds),positions=kw['position_ids'].detach().clone(),
                            visual_mask=kw['visual_pos_masks'].detach().clone(),autograd_enabled=True)
        def logits(_m,_args,out):
            if 'teacher_first_logits' in captured:raise ValueError('teacher lm head called more than once')
            captured['teacher_first_logits']=out[:,0,:].detach().clone()
        h=model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True);stack.callback(h.remove)
        h=model.lm_head.register_forward_hook(logits);stack.callback(h.remove)
        loss,leaves,grads=teacher.loss_and_gradients(model,ids,start,main,streams,positions,int(delta.item()),targets)
    if not captured.get('prefill_seen') or 'teacher_first_logits' not in captured or len(leaves)!=4 or len(grads)!=4:
        raise ValueError('teacher forward or four gradients absent')
    if not torch.isfinite(captured['teacher_first_logits']).all():raise ValueError('nonfinite teacher first logits')
    if any(g.shape!=(1024,4096) or not torch.isfinite(g).all() or not torch.count_nonzero(g) for g in grads):
        raise ValueError('gradient shape/nonfinite/zero failure')
    if any(not torch.equal(a,b) for a,b in zip(captured['streams'],leaves[1:])):
        raise ValueError('actual DS leaves changed')
    teacher_comparison=compare_logits(captured['teacher_first_logits'],native_ref,plan['native_first_logits_max_abs_tolerance'])
    utility=saliency.utilities(leaves,grads)
    result={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'target_ids':plan['target_token_ids'],
        'target_source':'synthetic fixed IDs, no label or prediction','gt_read':False,'scoring':False,'generation':False,
        'model_fully_frozen':True,'model_eval':True,'gradient_autograd_enabled':captured['autograd_enabled'],
        'native_first_logits_comparison':native_comparison,
        'teacher_first_logits_comparison':teacher_comparison,
        'native_actual_logits':tensor_meta(native_actual),'native_reference_logits':tensor_meta(native_ref),
        'teacher_first_logits':tensor_meta(captured['teacher_first_logits']),
        'teacher_first_vs_native_max_abs':float((captured['teacher_first_logits'].float()-native_ref.float()).abs().max().item()),
        'teacher_prefix_positions':captured['positions'][:,:,-(len(plan['target_token_ids'])-1):].cpu().tolist(),
        'native_rope_delta':int(delta.item()),'prompt_length':length,'image_start':start,
        'processor':{k:tensor_meta(inputs[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
        'features':[tensor_meta(x) for x in leaves],'gradients':[tensor_meta(x) for x in grads],
        'gradient_nonzero_counts':[int(torch.count_nonzero(x).item()) for x in grads],
        'synthetic_loss':float(loss.item()),'utility':[{'length':len(x),'min':min(x),'max':max(x)} for x in utility],
        'elapsed_seconds':time.monotonic()-clock}
    arrays={**{'feature'+str(i):x for i,x in enumerate(leaves)},**{'gradient'+str(i):x for i,x in enumerate(grads)},
            'native_actual_logits':native_actual,'native_reference_logits':native_ref,'teacher_first_logits':captured['teacher_first_logits']}
    result['arrays']=archive(arrays);on_event('gradients_saved')
    return result
