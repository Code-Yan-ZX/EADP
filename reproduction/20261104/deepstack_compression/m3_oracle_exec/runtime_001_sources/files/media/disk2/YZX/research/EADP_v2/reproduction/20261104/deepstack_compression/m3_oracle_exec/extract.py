"""Author-native full vision, prompt positions and actual K256 greedy order.

Caller owns the preverified DEV row and input identity function. No answers,
predictions, file contents or generation are accessed by this module.
"""

def extract(torch,wrapper,pruner_module,row,validate_input):
    if row.get('split')!='DEV' or wrapper.visual_token_num!=256 or wrapper.pruner.visual_token_num!=256:
        raise ValueError('expected DEV K256 model')
    wrapper.model.eval();wrapper.model.requires_grad_(False)
    if any(p.requires_grad for p in wrapper.model.parameters()) or any(m.training for m in wrapper.model.modules()):
        raise ValueError('model not frozen/eval')
    with torch.no_grad():
        messages=wrapper._build_messages(row['message'],dataset=row['task']);inputs=wrapper._processor_inputs(messages)
        validate_input(row,inputs)
        ids=inputs['input_ids'];mask=inputs['attention_mask']
        if ids.shape[0]!=1 or not torch.all(mask==1):raise ValueError('expected unpadded batch one')
        points=(ids[0]==wrapper.model.config.image_token_id).nonzero(as_tuple=True)[0]
        if points.numel()!=1024 or not torch.equal(points,torch.arange(points[0],points[0]+1024,device=ids.device)):
            raise ValueError('expected contiguous 1024 image tokens')
        start=int(points[0]);main,ds=wrapper.model.model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
        if len(main)!=1 or len(ds)!=3 or any(x.shape!=main[0].shape for x in ds):raise ValueError('invalid full visual streams')
        streams=[main[0].detach()]+[x.detach() for x in ds]
        position,delta=wrapper.model.model.get_rope_index(ids,inputs['image_grid_thw'],None,attention_mask=mask)
        sequence=wrapper._get_instruction_sequence_embedding(row['message'],dataset=row['task']);n=inputs['image_grid_thw'].shape[0]
        if n!=1:raise ValueError('expected one image')
        original=pruner_module._greed_select_impl;calls=[]
        def capture(*args,**kwargs):
            output=original(*args,**kwargs);calls.append((output[0].detach().clone().reshape(-1),output[1].detach().clone().reshape(-1)));return output
        pruner_module._greed_select_impl=capture
        try:chosen,splits=wrapper.pruner(streams[0],sequence.mean(dim=1).expand(n,-1),sequence.expand(n,-1,-1),inputs['image_grid_thw'])
        finally:pruner_module._greed_select_impl=original
        if len(calls)!=1 or splits!=[256] or calls[0][1].tolist()!=[256]:raise ValueError('actual greedy call count/budget mismatch')
        order=calls[0][0].tolist()
        if len(order)!=256 or len(set(order))!=256 or any(i<0 or i>=1024 for i in order):raise ValueError('invalid actual greedy order')
        idx=torch.tensor(sorted(order),device=streams[0].device,dtype=torch.long)
        if not torch.equal(chosen,streams[0].index_select(0,idx)):raise ValueError('actual greedy pruned features mismatch')
    return {'input_ids':ids.detach().clone(),'processor':{k:inputs[k].detach().clone() for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
            'start':start,'streams':streams,'positions':position.detach().clone(),'delta':int(delta.item()),'greedy_order':order}
