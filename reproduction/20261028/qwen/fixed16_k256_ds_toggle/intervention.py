"""Draft prefill toggle, consumes only actual captured author-path tensors."""
from contracts import compare_selection, mask_span

# Required capture is from this same native invocation, never reconstructed from
# the old cache: ordered_ids, spatial_ids, visual_features, pruner_input,
# pruner_output, split_sizes, deepstack, processor_input_ids, processor_mask,
# original_token_embeddings, generation_kwargs, original_prefill, rope_deltas.
def propose(torch, baseline, current, actual_prefill):
    """Return untouched kwargs on any failure; only two prefill fields may change.

    This function synchronizes actual GPU tensors to compare prerequisites. It
    is diagnostic instrumentation, not part of the author's timing or method.
    The caller must save the native response before judging an error.
    """
    try:
        def eq(a,b):
            if a is None or b is None:return a is b
            return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.device==b.device and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        def require(condition,message):
            if not condition:raise ValueError(message)
        for x in (baseline,current):
            ordered=x['ordered_ids'].detach().cpu().tolist();spatial=x['spatial_ids'].detach().cpu().tolist()
            compare_selection(ordered,spatial)
            require(x['split_sizes']==[256],'pruner split must be [256]')
            require(x['visual_features'].ndim==2 and x['visual_features'].shape[0]==1024,'visual input must contain 1024 spatial rows')
            require(eq(x['pruner_input'],x['visual_features']),'actual pruner input does not equal tower features')
            require(eq(x['pruner_output'],x['visual_features'].index_select(0,x['spatial_ids'])),'actual pruner output differs from selected tower rows')
            require(len(x['deepstack'])==3,'expected three real DeepStack outputs')
            require(all(z.shape==x['visual_features'].shape and z.dtype==x['visual_features'].dtype and z.device==x['visual_features'].device for z in x['deepstack']),'DeepStack shape, dtype, or device mismatch')
            ids=x['processor_input_ids'];require(ids.ndim==2 and ids.shape[0]==1,'batch must be one')
            ids_list=ids[0].detach().cpu().tolist();mask_list=mask_span(ids_list,x['image_token_id'])
            start=mask_list.index(True); embedding=x['generation_kwargs']['inputs_embeds'];attn=x['generation_kwargs']['attention_mask']
            original=x['original_token_embeddings'];original_attn=x['processor_mask']
            require(original.shape[:2]==ids.shape and original_attn.shape==ids.shape,'source embedding/mask shape mismatch')
            require(embedding.shape[0]==1 and embedding.shape[1]==len(mask_list),'compacted input length mismatch')
            require(eq(embedding[:,start:start+256],x['pruner_output'].unsqueeze(0)),'compacted image embedding differs from real selected pruner output')
            require(eq(embedding[:,:start],original[:,:start]) and eq(embedding[:,start+256:],original[:,start+1024:]),'nonvisual embedding differs from author source')
            expected_attn=torch.cat((original_attn[:,:start],torch.ones_like(original_attn[:,:256]),original_attn[:,start+1024:]),dim=1)
            require(eq(attn,expected_attn),'compacted attention differs from author construction')
        for k in ('ordered_ids','spatial_ids','visual_features','pruner_input','pruner_output','processor_input_ids','processor_mask','processor_pixel_values','processor_image_grid_thw','original_token_embeddings'):
            require(eq(baseline[k],current[k]),'paired actual capture differs: '+k)
        for j,(a,b) in enumerate(zip(baseline['deepstack'],current['deepstack'])):require(eq(a,b),'paired visual DeepStack differs: '+str(j))
        ga,gb=baseline['generation_kwargs'],current['generation_kwargs'];require(set(ga)==set(gb),'generation kwargs keys differ')
        scalars=dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
        require(set(ga)==set(scalars)|{'inputs_embeds','attention_mask'} and all(ga[k]==v for k,v in scalars.items()),'unexpected author generation configuration')
        for k in ga:
            require(eq(ga[k],gb[k]) if isinstance(ga[k],torch.Tensor) else ga[k]==gb[k],'generation kwarg differs: '+k)
        require(current['image_token_id']==baseline['image_token_id'],'image token ID differs')
        require(eq(baseline['original_prefill'].get('visual_pos_masks'),None) and baseline['original_prefill'].get('deepstack_visual_embeds') is None,'baseline legacy prefill already has visual fields')
        require(eq(actual_prefill.get('inputs_embeds'),gb['inputs_embeds']) and eq(actual_prefill.get('attention_mask'),gb['attention_mask']),'prefill differs from actual generation input')
        fields=('inputs_embeds','position_ids','attention_mask','cache_position')
        for k in fields:
            require(eq(baseline['original_prefill'].get(k),actual_prefill.get(k)),'prefill differs: '+k)
        require(eq(baseline['rope_deltas'],current['rope_deltas']),'prefill rope delta differs')
        require(actual_prefill.get('visual_pos_masks') is None and actual_prefill.get('deepstack_visual_embeds') is None,'legacy prefill already has visual fields')
        mask_list=mask_span(current['processor_input_ids'][0].detach().cpu().tolist(),current['image_token_id'])
        embeddings=actual_prefill['inputs_embeds'];mask=torch.tensor([mask_list],dtype=torch.bool,device=embeddings.device)
        require(mask.shape==embeddings.shape[:2] and int(mask.sum().item())==256,'compacted mask mismatch')
        selected=[x.index_select(0,current['spatial_ids']) for x in current['deepstack']]
        require(all(y.shape==(256,embeddings.shape[-1]) and y.dtype==embeddings.dtype and y.device==embeddings.device for y in selected),'selected DeepStack geometry mismatch')
        result=dict(actual_prefill);result['visual_pos_masks']=mask;result['deepstack_visual_embeds']=selected
        return result,True,None
    except Exception as error:
        return actual_prefill,False,repr(error)
