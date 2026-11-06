"""Per-question E1 P3-D1 identity gate, without reading files or references.

Caller must separately validate the E1 completion/reviewer/source/terminal
chain and prove that raw_record is the unique line in the accepted journal.
"""
import hashlib,json

def meta(x):
    import torch
    x=x.detach().contiguous().cpu()
    return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def validate(torch,model,row,full,raw_record,provenance,expected_scalars,expected_eos,on_logit_check):
    if (not isinstance(provenance,dict) or provenance.get('accepted_unique_line') is not True or
        any(not isinstance(provenance.get(k),str) or len(provenance[k])!=64 for k in ('acceptance_sha256','inputs_sha256','journal_sha256','review_sha256','weights_identity_sha256','plan_sha256'))):
        raise ValueError('accepted E1 provenance required')
    rec=json.loads(raw_record);ids=full['input_ids'];order=full['greedy_order'];spatial=sorted(order)
    if len(spatial)!=256 or len(set(spatial))!=256:raise ValueError('invalid baseline selection')
    start=full['start'];device=ids.device;pos=full['positions'];length=ids.shape[1]
    retained=torch.tensor(list(range(start))+[start+i for i in spatial]+list(range(start+1024,length)),device=device,dtype=torch.long)
    index=torch.tensor(spatial,device=full['streams'][0].device,dtype=torch.long)
    streams=[x.index_select(0,index) for x in full['streams']]
    with torch.no_grad():
        embeddings=model.get_input_embeddings()(ids);prefill=torch.cat((embeddings[:,:start],streams[0].unsqueeze(0),embeddings[:,start+1024:]),dim=1)
    mask=full['processor']['attention_mask'].index_select(1,retained)
    injected=torch.zeros_like(mask,dtype=torch.bool);injected[:,start:start+256]=True
    expected={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'message_sha256':row['message_sha256'],
        'arm':'P3-D1','forced':False,'processor':{k:meta(v) for k,v in full['processor'].items()},
        'image_start':start,'original_length':length,'ordered_ids':order,'spatial_ids':spatial,'retained_ids':retained.tolist(),
        'original_position_ids':pos.tolist(),'full_rope_delta':[[full['delta']]],'visual_sha256':meta(full['streams'][0]),
        'pruned_sha256':meta(streams[0]),'deepstack_full_sha256':[meta(x) for x in full['streams'][1:]],
        'prefill_embed_sha256':meta(prefill),'attention_mask_sha256':meta(mask),'prefill_position_ids':pos.index_select(-1,retained).tolist(),
        'rope_delta':[[full['delta']+768]],'injected_mask':injected.tolist(),
        'injected_deepstack':[meta(x) for x in streams[1:]],'selected_deepstack':[meta(x) for x in streams[1:]],
        'generation_scalars':expected_scalars,'eos_ids':expected_eos}
    for key,value in expected.items():
        if rec.get(key)!=value:raise ValueError('E1 baseline identity mismatch: '+key)
    # Execute the actual current K256/DS111 baseline prefill *before* the
    # caller opens its reference. Equality is bytewise to the accepted E1
    # first-step logit metadata; no after-the-fact tolerance is permitted.
    with torch.no_grad():
        out=model.model.language_model(inputs_embeds=prefill.clone(),attention_mask=mask.clone(),
            position_ids=pos.index_select(-1,retained).clone(),visual_pos_masks=injected.clone(),
            deepstack_visual_embeds=[x.clone() for x in streams[1:]],use_cache=True,
            cache_position=torch.arange(mask.shape[1],device=device))
        first=model.lm_head(out.last_hidden_state[:,-1,:]).detach()
        actual=meta(first);expected_logits=rec.get('logits',[None])[0]
        matched=bool(torch.isfinite(first).all()) and actual==expected_logits
        on_logit_check({'actual':actual,'expected':expected_logits,'identical':matched})
        if not matched:raise ValueError('E1 baseline first logits mismatch')
    if (not isinstance(rec.get('output_ids'),list) or not isinstance(rec.get('answer'),str) or
        not (rec.get('eos') is True or rec.get('truncated') is True)):
        raise ValueError('baseline generation endpoint invalid')
    return {'qid':str(row['qid']),'arm':'P3-D1','verified':True,'record_sha256':hashlib.sha256(raw_record).hexdigest(),
            'acceptance_sha256':provenance['acceptance_sha256'],'inputs_sha256':provenance['inputs_sha256'],
            'journal_sha256':provenance['journal_sha256'],'review_sha256':provenance['review_sha256'],
            'weights_identity_sha256':provenance['weights_identity_sha256'],'plan_sha256':provenance['plan_sha256']}
