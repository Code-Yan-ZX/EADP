"""CPU synthetic tests for full K256 verifier; no model, GPU, labels or scores."""
import copy
import unittest
import verify_generation as v


def m(dtype,shape,digit):return dict(dtype=dtype,shape=shape,sha256=digit*64)
def fixture(arm):
    ids=[8,9]+[99]*1024+[10,11];n=len(ids)-768
    proc=dict(input_ids=dict(dtype='torch.int64',shape=[1,len(ids)],sha256=v.ids_sha(ids)),
              attention_mask=m('torch.int64',[1,len(ids)],'a'),pixel_values=m('torch.bfloat16',[4096,1536],'b'),
              image_grid_thw=m('torch.int64',[1,3],'c'))
    embeds=m('torch.bfloat16',[1,n,4096],'d');attention=m('torch.int64',[1,n],'e')
    orig=dict(inputs_embeds=embeds,position_ids=m('torch.int64',[3,1,n],'f'),attention_mask=attention,
              cache_position=m('torch.int64',[n],'1'),visual_pos_masks=None,deepstack_0=None,deepstack_1=None,deepstack_2=None)
    final=copy.deepcopy(orig);mask=[False,False]+[True]*256+[False,False]
    layers=[m('torch.bfloat16',[1024,4096],str(j+2)) for j in range(3)]
    selected=[m('torch.bfloat16',[256,4096],str(j+5)) for j in range(3)]
    if arm=='ds_on':
        final['visual_pos_masks']=v.bool_meta(mask)
        for i in range(3):final['deepstack_'+str(i)]=selected[i]
    scalars=dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
    step=dict(position_ids=m('torch.int64',[3,1,1],'8'),cache_position=m('torch.int64',[1],'9'),
              rope_deltas=m('torch.int64',[1,1],'a'),visual_mask_present=False,deepstack_present=False)
    item=dict(position=7,index='abc',processor_expected=proc)
    pred=dict(position=7,index='abc',arm=arm,response='answer',native_failure=None)
    obs=dict(position=7,index='abc',arm=arm,native_failure=None,observer_errors=[],selection_errors=[],actual_processor=proc,
        input_ids=ids,image_mask=mask,ordered_ids=list(range(255,-1,-1)),spatial_ids=list(range(256)),quotas=[256],pruner_split_sizes=[256],
        vision_calls=1,pruner_calls=1,processor_calls=1,generate_calls=1,selection_calls=1,
        vision_features=m('torch.bfloat16',[1024,4096],'b'),pruner_input=m('torch.bfloat16',[1024,4096],'b'),
        selected_vision_features=m('torch.bfloat16',[256,4096],'c'),pruner_output=m('torch.bfloat16',[256,4096],'c'),
        visual_deepstack=layers,selected_visual_deepstack=selected,original_token_embeddings=m('torch.bfloat16',[1,len(ids),4096],'0'),
        reconstructed_inputs_embeds=embeds,reconstructed_attention_mask=attention,prefill_original=orig,prefill_proposed=copy.deepcopy(final),prefill_final=final,
        intervention_attempts=int(arm=='ds_on'),intervention_applied=arm=='ds_on',intervention_error=None,
        generate_non_tensor_kwargs=scalars,generate_kwargs_keys=sorted(set(scalars)|{'inputs_embeds','attention_mask'}),
        generate_tensor_kwargs=dict(inputs_embeds=embeds,attention_mask=attention),rope_deltas=m('torch.int64',[1,1],'a'),
        output_ids=[7,8,9],native_redecode='answer',continuation_steps=2,llm_calls=3,continuation=[copy.deepcopy(step),copy.deepcopy(step)])
    return item,pred,obs

class Tests(unittest.TestCase):
    def setUp(self):
        self.i,self.p,self.off=fixture('ds_off');_,self.q,self.on=fixture('ds_on')
    def check(self,obs,arm):v.check_observation(self.i,self.p if arm=='ds_off' else self.q,obs,arm,99,1000)
    def test_valid_both_and_pair(self):
        self.check(self.off,'ds_off');self.check(self.on,'ds_on');self.assertTrue(v.check_pair(self.off,self.on)['output_ids_equal'])
    def test_greedy_order_invalid_or_spatial_not_sorted_rejected(self):
        for field in ('ordered_ids','spatial_ids'):
            x=copy.deepcopy(self.on);x[field][0]=x[field][1]
            with self.assertRaises(AssertionError):self.check(x,'ds_on')
    def test_wrong_mask_and_all_three_selected_layers_rejected(self):
        x=copy.deepcopy(self.on);x['image_mask'][0]=True
        with self.assertRaises(AssertionError):self.check(x,'ds_on')
        for j in range(3):
            x=copy.deepcopy(self.on);x['selected_visual_deepstack'][j]['sha256']='e'*64
            with self.assertRaises(AssertionError):self.check(x,'ds_on')
    def test_reconstruction_and_pruner_summary_must_match(self):
        for field in ('selected_vision_features','reconstructed_inputs_embeds','reconstructed_attention_mask'):
            x=copy.deepcopy(self.on);x[field]['sha256']='0'*64
            with self.assertRaises(AssertionError):self.check(x,'ds_on')
    def test_pair_source_and_positions_must_agree(self):
        for field in ('actual_processor','ordered_ids','rope_deltas'):
            x=copy.deepcopy(self.on)
            if isinstance(x[field],list):x[field][0]+=1
            else:x[field]['sha256']='e'*64
            with self.assertRaises(AssertionError):v.check_pair(self.off,x)
        x=copy.deepcopy(self.on);x['prefill_original']['position_ids']['sha256']='e'*64
        with self.assertRaises(AssertionError):v.check_pair(self.off,x)
    def test_continuation_missing_and_present_ds_rejected(self):
        x=copy.deepcopy(self.on);x['continuation'][0]['position_ids']=None
        with self.assertRaises(AssertionError):self.check(x,'ds_on')
        x=copy.deepcopy(self.on);x['continuation'][0]['deepstack_present']=True
        with self.assertRaises(AssertionError):self.check(x,'ds_on')
    def test_continuation_difference_reported_only_on_same_prefix(self):
        x=copy.deepcopy(self.on);x['continuation'][0]['position_ids']['sha256']='e'*64
        self.check(x,'ds_on');r=v.check_pair(self.off,x);self.assertEqual(r['same_prefix_continuation_differing_steps']['position_ids'],[0])
        x['output_ids'][0]=42;r=v.check_pair(self.off,x);self.assertEqual(r['common_prefix_continuation_steps'],0)
    def test_ordered_is_exact_permutation_preserving_payload_and_ids(self):
        items=[dict(position=i,index=str(i)) for i in range(3)]
        source=[dict(position=i,index=str(i),response='answer-'+str(i)) for i in (2,0,1)]
        ordered=sorted(source,key=lambda r:r['position']);v.check_ordered(items,source,ordered)
        for field in ('response','index'):
            changed=copy.deepcopy(ordered);changed[1][field]='tampered'
            with self.assertRaises(AssertionError):v.check_ordered(items,source,changed)
        changed=copy.deepcopy(ordered);changed[1]['position']=2
        with self.assertRaises(AssertionError):v.check_ordered(items,source,changed)
    def test_eos_and_limit(self):
        self.assertTrue(v.ending([3,9],[9])['EOS_at_end']);self.assertFalse(v.ending([9,3],[9])['EOS_at_end'])
        self.assertTrue(v.ending([3]*2048,[9])['at_max_tokens']);self.assertFalse(v.ending([3]*2048,[9])['EOS_present'])
    def test_real_saved_fixed16_archive_anchor_and_tamper(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        import common as c
        prior=c.FIXED/'execution';key='00_ds_on';rec=c.read(prior/(key+'.json'))
        with np.load(prior/(key+'.npz'),allow_pickle=False) as data, tempfile.TemporaryDirectory() as tmp:
            here=Path(tmp);d=here/'execution'/'anchors';d.mkdir(parents=True)
            for ext in ('.json','.npz'):(d/(key+ext)).symlink_to(prior/(key+ext))
            meta=rec['arrays'];o=dict(position=rec['position'],index=rec['index'])
            for name in ('pruner_input','pruner_output','original_token_embeddings','rope_deltas'):o[name]=meta[name]
            o['vision_features']=meta['visual_features']
            o['actual_processor']={k:meta['processor_'+k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
            o['visual_deepstack']=[meta['visual_deepstack_'+str(j)] for j in range(3)]
            o['generate_tensor_kwargs']={k:meta['generate_'+k] for k in ('inputs_embeds','attention_mask')}
            for name,prefix in (('prefill_original','original_prefill_'),('prefill_proposed','proposed_prefill_'),('prefill_final','final_prefill_')):
                o[name]={k:meta.get(prefix+k) for k in v.FIELDS}
            o.update(ordered_ids=data['ordered_ids'][0].tolist(),spatial_ids=data['spatial_ids'].tolist(),quotas=data['quotas'].tolist(),output_ids=data['output_ids'][0].tolist(),generate_kwargs_keys=rec['generate_kwargs_keys'],generate_non_tensor_kwargs=rec['generate_non_tensor_kwargs'])
            o['continuation']=[]
            for j,presence in enumerate(rec['continuation_presence']):
                o['continuation'].append({**presence,**{k:meta['continuation_%04d_%s'%(j,k)] for k in ('position_ids','cache_position','rope_deltas')}})
            def bf(x):return dict(dtype='torch.bfloat16',shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest())
            spatial=np.array(o['spatial_ids']);o['selected_vision_features']=bf(data['visual_features'][spatial])
            o['selected_visual_deepstack']=[bf(data['visual_deepstack_'+str(j)][spatial]) for j in range(3)]
            ids=data['processor_input_ids'][0].tolist();start=ids.index(c.read(Path(c.read(c.OLD/'run_plan.json')['recipe']['model_path'])/'config.json')['image_token_id'])
            o['image_mask']=[start<=i<start+256 for i in range(len(ids)-768)]
            o['reconstructed_inputs_embeds']=meta['generate_inputs_embeds'];o['reconstructed_attention_mask']=meta['generate_attention_mask']
            adapter=SimpleNamespace(HERE=here,FIXED=c.FIXED,read=c.read,sha=c.sha)
            anchors=dict(records={key:c.sha(d/(key+'.json'))},archives={key:c.sha(d/(key+'.npz'))})
            pred=dict(response=rec['response'])
            v.check_anchor(adapter,0,'ds_on',pred,o,anchors)
            tampered=copy.deepcopy(o);tampered['selected_visual_deepstack'][1]['sha256']='0'*64
            with self.assertRaises(AssertionError):v.check_anchor(adapter,0,'ds_on',pred,tampered,anchors)
    def test_invalid_output_and_failure_rejected(self):
        x=copy.deepcopy(self.on);x['output_ids']=[]
        with self.assertRaises(AssertionError):self.check(x,'ds_on')
        x=copy.deepcopy(self.on);x['native_failure']='error'
        with self.assertRaises(AssertionError):self.check(x,'ds_on')

if __name__=='__main__':unittest.main(verbosity=2)
