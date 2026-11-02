"""Synthetic CPU tamper tests for K256 terminal audit; no model or labels."""
import copy
import unittest
import numpy as np
import verify_terminal as v


def fixture(arm):
    ids=np.array([[8,9]+[99]*1024+[10,11]],dtype=np.int64)
    spatial=np.arange(256,dtype=np.int64);ordered=spatial[::-1].copy()[None,:]
    vision=np.arange(1024*8,dtype=np.uint16).reshape(1024,8)
    original=np.arange(ids.size*8,dtype=np.uint16).reshape(1,ids.size,8)
    embed=np.concatenate((original[:,:2],vision[spatial][None,:,:],original[:,1026:]),axis=1)
    attn=np.ones((1,260),dtype=np.int64)
    data=dict(ordered_ids=ordered,spatial_ids=spatial,quotas=np.array([256],dtype=np.int64),
        visual_features=vision,pruner_input=vision.copy(),pruner_output=vision[spatial],
        processor_input_ids=ids,processor_attention_mask=np.ones_like(ids),processor_pixel_values=np.ones((2,4),np.uint16),
        processor_image_grid_thw=np.array([[1,64,64]],np.int64),original_token_embeddings=original,
        generate_inputs_embeds=embed,generate_attention_mask=attn,rope_deltas=np.zeros((1,1),np.int64),
        output_ids=np.array([[7,8,9]],np.int64))
    for j in range(3):data['visual_deepstack_'+str(j)]=vision+j+1
    base=dict(inputs_embeds=embed,position_ids=np.zeros((3,1,260),np.int64),attention_mask=attn,cache_position=np.arange(260,dtype=np.int64))
    for prefix in ('original_prefill_','proposed_prefill_','final_prefill_'):
        for k,x in base.items():data[prefix+k]=x.copy()
    if arm=='ds_on':
        mask=np.zeros((1,260),np.bool_);mask[0,2:258]=True
        for prefix in ('proposed_prefill_','final_prefill_'):
            data[prefix+'visual_pos_masks']=mask.copy()
            for j in range(3):data[prefix+'deepstack_'+str(j)]=data['visual_deepstack_'+str(j)][spatial].copy()
    for i in range(2):
        for k,shape in (('position_ids',(3,1,1)),('cache_position',(1,)),('rope_deltas',(1,1))):
            data['continuation_%04d_%s'%(i,k)]=np.zeros(shape,np.int64)
    rec=dict(status='historical_anchor_passed' if arm=='ds_off' else 'observations_preserved',native_failure=None,
        observation_errors=[],selection_errors=[],response='answer',native_redecode='answer',GT_read=False,scoring_started=False,
        vision_calls=1,pruner_calls=1,processor_calls=1,generate_calls=1,selection_calls=1,llm_calls=3,
        intervention_attempts=int(arm=='ds_on'),intervention_applied=arm=='ds_on',intervention_error=None,
        generate_kwargs_keys=sorted(['inputs_embeds','attention_mask','do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty']),
        generate_non_tensor_kwargs=dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.),
        continuation_steps=2,continuation_presence=[dict(visual_mask_present=False,deepstack_present=False) for _ in range(2)])
    return data,rec

class Tests(unittest.TestCase):
    def setUp(self):self.off,self.off_r=fixture('ds_off');self.on,self.on_r=fixture('ds_on')
    def check(self,d,r,arm):v.check_arm(d,r,arm,99,1000)
    def test_valid_both_and_pair(self):
        self.check(self.off,self.off_r,'ds_off');self.check(self.on,self.on_r,'ds_on')
        self.assertTrue(v.check_pair(self.off,self.on)['output_ids_equal'])
    def test_sorted_selection_and_pruned_rows_tampering(self):
        for key in ('ordered_ids','spatial_ids','pruner_output'):
            d=copy.deepcopy(self.on);d[key].reshape(-1)[0]+=1
            with self.assertRaises(AssertionError):self.check(d,self.on_r,'ds_on')
    def test_mask_and_each_deepstack_must_be_exact_sorted_rows(self):
        for key in ('final_prefill_visual_pos_masks','final_prefill_deepstack_0','final_prefill_deepstack_1','final_prefill_deepstack_2'):
            d=copy.deepcopy(self.on);d[key].reshape(-1)[0]^=1
            with self.assertRaises(AssertionError):self.check(d,self.on_r,'ds_on')
    def test_position_or_embedding_changed_between_arms_rejected(self):
        for key in ('original_prefill_position_ids','generate_inputs_embeds','rope_deltas'):
            d=copy.deepcopy(self.on);d[key].reshape(-1)[0]+=1
            with self.assertRaises(AssertionError):v.check_pair(self.off,d)
    def test_bad_continuation_rejected_but_differences_reported(self):
        for key in ('continuation_0000_cache_position','continuation_0001_rope_deltas'):
            d=copy.deepcopy(self.on);del d[key]
            with self.assertRaises(AssertionError):self.check(d,self.on_r,'ds_on')
        r=copy.deepcopy(self.on_r);r['continuation_presence'][0]['deepstack_present']=True
        with self.assertRaises(AssertionError):self.check(self.on,r,'ds_on')
        d=copy.deepcopy(self.on);d['continuation_0000_cache_position'][0]=7
        self.check(d,self.on_r,'ds_on')
        self.assertEqual(v.check_pair(self.off,d)['same_prefix_continuation_differing_steps']['cache_position'],[0])
    def test_diverged_suffix_excludes_later_continuation(self):
        d=copy.deepcopy(self.on);d['output_ids'][0,0]=42;d['continuation_0000_cache_position'][0]=7
        result=v.check_pair(self.off,d)
        self.assertEqual(result['common_prefix_continuation_steps'],0)
        self.assertEqual(result['same_prefix_continuation_differing_steps']['cache_position'],[])
    def test_bitwise_comparison_distinguishes_signed_zero(self):
        a=np.array([0.],np.float32);b=np.array([-0.],np.float32)
        self.assertTrue(np.array_equal(a,b));self.assertFalse(v.byte_equal(a,b))
    def test_eos_and_token_limit_are_recorded_without_implying_completion(self):
        self.assertEqual(v.ending(np.array([1,2,9]),[9]),dict(output_tokens=3,EOS_present=True,EOS_at_end=True,at_max_tokens=False))
        self.assertEqual(v.ending(np.ones(2048,dtype=np.int64),[9]),dict(output_tokens=2048,EOS_present=False,EOS_at_end=False,at_max_tokens=True))
        self.assertTrue(v.ending(np.array([9,1]),[9])['EOS_present'])
        self.assertFalse(v.ending(np.array([9,1]),[9])['EOS_at_end'])
    def test_missing_output_or_wrong_native_status_rejected(self):
        d=copy.deepcopy(self.on);d['output_ids']=np.zeros((1,0),np.int64)
        with self.assertRaises(AssertionError):self.check(d,self.on_r,'ds_on')
        r=copy.deepcopy(self.off_r);r['intervention_applied']=True
        with self.assertRaises(AssertionError):self.check(self.off,r,'ds_off')

if __name__=='__main__':unittest.main(verbosity=2)
