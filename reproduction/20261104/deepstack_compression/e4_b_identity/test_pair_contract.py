import copy
import unittest
from pair_contract import validate_pair,validate_forced_pair

def fixture():
    shared=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual',
            'ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids',
            'adjusted_delta','prefill_embeds','prefill_mask','first_logits','first_argmax')
    d={k:k for k in shared};d['retained_ids']=[0];d.update(mode='C-DROP',output_ids=[3,9],answer='test',eos_ids=[9],eos=True,
        truncated=False,max_new_tokens=2048,visual_calls=1,greedy_calls=1,block_calls=[1,1],main_merger_calls=1,
        ds_calls=[1,1,1],ds_results=[[0,False],[1,False],[2,False]],actual_ds_off=True,
        generation_scalars={'do_sample':False,'max_new_tokens':2048,'top_p':0.5,'top_k':5,'temperature':0.1,'repetition_penalty':1.0},
        min_new_tokens=None,decode_trace=[{'layer_cache_lengths_after':[1]*36},{}],cache_fingerprints=[cache(1)])
    s=copy.deepcopy(d);s.update(mode='C-SKIP',ds_results=[[0,True],[1,True],[2,True]])
    on=copy.deepcopy(d);on.update(budget=256,ds_on=True,forced=False)
    off=copy.deepcopy(d);off.update(budget=256,ds_on=False,forced=False,prefill_logits=d['first_logits'])
    return d,s,on,off

def cache(length):
    meta={'dtype':'torch.bfloat16','shape':[1,8,length,128],'sha256':'a'*64}
    return [{'key':copy.deepcopy(meta),'value':copy.deepcopy(meta)} for _ in range(36)]

class PairTests(unittest.TestCase):
    def test_gpu_gate_currently_unconditionally_locked(self):
        import b_gate
        from unittest.mock import patch
        from pathlib import Path
        with patch.object(b_gate,'APPROVAL',Path('/nonexistent/b_approval.json')):
            with self.assertRaisesRegex(ValueError,'E4-B locked'):b_gate.check_approval()
    def test_matching_pair_and_reference(self):
        self.assertEqual(validate_pair(*fixture())['status'],'identity_passed_bitwise')
    def test_reject_changed_visual_selection_or_output(self):
        for index,key in ((1,'main_visual'),(2,'ordered_ids'),(3,'output_ids')):
            args=list(fixture());args[index][key]='corrupted'
            with self.assertRaises(ValueError):validate_pair(*args)
    def test_reject_incomplete_natural_or_fake_skip(self):
        for change in ('endpoint','mergers'):
            d,s,on,off=fixture()
            if change=='endpoint':d['eos']=False
            else:s['ds_results']=[[0,False],[1,True],[2,True]]
            with self.assertRaises(ValueError):validate_pair(d,s,on,off)
    def test_forced_pair_binds_same_prefill_and_32_cache_steps(self):
        natural_d,natural_s,_,_=fixture();d=copy.deepcopy(natural_d);s=copy.deepcopy(natural_s)
        for r in (d,s):
            r.update(forced=True,forced_token=5,output_ids=[5]*33,eos_ids=[],eos=False,truncated=False,max_new_tokens=33,
                min_new_tokens=33,decode_trace=[{'layer_cache_lengths_after':[1+i]*36} for i in range(33)],
                cache_fingerprints=[cache(1+i) for i in range(33)])
            r['generation_scalars']['max_new_tokens']=33
        validate_forced_pair(d,s,natural_d,natural_s,5)
        s['cache_fingerprints'][17][3]['key']['sha256']='b'*64
        with self.assertRaises(ValueError):validate_forced_pair(d,s,natural_d,natural_s,5)
        s['cache_fingerprints'][17][3]['key']['sha256']='a'*64;s['first_logits']='changed'
        with self.assertRaises(ValueError):validate_forced_pair(d,s,natural_d,natural_s,5)
    def test_invalid_identical_kv_and_generation_recipe_are_rejected(self):
        d,s,on,off=fixture()
        for r in (d,s):r['cache_fingerprints'][0][0]={}
        with self.assertRaisesRegex(ValueError,'KV'):validate_pair(d,s,on,off)
        d,s,on,off=fixture();d['generation_scalars']['do_sample']=True
        with self.assertRaises(ValueError):validate_pair(d,s,on,off)

if __name__=='__main__':unittest.main()
