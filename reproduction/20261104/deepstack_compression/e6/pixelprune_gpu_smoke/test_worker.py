import copy
import unittest
from types import SimpleNamespace
import torch
from worker import cache_prefix, cache_shapes, checks, preflight


def sample(name='patched_enabled'):
    seq=1052; n=256 if name=='patched_enabled' else 1024; first=seq-1024+n
    trace=[{'positions':[[[100+i]]]*3, 'input_length':first if i==0 else 1} for i in range(33)]
    post=[{'finite_logits':True,'rope_delta':[[-900]],'layers':[
        {'key':[1,8,first+i,128],'value':[1,8,first+i,128]} for _ in range(36)]} for i in range(33)]
    prefixes=[[{'key':str(j),'value':str(j)} for j in range(36)] for _ in range(2)]
    return {'name':name,'status':'completed','forced_tokens_match':True,'selected_count':n,'selected_indices':list(range(n)),
        'original_sequence_length':seq,'original_max_position':100,'original_rope_delta':[[-900]],
        'trace':trace,'post':post,'prefix':prefixes,'prefill_positions_match':True,'prefill_main_match':True,
        'prefill_text_match':True,'prefill_ds_matches':[True]*3,
        'logit_checks':[{'comparable':True,'allclose':True,'argmax_equal':True} for _ in range(33)]}


class Tests(unittest.TestCase):
    def test_valid_three_arms(self):
        for name in ('unpatched','patched_disabled','patched_enabled'):
            self.assertTrue(all(checks(sample(name),36).values()))

    def test_reject_missing_calls_rope_and_ds(self):
        x=sample(); x['trace'].pop(); x['post'][2]['rope_delta']=None; x['prefill_ds_matches'][0]=False
        c=checks(x,36)
        self.assertFalse(c['one_prefill_32_decode']); self.assertFalse(c['rope_delta']); self.assertFalse(c['prefill_three_ds'])

    def test_reject_cache_and_indices(self):
        x=sample(); x['selected_indices'][1]=x['selected_indices'][0]; x['post'][5]['layers'][34]['value'][-2]-=1
        x['prefix'][1][0]['key']='different'; c=checks(x,36)
        self.assertFalse(c['selected_count_valid']); self.assertFalse(c['cache_lengths']); self.assertFalse(c['cache_prefix_unchanged'])

    def test_reject_nonfinite_and_disabled_mismatch(self):
        x=sample('patched_disabled');x['post'][3]['finite_logits']=False;x['logit_checks'][7]['allclose']=False
        c=checks(x,36);self.assertFalse(c['finite_logits']);self.assertFalse(c['disabled_logits_match_unpatched'])

    def test_real_toy_cache_hash_and_shape(self):
        t=torch.arange(24).reshape(1,2,3,4); c=SimpleNamespace(layers=[SimpleNamespace(keys=t,values=t.clone())])
        first=cache_prefix(c,2);c.layers[0].keys=torch.cat([t,t],dim=-2);c.layers[0].values=torch.cat([t,t],dim=-2)
        self.assertEqual(first,cache_prefix(c,2));self.assertEqual(cache_shapes(c)[0]['key'],[1,2,6,4])

    def test_source_mismatch(self):
        with self.assertRaises(ValueError):preflight({'sha256':{__file__:'bad'}})


if __name__=='__main__':unittest.main(verbosity=2)
