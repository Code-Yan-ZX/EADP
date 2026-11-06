import copy
import unittest
from types import SimpleNamespace
import torch
from worker import cache_prefix, cache_shapes, checks, preflight, require_pristine


def sample(name='patched_enabled'):
    seq=1052; n=256 if name=='patched_enabled' else 1024; first=seq-1024+n
    trace=[{'positions':[[[100+i]]]*3, 'input_length':first if i==0 else 1,
            'cache_position':list(range(first)) if i==0 else [seq+i-1],
            'cache_before':[] if i==0 else [{'key':[1,8,first+i-1,128],'value':[1,8,first+i-1,128]} for _ in range(36)]}
           for i in range(33)]
    post=[{'finite_logits':True,'rope_delta':[[-900]],'model_rope_delta':[[-900]],'layers':[
        {'key':[1,8,first+i,128],'value':[1,8,first+i,128]} for _ in range(36)]} for i in range(33)]
    prefixes=[[{'key':str(j),'value':str(j)} for j in range(36)] for _ in range(2)]
    return {'name':name,'status':'completed','forced_tokens_match':True,'selected_count':n,'selected_indices':list(range(n)),
        'original_sequence_length':seq,'original_max_position':100,'original_rope_delta':[[-900]],
        'trace':trace,'post':post,'prefix':prefixes,'prefill_positions_match':True,'prefill_main_match':True,
        'prefill_text_match':True,'prefill_ds_matches':[True]*3,'prefill_ds_count':3,'prefill_ds_rows':[n]*3,
        'prefill_visual_count':n,'author_dispatch_valid':True,'hook_cleanup_valid':True,
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

    def test_control_must_keep_all_and_three_ds_match(self):
        x=sample('patched_disabled');x['selected_indices']=list(range(256));x['selected_count']=256
        x['prefill_ds_rows']=[256,256,255];x['hook_cleanup_valid']=False
        c=checks(x,36);self.assertFalse(c['selected_count_valid']);self.assertFalse(c['prefill_three_ds']);self.assertFalse(c['author_dispatch_and_hook_cleanup'])

    def test_cache_before_position_and_model_delta(self):
        x=sample();x['trace'][4]['cache_position']=[1];x['trace'][5]['cache_before'][35]['value'][-2]-=1
        x['post'][8]['model_rope_delta']=None;c=checks(x,36)
        self.assertFalse(c['input_and_cache_positions']);self.assertFalse(c['cache_before_decode']);self.assertFalse(c['rope_delta'])

    def test_real_toy_cache_hash_and_shape(self):
        t=torch.arange(24).reshape(1,2,3,4); c=SimpleNamespace(layers=[SimpleNamespace(keys=t,values=t.clone())])
        first=cache_prefix(c,2);c.layers[0].keys=torch.cat([t,t],dim=-2);c.layers[0].values=torch.cat([t,t],dim=-2)
        self.assertEqual(first,cache_prefix(c,2));self.assertEqual(cache_shapes(c)[0]['key'],[1,2,6,4])

    def test_source_mismatch(self):
        with self.assertRaises(ValueError):preflight({'sha256':{__file__:'bad'}})

    def test_pristine_rejects_marker_and_foreign_source(self):
        class A:
            def forward(self): pass
        require_pristine([(A,'forward')],__file__)
        A.__patch_selector_patched__=True
        with self.assertRaisesRegex(ValueError,'marker'):require_pristine([(A,'forward')],__file__)
        del A.__patch_selector_patched__
        A.forward=checks
        with self.assertRaisesRegex(ValueError,'source'):require_pristine([(A,'forward')],__file__)


if __name__=='__main__':unittest.main(verbosity=2)
