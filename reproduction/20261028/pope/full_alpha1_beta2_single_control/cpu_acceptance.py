"""Bounded stdlib gates; no generation, labels, scoring, tensors or GPU access."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import common as c


def fixture():
    q=dict(question_id=1,image='fixed.jpg',text='Is there a cat?\nAnswer the question using a single word or phrase.')
    ids=[1,3,-200,4]
    row=dict(position=0,question_id=1,image=q['image'],full_prompt_sha256=c.text_sha(q['text']),
        actual_guidance_sha256=c.text_sha(c.guidance(q['text'])),actual_alpha_beta_K=[1.,2.,32],
        calls=dict(encode=1,greedy=1),post_generation_tensor_reads_only=True,input_token_ids=ids,
        input_ids_sha256=c.int64_sha(ids),input_ids_dtype='torch.int64',input_ids_shape=[1,4],
        quotas=[32],mask_shape=[1,576],effective_ordered_prefix=list(range(31,-1,-1)),effective_mask_indices=list(range(32)),
        returned_retained_tokens=32,projected_feature_shape=[1,576,4096],projected_feature_dtype='torch.float16',
        features_finite=True,importance_finite=True,importance_shape=[1,576],output_token_ids=[1,3869,2],
        new_tokens=2,EOS_present=True,generation_horizon=128,text='Yes')
    return q,row


class Gates(unittest.TestCase):
    def test_observation_valid_and_horizon_preserved(self):
        q,r=fixture();c.check_observation(r,q,0)
        r.update(output_token_ids=[1]+[3]*128,new_tokens=128,EOS_present=False,text='')
        c.check_observation(r,q,0)  # Keep legal EOS/horizon/empty outcomes.

    def test_ID_hash_and_full_guidance_reject_corruption(self):
        q,r=fixture()
        for key,value in [('input_ids_sha256','0'*64),('actual_guidance_sha256','0'*64),('actual_alpha_beta_K',[1.,1.,32])]:
            altered=copy.deepcopy(r);altered[key]=value
            with self.assertRaises(AssertionError): c.check_observation(altered,q,0)

    def test_mask_and_effective_prefix_reject_corruption(self):
        q,r=fixture()
        for key,value in [('quotas',[31]),('effective_ordered_prefix',[0]*32),('effective_mask_indices',list(range(1,33)))]:
            altered=copy.deepcopy(r);altered[key]=value
            with self.assertRaises(AssertionError): c.check_observation(altered,q,0)

    def test_panel_anchor_flattening_and_output_rejection(self):
        q,r=fixture();c.check_anchor(r,r)
        for key in ['output_token_ids','effective_ordered_prefix']:
            changed=copy.deepcopy(r);changed[key]=[r[key]]
            with self.assertRaises(AssertionError): c.check_anchor(r,changed)

    def test_allocation_rejects_wrong_plan_before_GPU_queries(self):
        from run import allocation
        plan=dict(run_id=c.RUN_ID)
        note=dict(authorization='explicit_root_gpu_allocation',run_id=c.RUN_ID,plan_sha256='wrong')
        with patch('run.gpu_pids',side_effect=RuntimeError('must not query GPU')):
            with self.assertRaises(AssertionError): allocation(note,plan)

    def test_actual_frozen_guards_and_predecessor(self):
        c.check(c.read(c.PLAN));assert len(c.read(c.HERE/'beta2_panel_anchors.json')['rows'])==16
        assert all(not (c.HERE/name).exists() for name in ['execution','state.json','launch.claim.json'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Gates)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert 'torch' not in sys.modules
    c.save(c.HERE/'cpu_acceptance.json',dict(success=result.wasSuccessful(),n_tests=result.testsRun,
        plan_sha256=c.sha(c.PLAN),CPU_only=True,torch_imported=False,GPU_started=False,model_loaded=False,
        GT_read=False,scoring_started=False,queue_started=False))
    sys.exit(0 if result.wasSuccessful() else 1)
