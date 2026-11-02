"""Small CPU gates; no model/torch imports or generated prefixes are read."""
import copy
import json
import unittest
from unittest.mock import patch

import worker
import verify


class Acceptance(unittest.TestCase):
    def test_exact_adapter_inverse(self):
        full, proof = worker.adapted_module()
        self.assertTrue(proof['restored_AST_identical'])
        self.assertEqual(full.PARAMETERS['visual_token_num'], 64)
        self.assertEqual(full.PARAMETERS['budget_nominal'], 320)

    def test_non320_native_floor_total_is_valid(self):
        r = dict(position=0, input_shape=[1, 2], input_dtype='torch.int64', input_ids_sha256='ids', crop_count=5,
            initial_quotas=[64]*5, allocated_quotas=[63]*5, greedy_quotas=[63]*5, effective_mask_counts=[63]*5,
            effective_mask_shape=[5,576], effective_mask_indices=[list(range(63)) for _ in range(5)],
            greedy_effective_prefixes=[list(range(62,-1,-1)) for _ in range(5)], returned_retained_tokens=315,
            importance_shape=[5,576], importance_finite=True, projected_feature_shape=[5,576,4096],
            projected_feature_dtype='torch.float16', post_generation_tensor_reads_only=True)
        ref = dict(position=0, n_tokens=2, token_ids_sha256='ids')
        verify.check_observation(r, ref, dict(actual_visual_tokens_retained=315), 0)
        for key, value in [('input_ids_sha256','bad'), ('initial_quotas',[128]*5), ('effective_mask_counts',[64]*5),
                           ('returned_retained_tokens',320), ('post_generation_tensor_reads_only',False)]:
            bad = copy.deepcopy(r);bad[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                verify.check_observation(bad, ref, dict(actual_visual_tokens_retained=315), 0)
        for field in ('effective_mask_indices','greedy_effective_prefixes'):
            bad=copy.deepcopy(r);bad[field][0][0]=999
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify.check_observation(bad, ref, dict(actual_visual_tokens_retained=315), 0)

    def test_incomplete_gate_before_import_or_scoring(self):
        with patch('verify.read', return_value=dict(status='generating', n=4999)), patch.dict(verify.os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(AssertionError):verify.main()

    def test_consensus(self):
        self.assertEqual([round(verify.consensus(n),2) for n in range(11)], [0,.3,.6,.9,1,1,1,1,1,1,1])

    def test_allocation_requires_exact_other_PIDs_and_capacity(self):
        import run
        note=dict(authorization='explicit_root_gpu_allocation',run_id='NeXT_public_default_K64_alpha05_beta1_wait',plan_sha256='fixed',GPU_slot='A',allowed_other_gpu_pids=[])
        with patch('run.gpu_pids',return_value=[]):run.validate_allocation(note,'fixed')
        with patch('run.gpu_pids',return_value=[1,2]),self.assertRaises(AssertionError):run.validate_allocation(note,'fixed')
        with patch('run.gpu_pids',return_value=[]),self.assertRaises(AssertionError):run.validate_allocation(note,'wrong')


if __name__ == '__main__':
    unittest.main()
