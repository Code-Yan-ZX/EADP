"""Bounded CPU guard tests; never imports torch or requests a CUDA device."""
import copy
import json
import os
import subprocess
import sys
import unittest
import numpy as np
import common as c
from verify_terminal import source_checks


class Gates(unittest.TestCase):
    def test_single_source_AST_change(self):
        proof=c.helper().nodes()[-1]
        self.assertTrue(proof['aggregation_inverse_restores_source'])
        self.assertTrue(proof['greedy_observer_inverse_restores_source'])
        self.assertEqual(proof,c.read(c.PLAN)['source_AST_proof'])

    def test_all_four_source_anchors_required(self):
        original=dict(native_importance=np.array([[1]],np.float32),native_ordered_ids=np.array([[0]],np.int64),
            source__gain_trace=np.array([[[1]]],np.float32),native_pruned_BF16_bits=np.array([[15]],np.uint16))
        keys=dict(source__importance='native_importance',source__ordered_ids='native_ordered_ids',
            source__gains='source__gain_trace',source__final_BF16_bits='native_pruned_BF16_bits')
        source={key:original[value].copy() for key,value in keys.items()}
        self.assertTrue(all(source_checks(source,original,np).values()))
        for key in keys:
            altered=copy.deepcopy(source);altered[key].flat[0]+=1
            self.assertFalse(all(source_checks(altered,original,np).values()),key)

    def test_exact_root_allocation_and_PID_identity(self):
        note=dict(authorization='explicit_root_gpu_allocation',run_id=c.RUN_ID,plan_sha256='a'*64,GPU_slot='A',
            CUDA_VISIBLE_DEVICES='0',maximum_concurrent_models_including_loading=2,
            allowed_other_gpu_pids=[],allowed_other_gpu_processes=[])
        c.validate_allocation(note,'a'*64,[],{})
        for key,value in [('plan_sha256','b'*64),('GPU_slot','B'),('maximum_concurrent_models_including_loading',3)]:
            bad=dict(note);bad[key]=value
            with self.assertRaises(AssertionError):c.validate_allocation(bad,'a'*64,[],{})
        process=dict(pid=17,start_ticks=300,boot_id='fixed')
        note.update(allowed_other_gpu_pids=[17],allowed_other_gpu_processes=[process])
        c.validate_allocation(note,'a'*64,[17],{17:process})
        with self.assertRaises(AssertionError):c.validate_allocation(note,'a'*64,[17],{17:dict(process,start_ticks=301)})
        with self.assertRaises(AssertionError):c.validate_allocation(note,'a'*64,[17,18],{17:process})

    def test_default_check_has_no_runtime_side_effects(self):
        forbidden=[c.HERE/n for n in ['launch.claim.json','state.json','execution','failure.json','worker_failure.json']]
        before=[p.exists() for p in forbidden];self.assertFalse(any(before))
        output=subprocess.check_output([str(c.PYTHON),'-B',str(c.HERE/'run.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True)
        result=json.loads(output)
        self.assertFalse(result['torch_imported']);self.assertFalse(result['GPU_started'])
        self.assertEqual(before,[p.exists() for p in forbidden])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Gates))
    assert 'torch' not in sys.modules and not c.forbidden_imports()
    if not result.wasSuccessful():raise SystemExit(1)
    c.save(c.HERE/'CPU_acceptance.json',dict(status='passed',tests_run=result.testsRun,plan_sha256=c.sha(c.PLAN),
        torch_imported=False,GPU_started=False,model_loaded=False,source_anchor_failures_explicitly_unmeasured=True),exclusive=True)
