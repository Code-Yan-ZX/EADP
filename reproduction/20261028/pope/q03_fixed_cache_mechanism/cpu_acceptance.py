"""Only bounded source/byte/allocation gates; no torch, selector or CUDA call."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import common as c
import helpers


class Gates(unittest.TestCase):
    def test_only_actual_filter_ratio_changes_and_original_topk_body_remains(self):
        proof = helpers.nodes()[-1]
        self.assertTrue(proof['ratio_call_inverse_restores_source'])
        self.assertTrue(proof['greedy_observer_removed_restores_original'])
        self.assertEqual(proof, c.read(c.PLAN)['source_AST_proof'])
        self.assertEqual([max(1, int(77*q)) for q in [.2,.3]], [15,23])
        self.assertEqual(max(1,int(1*.2)),1)

    def test_every_source_byte_anchor_and_signed_zero(self):
        from verify_terminal import source_checks
        original = dict(projected_fp16=np.arange(8,dtype=np.float16).reshape(1,4,2),
            alpha05_beta1__source__importance=np.array([[0.,1.,2.,3.]],np.float32),
            alpha05_beta1__source__ids=np.array([[0,2]],np.int64),
            alpha05_beta1__source__gain_trace=np.ones((2,1,4),np.float32),
            alpha05_beta1__source__final_mask=np.array([[True,False,True,False]],bool))
        source = dict(source__importance=original['alpha05_beta1__source__importance'].copy(),
            source__ordered_ids=original['alpha05_beta1__source__ids'].copy(),
            source__gains=original['alpha05_beta1__source__gain_trace'].copy(),
            source__final_mask=original['alpha05_beta1__source__final_mask'].copy(),
            source__final_projected_FP16_sequence=original['projected_fp16'][0,[0,2]].copy())
        self.assertTrue(all(source_checks(source,original,np).values()))
        for key in source:
            altered = copy.deepcopy(source)
            if altered[key].dtype == np.bool_: altered[key].flat[0] = not altered[key].flat[0]
            else: altered[key].flat[0] += 1
            self.assertFalse(all(source_checks(altered,original,np).values()), key)
        source['source__importance'][0,0] = -0.
        self.assertFalse(source_checks(source,original,np)['importance'])

    def test_allocation_requires_exact_identity_boot_slot_and_plan(self):
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        note = dict(authorization='explicit_root_gpu_allocation',run_id=c.RUN_ID,plan_sha256='a'*64,
            GPU_slot='A',CUDA_VISIBLE_DEVICES='0',maximum_concurrent_models_including_loading=2,
            allowed_other_gpu_pids=[],allowed_other_gpu_processes=[],boot_id=boot)
        c.validate_allocation(note,'a'*64,[],{})
        for key,value in [('plan_sha256','b'*64),('GPU_slot','B'),('boot_id','wrong'),('maximum_concurrent_models_including_loading',3)]:
            bad = dict(note); bad[key] = value
            with self.assertRaises(AssertionError): c.validate_allocation(bad,'a'*64,[],{})
        process = dict(pid=17,start_ticks=300,boot_id=boot)
        note.update(allowed_other_gpu_pids=[17],allowed_other_gpu_processes=[process])
        c.validate_allocation(note,'a'*64,[17],{17:process})
        with self.assertRaises(AssertionError): c.validate_allocation(note,'a'*64,[17],{17:dict(process,start_ticks=301)})
        with self.assertRaises(AssertionError): c.validate_allocation(note,'a'*64,[17,18],{17:process})

    def test_cache_recipe_boundary_is_not_relabelled_native(self):
        plan = c.read(c.PLAN)
        self.assertIn('not a separate native encode',plan['anchor_boundary'])
        self.assertIn('BEFORE entropy filtering',plan['graph_cut'])
        self.assertEqual([r['position'] for r in plan['records']],list(range(16)))
        for row in plan['records']:
            self.assertEqual(row['reference_case']['recipe']['name'],'alpha05_beta1')
            self.assertEqual(row['reference_case']['variant'],'source')

    def test_default_check_no_side_effects_or_model_import(self):
        paths = [c.HERE/name for name in ['launch.claim.json','state.json','execution','failure.json','worker_failure.json']]
        before = [p.exists() for p in paths]; self.assertFalse(any(before))
        result = json.loads(subprocess.check_output([str(c.PYTHON),'-B',str(c.HERE/'run.py')],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True))
        self.assertFalse(result['torch_imported']); self.assertFalse(result['GPU_started'])
        self.assertEqual(before,[p.exists() for p in paths])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Gates))
    assert 'torch' not in sys.modules and not c.forbidden_imports()
    if not result.wasSuccessful(): raise SystemExit(1)
    c.save(c.HERE/'CPU_acceptance.json', dict(status='passed',tests_run=result.testsRun,plan_sha256=c.sha(c.PLAN),
        torch_imported=False,GPU_started=False,model_loaded=False,selector_executed=False,scorer_invoked=False,
        source_anchor_failures_explicitly_unmeasured=True),exclusive=True)
