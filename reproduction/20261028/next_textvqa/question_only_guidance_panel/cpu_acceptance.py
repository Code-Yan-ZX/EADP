"""Small CPU contracts for the fixed panel; never load a model or CUDA."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from contracts import POSITIONS,guidance,spatial_order,selection,embeddings,equal_bits,require_all_sources,sha,save

HERE=Path(__file__).resolve().parent


class Contracts(unittest.TestCase):
    def sample(self):
        mask=np.zeros((5,576),dtype=np.bool_)
        for crop,indices in enumerate([[1],[1],[0,24],[0],[0]]):mask[crop,indices]=True
        quotas=mask.sum(1).astype(np.int64);ids=np.zeros((5,2),dtype=np.int64)
        for crop in range(5):ids[crop,:quotas[crop]]=np.flatnonzero(mask[crop])
        return mask,ids,quotas

    def test_only_guidance_changes(self):
        prompt='Which brand?\nReference OCR token: ABC\nAnswer the question using a single word or phrase.'
        self.assertEqual(guidance(prompt,'source'),'Which brand?\nReference OCR token: ABC')
        self.assertEqual(guidance(prompt,'question_only'),'Which brand?')
        self.assertIn('Reference OCR token: ABC',prompt)
        with self.assertRaises(AssertionError):guidance('missing separator','question_only')

    def test_spatial_order_independent_known_coordinates(self):
        mask,_,_=self.sample()
        self.assertEqual(spatial_order(mask).tolist(),[1,577,1152,1176,1728,2304])

    def test_effective_prefix_not_discarded_suffix(self):
        mask,ids,quota=self.sample()
        observed=selection(mask,ids,quota)
        self.assertEqual(observed['returned_retained_tokens'],6)
        self.assertEqual(observed['greedy_effective_prefixes'][0],[1])
        ids[0,1]=575
        self.assertEqual(selection(mask,ids,quota),observed)

    def test_bad_real_prefix_rejected(self):
        mask,ids,quota=self.sample();ids[2,1]=0
        with self.assertRaises(AssertionError):selection(mask,ids,quota)

    def test_actual_visual_embedding_bit_check(self):
        mask,_,_=self.sample();features=np.arange(5*576*3,dtype=np.float16).reshape(5,576,3)
        input_ids=np.asarray([[1,8,-200,9]],dtype=np.int64)
        prefix=np.ones((2,3),dtype=np.float16);suffix=np.ones((1,3),dtype=np.float16)*2
        visual=features.reshape(2880,3)[spatial_order(mask)]
        embedded=np.concatenate([prefix,visual,suffix])[None]
        nv,_=embeddings(embedded,features,mask,input_ids)
        self.assertTrue(equal_bits(nv,np.concatenate([prefix,suffix])))
        broken=embedded.copy();broken[0,2,0]=0
        with self.assertRaises(AssertionError):embeddings(broken,features,mask,input_ids)

    def test_all16_source_phase_gate(self):
        records=[dict(position=p,arm='source',all_source_anchors_passed=True) for p in POSITIONS]
        require_all_sources(records)
        for wrong in [records[:-1],list(reversed(records)),[dict(x,all_source_anchors_passed=False) for x in records]]:
            with self.assertRaises(AssertionError):require_all_sources(wrong)

    def test_strided_bits_and_signed_zero(self):
        values=np.arange(12,dtype=np.float32).reshape(3,4).T
        self.assertTrue(equal_bits(values,values.copy()))
        self.assertFalse(equal_bits(np.asarray([0.],dtype=np.float32),np.asarray([-0.],dtype=np.float32)))

    def test_no_forward_tensor_content_observation(self):
        tree=ast.parse((HERE/'probe.py').read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        observers=[n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name in ['encode','cross','greedy','prepare_inputs']]
        self.assertEqual(len(observers),4)
        for function in observers:
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['cpu','clone','numpy','item','tolist'] for n in ast.walk(function)))

    def test_attempt_saved_before_postgeneration_anchors(self):
        tree=ast.parse((HERE/'probe.py').read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        attempts=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save' and isinstance(n.args[0],ast.Name) and n.args[0].id=='attempt_path']
        anchors=[n for n in ast.walk(worker) if isinstance(n,ast.Assert) and n.msg is not None and 'old_native_feature_anchor' in ast.unparse(n.msg)]
        self.assertEqual(len(attempts),1);self.assertEqual(len(anchors),1)
        self.assertLess(attempts[0].lineno,anchors[0].lineno)


if __name__=='__main__':
    import probe
    plan=probe.read(HERE/'plan.json');probe.check(plan)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==9 and 'torch' not in sys.modules
    save(HERE/'cpu_acceptance.json',dict(success=True,n_tests=9,checked_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(HERE/'plan.json'),test_sha256=sha(__file__),source_and_model_and_image_and_cache_guards_passed=True,
        model_loaded=False,GPU_started=False,GT_or_accuracy_used=False))
    print(json.dumps(dict(success=True,n_tests=9,plan_sha256=sha(HERE/'plan.json'))))
