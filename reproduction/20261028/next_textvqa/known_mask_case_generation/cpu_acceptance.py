"""Real cache CPU contracts and rejection checks; no model/torch/GPU/GT."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from cache_contract import load,ARMS,visual_order,verify_embeddings


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays,cls.contract=load()
        cls.embeddings={}
        for arm in ARMS:
            visual=cls.arrays['projected_fp16'].reshape(2880,4096)[visual_order(cls.arrays[arm+'_final_mask'])]
            cls.embeddings[arm]=np.concatenate([np.zeros((2,4096),np.float16),visual,np.ones((3,4096),np.float16)])[None]

    def test_actual_cache_native_spatial_order(self):
        verify_embeddings(self.embeddings,6,2,self.arrays,self.contract)
        self.assertEqual(self.contract['shifted_visual_sequence_positions'],list(range(465,548)))

    def test_changed_nonvisual_or_visual_content_rejected(self):
        for offset in (0,10):
            bad={k:v.copy() for k,v in self.embeddings.items()}
            bad[ARMS[1]].view(np.uint16)[0,offset,0]^=1
            with self.subTest(offset=offset),self.assertRaises(AssertionError):
                verify_embeddings(bad,6,2,self.arrays,self.contract)

    def test_incomplete_K64_rejected_without_reading_answers(self):
        import probe
        with patch('probe.read',return_value={'status':'generating'}),self.assertRaises(AssertionError):probe.k64_terminal_gate()

    def test_root_owned_return_and_SHA_identity_required(self):
        import probe
        state=dict(controller=dict(pid=2734202,start_ticks=1332948026),native_worker=dict(pid=2734206,start_ticks=1332948051))
        claim=dict(controller=state['controller'],plan_sha256=probe.K64_PLAN_SHA,allocation_note_sha256=probe.K64_NOTE_SHA)
        root=dict(root_owned_exec_handle=63618,root_owned_exec_actual_exit_code=0,all_original_processes_exited=True,
            controller=state['controller'],native_worker=state['native_worker'],plan_sha256=probe.K64_PLAN_SHA,
            allocation_note_sha256=probe.K64_NOTE_SHA,release_sha256='release',independent_verification_sha256='report')
        probe.validate_k64_root(root,state,claim,'release','report')
        for key,value in [('root_owned_exec_actual_exit_code',1),('all_original_processes_exited',False),
                          ('release_sha256','bad'),('independent_verification_sha256','bad'),('allocation_note_sha256','bad')]:
            bad=copy.deepcopy(root);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):probe.validate_k64_root(bad,state,claim,'release','report')


if __name__=='__main__':unittest.main()
