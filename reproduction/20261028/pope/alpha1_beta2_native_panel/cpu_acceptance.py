"""Bounded native-panel contracts, NumPy/stdlib only; no model/selector/GPU."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import numpy as np
import common as c
from contracts import guidance,selection,embeddings,equal_bits,require_all_sources


class Contracts(unittest.TestCase):
    def sample(self):
        ids=np.linspace(0,575,32,dtype=np.int64).reshape(1,32)
        mask=np.zeros((1,576),dtype=bool);mask[0,ids[0]]=True
        return mask,ids,np.array([32],np.int64)

    def test_beta_only_prompt_and_source_phase(self):
        prompt='Is there a truck in the image?\nAnswer the question using a single word or phrase.'
        self.assertEqual(guidance(prompt,'source'),guidance(prompt,'beta2'))
        records=[dict(position=i,arm='source',all_source_anchors_passed=True) for i in range(16)]
        require_all_sources(records)
        for bad in [records[:-1],list(reversed(records)),[dict(r,all_source_anchors_passed=False) for r in records]]:
            with self.assertRaises(AssertionError):require_all_sources(bad)

    def test_original_greedy_AST_restored_by_removing_reference_observer(self):
        _,proof=c.gain_observer.transformed()
        self.assertTrue(proof['inverse_restores_original'])
        self.assertFalse(proof['forward_tensor_content_reads']);self.assertFalse(proof['forward_tensor_copies'])
        self.assertEqual(proof,c.read(c.PLAN)['gain_observer_proof'])

    def test_actual_mask_membership_and_gain_argmax(self):
        from verify_terminal import check_native_arrays
        from contracts import array_sha
        mask,ids,quota=self.sample();observed=selection(mask,ids,quota)
        gains=np.zeros((32,1,576),np.float32)
        for i in range(32):gains[i,0,ids[0,:i]]=-1e9;gains[i,0,ids[0,i]]=1
        projected=np.zeros((1,576,4096),np.float16);llm=np.array([[1,9,-200,7]],np.int64)
        embedded=np.zeros((1,35,4096),np.float16);_,contract=embeddings(embedded,projected,mask,llm)
        arrays=dict(projected_fp16=projected,image_embeds_fp32=np.zeros((1,576,768),np.float32),
            text_global_fp32=np.zeros((1,768),np.float32),text_dense_fp32=np.zeros((1,77,768),np.float32),
            global_fp32=np.zeros((1,576,1),np.float32),dense_fp32=np.zeros((1,576,1,77),np.float32),
            visual_sim_fp32=np.zeros((1,576,576),np.float32),importance_fp32=np.zeros((1,576),np.float32),
            internal_ids=ids,quotas=quota,gain_trace_fp32=gains,final_mask=mask,actual_input_fp16=np.zeros((1,3,336,336),np.float16),
            actual_LLM_input_ids=llm,prepared_embeddings_fp16=embedded)
        record=dict(selection=observed,embedded_input_contract=contract,actual_LLM_input_ids_sha256=array_sha(llm),
            tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for k,v in arrays.items()})
        check_native_arrays(arrays,record,np)
        bad=dict(arrays);bad['gain_trace_fp32']=gains.copy();bad['gain_trace_fp32'][7,0,ids[0,7]]=0
        bad_record=copy.deepcopy(record);bad_record['tensors']['gain_trace_fp32']['sha256']=array_sha(bad['gain_trace_fp32'])
        with self.assertRaises(AssertionError):check_native_arrays(bad,bad_record,np)
        bad=ids.copy();bad[0,1]=bad[0,0]
        with self.assertRaises(AssertionError):selection(mask,bad,quota)

    def test_actual32_visual_rows_and_nonvisual_bits(self):
        mask,_,_=self.sample();projected=np.zeros((1,576,4096),np.float16)
        projected[0,:,0]=np.arange(576,dtype=np.float16)
        ids=np.array([[1,9,-200,7]],np.int64)
        prefix=np.ones((2,4096),np.float16);suffix=np.ones((1,4096),np.float16)*2
        embedded=np.concatenate([prefix,projected[0,np.flatnonzero(mask[0])],suffix])[None]
        nv,contract=embeddings(embedded,projected,mask,ids)
        self.assertTrue(equal_bits(nv,np.concatenate([prefix,suffix])))
        self.assertEqual(contract['n_visual'],32)
        bad=embedded.copy();bad[0,2,0]+=1
        with self.assertRaises(AssertionError):embeddings(bad,projected,mask,ids)
        self.assertFalse(equal_bits(np.array([0.],np.float32),np.array([-0.],np.float32)))

    def test_observers_read_no_tensor_content_and_raw_attempt_precedes_anchors(self):
        fn=next(n for n in ast.parse((c.HERE/'worker.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        observers=[n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name in ['encode','cross','greedy','prepare_inputs']]
        self.assertEqual(len(observers),4)
        for observer in observers:
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['cpu','clone','numpy','item','tolist','detach'] for n in ast.walk(observer)))
        attempt=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='save' and isinstance(n.args[0],ast.Name) and n.args[0].id=='attempt_path')
        anchors=[n for n in ast.walk(fn) if isinstance(n,ast.Assert) and n.msg is not None and 'old_native_or_helper_anchor' in ast.unparse(n.msg)]
        self.assertEqual(len(anchors),1);self.assertLess(attempt.lineno,anchors[0].lineno)

    def test_history_has_text_anchor_without_fabricated_token_IDs(self):
        fixture=c.read(c.HERE/'source_anchor_fixture.json')
        self.assertFalse(fixture['historical_output_IDs_available'])
        self.assertFalse(fixture['historical_input_IDs_available'])
        self.assertEqual(len(fixture['rows']),16)
        for row in fixture['rows']:
            self.assertIn('expected_source_text',row)
            self.assertNotIn('expected_source_output_token_ids',row)

    def test_default_check_no_runtime_and_allocation_rejects_identity_changes(self):
        import run
        paths=[c.HERE/name for name in ['launch.claim.json','state.json','execution','failure.json','worker_failure.json']]
        self.assertFalse(any(p.exists() for p in paths))
        result=json.loads(subprocess.check_output([c.PYTHON,'-B',str(c.HERE/'run.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True))
        self.assertFalse(result['torch_imported']);self.assertFalse(result['GPU_started'])
        process=dict(pid=17,start_ticks=300,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        cpu=dict(success=True,plan_sha256='p',GPU_started=False)
        note=dict(authorization='explicit_root_gpu_allocation',run_id='test',plan_sha256='p',GPU_slot='A',GPU_device=0,
            single_attempt=True,automatic_retry=False,CPU_acceptance_sha256='cpu',boot_id=process['boot_id'],
            maximum_concurrent_models_including_loading=2,allowed_other_gpu_processes=[process])
        with mock.patch.object(c,'sha',side_effect=lambda p:'p' if Path(p)==c.PLAN else 'cpu'),mock.patch.object(c,'read',return_value=cpu),\
             mock.patch.object(run,'gpu_pids',return_value=[17]),mock.patch.object(run,'identity',return_value=process):
            run.allocation(note,dict(run_id='test'))
            bad=copy.deepcopy(note);bad['allowed_other_gpu_processes'][0]['start_ticks']+=1
            with self.assertRaises(AssertionError):run.allocation(bad,dict(run_id='test'))
        self.assertFalse(any(p.exists() for p in paths))


if __name__=='__main__':
    c.check(c.read(c.PLAN))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert result.wasSuccessful() and 'torch' not in sys.modules
    c.save(c.HERE/'cpu_acceptance.json',dict(success=True,n_tests=result.testsRun,plan_sha256=c.sha(c.PLAN),
        test_sha256=c.sha(__file__),model_loaded=False,GPU_started=False,selector_executed=False,scorer_invoked=False,
        historical_source_answers_used_only_for_anchor=True,GT_read=False,accuracy_computed=False))
