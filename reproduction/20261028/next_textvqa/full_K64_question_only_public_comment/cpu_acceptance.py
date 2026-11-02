"""CPU contract checks only; never executes model, scorer or future run."""
import ast
import copy
import hashlib
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import guidance_patch
import worker
import freeze_plan
import run


class CPUContracts(unittest.TestCase):
    def test_native_AST_only_guidance_insertion_and_executed_code_binding(self):
        tree,proof=guidance_patch.transformed()
        self.assertTrue(proof['inverse_restores_original'])
        original=ast.parse(guidance_patch.ENTRY.read_text())
        original_dataset=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='CustomDataset')
        new_dataset=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CustomDataset')
        self.assertEqual(ast.dump(original_dataset),ast.dump(new_dataset))
        code=compile(tree,str(guidance_patch.ENTRY)+'[question-only-guidance AST]','exec')
        actual=next(v for v in code.co_consts if isinstance(v,types.CodeType) and v.co_name=='eval_model')
        self.assertEqual(hashlib.sha256(marshal.dumps(actual)).hexdigest(),proof['expected_eval_model_code_sha256'])

    def test_all5000_question_guidance_and_unchanged_full_prompt(self):
        questions=[json.loads(s) for s in (worker.NEXT/'author_linked_textvqa_questions.jsonl').read_text().splitlines()]
        refs=[json.loads(s) for s in (worker.NEXT/'guidance_question_boundary_exact_token_rows.jsonl').read_text().splitlines()]
        self.assertEqual(len(questions),5000);self.assertEqual(len(refs),5000)
        for i,(q,r) in enumerate(zip(questions,refs)):
            self.assertEqual(i,r['position']);self.assertEqual(q['question_id'],r['question_id'])
            self.assertEqual(q['text'].count(guidance_patch.SEPARATOR),1)
            self.assertEqual(hashlib.sha256(q['text'].encode()).hexdigest(),r['unchanged_full_LLM_prompt_sha256'])
            expected=q['text'].split(guidance_patch.SEPARATOR)[0].replace(guidance_patch.SUFFIX,'')
            self.assertTrue(expected.strip())
            self.assertEqual(hashlib.sha256(expected.encode()).hexdigest(),r['question_only_guidance']['guidance_sha256'])

    def test_predecessor_full_terminal_and_actual_input_change_not_score(self):
        root=freeze_plan.read(freeze_plan.PANEL/'root_terminal_process_proof.json')
        report=freeze_plan.read(freeze_plan.PANEL/'execution/independent_verification.json')
        freeze_plan.check_panel_metadata(root,report)
        wrong=copy.deepcopy(root);wrong['root_owned_exec_actual_exit_code']=1
        with self.assertRaises(AssertionError):freeze_plan.check_panel_metadata(wrong,report)
        wrong=copy.deepcopy(report);wrong['source_anchors_verified']=15
        with self.assertRaises(AssertionError):freeze_plan.check_panel_metadata(root,wrong)
        wrong=copy.deepcopy(report)
        for item in wrong['changes']:item['final_masks_changed']=False
        with self.assertRaises(AssertionError):freeze_plan.check_panel_metadata(root,wrong)
        unchanged_answers=copy.deepcopy(report)
        for item in unchanged_answers['changes']:
            item['output_token_IDs_changed']=False;item['raw_text_changed']=False
        freeze_plan.check_panel_metadata(root,unchanged_answers)

    def test_max_two_and_exact_allocation_process_identity(self):
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        proc=dict(pid=17,start_ticks=88,boot_id=boot)
        note=dict(authorization='explicit_root_gpu_allocation',run_id=worker.RUN_ID,plan_sha256='a'*64,
            GPU_slot='A',boot_id=boot,allowed_other_gpu_pids=[17],allowed_other_gpu_processes=[proc])
        with patch.object(run,'gpu_pids',return_value=[17]),patch.object(run,'identity',return_value=proc):
            run.validate_allocation(note,'a'*64)
            with self.assertRaises(AssertionError):run.validate_allocation(dict(note,plan_sha256='b'*64),'a'*64)
        with patch.object(run,'gpu_pids',return_value=[17]),patch.object(run,'identity',return_value=dict(proc,start_ticks=89)):
            with self.assertRaises(AssertionError):run.validate_allocation(note,'a'*64)

    def test_default_check_no_launch_or_torch(self):
        paths=[worker.OUT/n for n in ['launch.claim.json','execution','controller_state.json','failure.json']]
        self.assertFalse(any(p.exists() for p in paths))
        output=subprocess.check_output(['/home/dell/miniconda3/envs/llava_pruner/bin/python','-B',str(worker.OUT/'run.py'),'check'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True)
        result=json.loads(output);self.assertTrue(result['CPU_validation']);self.assertFalse(result['GPU_started'])
        self.assertFalse(any(p.exists() for p in paths))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUContracts))
    assert 'torch' not in sys.modules
    if not result.wasSuccessful():raise SystemExit(1)
    out=worker.OUT/'CPU_acceptance.json'
    with out.open('x') as stream:json.dump(dict(status='passed',tests_run=result.testsRun,plan_sha256=worker.sha(worker.PLAN),
        torch_imported=False,GPU_started=False,model_loaded=False,scorer_invoked=False,prefix_scored=False),stream,indent=2);stream.write('\n')
