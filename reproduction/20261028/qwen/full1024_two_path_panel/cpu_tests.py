"""Bounded CPU tests for the plan and source path contracts; no model or CUDA."""
import ast
import sys
import unittest
from pathlib import Path
import common as c

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=c.read(c.PLAN);cls.old=c.read(c.PREV/'run_plan.json')
        cls.source=Path(cls.old['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'
        cls.tree=ast.parse(cls.source.read_text())
    def test_original_guards(self):
        c.plan_checks(self.p)
        self.assertEqual(len(self.p['panel']),16)
        self.assertNotIn('torch',sys.modules)
    def test_unregistered_1024_and_fixed_inputs(self):
        self.assertFalse(self.p['registered_model']);self.assertEqual(self.p['visual_token_num'],1024)
        self.assertEqual([x['processor_expected']['image_grid_thw']['shape'] for x in self.p['panel']],[[1,3]]*16)
    def test_paths_are_original_class_methods(self):
        classes={x.name:x for x in self.tree.body if isinstance(x,ast.ClassDef)}
        self.assertIn('Qwen3VLChatCDPruner',[x.id for x in classes['Qwen3VLChatEADP'].bases if isinstance(x,ast.Name)])
        for name in ['Qwen3VLChatFixedRes','Qwen3VLChatCDPruner']:
            self.assertEqual(sum(isinstance(x,ast.FunctionDef) and x.name=='generate_inner_transformers_fixed' for x in classes[name].body),1)
    def test_pruner_has_full_budget_bypass(self):
        p=ast.parse(Path(self.old['pruner_path']).read_text())
        cls=next(x for x in p.body if isinstance(x,ast.ClassDef) and x.name=='VisualTokenPruner')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
        cases=[x for x in ast.walk(fn) if isinstance(x,ast.If) and ast.unparse(x.test)=='token_num >= N_i']
        self.assertEqual(len(cases),1)
        self.assertIn('pruned_features_list.append(img_feats)',ast.unparse(cases[0]))
        self.assertIn('continue',ast.unparse(cases[0]))
    def test_metadata_comparison_distinguishes_missing_and_change(self):
        a={'x':{'sha256':'a'},'y':1};b={'x':{'sha256':'a'},'z':1}
        self.assertEqual(c.compare_metadata(a,b),{'x':True,'y':None,'z':None})
        self.assertFalse(c.compare_metadata({'x':1},{'x':2})['x'])
    def test_observer_failure_or_duplicate_does_not_replace_native_return(self):
        from observer import Recorder,passthrough
        rec=Recorder();sentinel=object()
        self.assertIs(passthrough(rec,'native',lambda:sentinel),sentinel)
        rec.record('broken',lambda:1/0)
        rec.record('vision',lambda:'first');rec.record('vision',lambda:'second')
        self.assertEqual(rec.count('vision'),2);self.assertEqual(len(rec.errors),1)
        self.assertIs(passthrough(rec,'next',lambda:sentinel),sentinel)
    def test_missing_duplicate_wrong_split_rejected_after_raw_preservation(self):
        import json,tempfile
        from observer import validate_capture
        keys=['visual_features','prefill_inputs_embeds','prefill_position_ids','prefill_attention_mask',
            'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw','output_ids','output_suffix_ids',
            'pruner_input','pruner_output','generate_inputs_embeds']
        metadata={k:True for k in keys}
        counts=dict(vision=1,prefill=1,generate_result=1,processor_inputs=1,pruner=1)
        validate_capture('eadp1024',counts,[],None,'saved response',metadata,[1024],{})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'raw.json';path.write_text(json.dumps({'response':'saved response'}))
            for changed,splits in [(dict(counts,vision=0),[1024]),(dict(counts,vision=2),[1024]),(counts,[256])]:
                with self.assertRaises(AssertionError):validate_capture('eadp1024',changed,[],None,'saved response',metadata,splits,{})
                self.assertEqual(json.loads(path.read_text())['response'],'saved response')
    def test_worker_draft_must_not_execute_by_default(self):
        w=ast.parse((c.HERE/'worker.py').read_text())
        self.assertTrue(any(isinstance(x,ast.Assert) for x in ast.walk(w)))
        self.assertFalse((c.HERE/'execution').exists())
        self.assertFalse((c.HERE/'root_gpu_allocation.json').exists())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assert 'torch' not in sys.modules
    c.save(c.HERE/'cpu_tests.json',dict(success=result.wasSuccessful(),tests=result.testsRun,plan_sha256=c.sha(c.PLAN),
        CPU_only=True,torch_imported=False,model_loaded=False,GPU_started=False,GT_read=False,scoring_started=False))
    sys.exit(0 if result.wasSuccessful() else 1)
