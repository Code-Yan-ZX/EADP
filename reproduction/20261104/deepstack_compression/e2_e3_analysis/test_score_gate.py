import unittest
from types import SimpleNamespace
from unittest.mock import patch
import score_panel

class ScoreGateTests(unittest.TestCase):
    def test_missing_formal_acceptance_fails_before_panel_or_truth(self):
        args=SimpleNamespace(verify_started='s',verify_terminal='t',panel_proof='p')
        with patch.object(score_panel.panel_gate,'terminal'), patch.object(score_panel,'read',return_value={'status':'failed'}), patch.object(score_panel.panel_worker,'inputs') as inputs, patch.object(score_panel.e1_score,'read_truths') as truth:
            with self.assertRaisesRegex(ValueError,'formal full panel'):score_panel.gate(args)
            inputs.assert_not_called();truth.assert_not_called()
    def test_independent_acceptance_cannot_be_skipped(self):
        args=SimpleNamespace(verify_started='s',verify_terminal='t',panel_proof='p',review_started='rs',review_terminal='rt',review_proof='r')
        formal={'status':'passed','questions':384,'physical_generations':3840,'E2_records':3072,'E3_new_records':768,'E3_endpoint_aliases':768,'GT_opened':False,'scoring_performed':False,'verifier_sha256':'ok'}
        with patch.object(score_panel.panel_gate,'terminal'),patch.object(score_panel,'sha',return_value='ok'),patch.object(score_panel,'read',side_effect=[formal,{'status':'failed'}]),patch.object(score_panel.panel_worker,'inputs') as inputs,patch.object(score_panel.e1_score,'read_truths') as truth:
            with self.assertRaisesRegex(ValueError,'independent full panel'):score_panel.gate(args)
            inputs.assert_not_called();truth.assert_not_called()
    def test_actual_author_and_independent_scorers_on_synthetic_truth(self):
        author=score_panel.metrics.author_functions(); constants=score_panel.metrics._author_constants()
        cases=(('DocVQA_VAL','the answer',{'answers':['the answer']},1.),
               ('TextVQA_VAL','cat',{'answers':['cat']*10},1.),
               ('AI2D_TEST','A',{'answer':'A','choices':{'A':'cat','B':'dog'}},1.))
        for task,pred,truth,want in cases:
            result=score_panel.metrics.score(task,pred,truth,author,constants,scoring_prediction=pred)
            self.assertEqual(result['official'],want);self.assertEqual(result['independent'],want)
if __name__=='__main__':unittest.main()

class FullGateSyntheticTests(unittest.TestCase):
    def fixture(self, root):
        import json
        from pathlib import Path
        d=Path(root);run=d/'run';(run/'data').mkdir(parents=True);reviewdir=d/'review';reviewdir.mkdir()
        reviewer=reviewdir/'review_e2e3full_run.py';reviewer.write_text('# synthetic reviewer\n')
        paths={}
        for name in ('e1_acceptance','e1_started','e1_terminal','run_started','run_terminal','verify_started','verify_terminal','review_started','review_terminal'):
            paths[name]=d/name;paths[name].write_text('{}')
        for rel in ('launch.json','exit.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl','data/e3_views.jsonl'):
            (run/rel).write_text('{}')
        source=d/'source.py';source.write_text('synthetic source')
        (run/'frozen.json').write_text(json.dumps({'code_sha256':{str(source):score_panel.sha(source)}}))
        arr=d/'array.npz';arr.write_bytes(b'synthetic array')
        required=[paths[k] for k in ('e1_acceptance','e1_started','e1_terminal','run_started','run_terminal')]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl','data/e3_views.jsonl')]
        proof={'status':'passed','questions':384,'physical_generations':3840,'E2_records':3072,'E3_new_records':768,'E3_endpoint_aliases':768,'GT_opened':False,'scoring_performed':False,'verifier_sha256':score_panel.sha(score_panel.FULL/'panel_verify.py'),'files_sha256':{str(x.resolve()):score_panel.sha(x) for x in required},'array_sha256':{str(arr):score_panel.sha(arr)}}
        panelproof=d/'proof.json';panelproof.write_text(json.dumps(proof))
        review={'status':'passed','independent_review':True,'questions':384,'physical_generations':3840,'GT_opened':False,'scoring_performed':False,'formal_sha256':score_panel.sha(panelproof),'formal_started_sha256':score_panel.sha(paths['verify_started']),'formal_terminal_sha256':score_panel.sha(paths['verify_terminal']),'reviewer_sha256':score_panel.sha(reviewer)}
        reviewproof=d/'review.json';reviewproof.write_text(json.dumps(review))
        args=SimpleNamespace(run=run,panel_proof=panelproof,review_proof=reviewproof,**paths)
        return args,reviewdir,source,arr
    def run_fixture(self, mutate=None, terminal_failure=False):
        import tempfile
        from types import SimpleNamespace as NS
        with tempfile.TemporaryDirectory() as root:
            args,reviewdir,source,arr=self.fixture(root)
            if mutate=='record':(args.run/'data/records.jsonl').write_text('mutated')
            if mutate=='array':arr.write_bytes(b'mutated')
            if mutate=='source':source.write_text('mutated')
            effect=ValueError('verifier still live') if terminal_failure else None
            with patch.object(score_panel,'REVIEW',reviewdir),patch.object(score_panel.panel_gate,'terminal',side_effect=effect),patch.object(score_panel.panel_verify,'validate_exit') as exit_check,patch.object(score_panel.panel_verify,'check_e1',return_value={'GT_opened':False,'scoring_performed':False}),patch.object(score_panel.panel_worker,'inputs',return_value=(['synthetic'],{})),patch.object(score_panel,'local',return_value=NS(sources=lambda:[str(source)])):
                if mutate or terminal_failure:
                    with self.assertRaises(ValueError):score_panel.gate(args)
                else:
                    _,_,rows=score_panel.gate(args);self.assertEqual(rows,['synthetic']);exit_check.assert_called_once()
    def test_positive_full_acceptance(self):self.run_fixture()
    def test_tampering_or_live_verifier_fails(self):
        for bad in ('record','array','source'):self.run_fixture(mutate=bad)
        self.run_fixture(terminal_failure=True)

class AnalysisReviewTests(unittest.TestCase):
 def test_unfinished_review_fails_before_opening_source_proof(self):
  with patch.object(score_panel.panel_gate,'terminal',side_effect=ValueError('unfinished')),patch.object(score_panel,'read') as read:
   with self.assertRaisesRegex(ValueError,'unfinished'):score_panel.validate_analysis_review()
   read.assert_not_called()

class StatsReviewTests(unittest.TestCase):
 def test_live_or_forged_stats_review_fails(self):
  with patch.object(score_panel.panel_gate,'terminal',side_effect=ValueError('stats reviewer still live')),patch.object(score_panel,'read') as read:
   with self.assertRaisesRegex(ValueError,'still live'):score_panel.validate_statistics_review()
   read.assert_not_called()
  with patch.object(score_panel.panel_gate,'terminal'),patch.object(score_panel,'read',side_effect=[{'status':'passed','contrasts_verified':0},{},{}]),patch.object(score_panel,'sha',return_value='x'):
   with self.assertRaisesRegex(ValueError,'statistics review invalid'):score_panel.validate_statistics_review()
