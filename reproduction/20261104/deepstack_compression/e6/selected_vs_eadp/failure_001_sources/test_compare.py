import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import compare as m
from compare import analyze,unique,key,require_proof,require_pair
class Comparison(unittest.TestCase):
 def test_fixed_paired_differences_and_reproducibility(self):
  rows=[{'task':'a','group_id':'g1','eadp':.5,'div':1.,'cd':0.},{'task':'a','group_id':'g2','eadp':0.,'div':.5,'cd':1.}]
  x=analyze(rows,replicates=500);self.assertEqual(x,analyze(rows,replicates=500));self.assertEqual(x['contrasts']['div-adapted_minus_eadp']['difference_pp'],50.);self.assertEqual(x['contrasts']['cd-adapted_minus_eadp']['difference_pp'],25.)
 def test_fail_invalid_and_duplicates(self):
  with self.assertRaises(ValueError):analyze([])
  with self.assertRaises(ValueError):analyze([{'group_id':'g','eadp':float('nan'),'div':0.,'cd':0.}])
  r={'task':'a','qid':'1','group_id':'g'}
  with self.assertRaises(ValueError):unique([r,r],key)
 def test_cluster_denominator(self):
  rows=[{'group_id':'g1','eadp':0.,'div':1.,'cd':0.},{'group_id':'g1','eadp':0.,'div':1.,'cd':0.},{'group_id':'g2','eadp':0.,'div':0.,'cd':0.}]
  x=analyze(rows,replicates=1000);self.assertEqual(x['n_questions'],3);self.assertEqual(x['n_groups'],2);self.assertAlmostEqual(x['contrasts']['div-adapted_minus_eadp']['difference_pp'],200/3)
 def test_proof_and_input_mutations_rejected(self):
  expected={'status':'passed','reviewer_sha256':'h'};require_proof(expected,expected)
  for p in ({**expected,'status':'failed'},{**expected,'reviewer_sha256':'bad'}):
   with self.assertRaises(ValueError):require_proof(p,expected)
  er={'message_sha256':'m','processor':{},'visual_sha256':{},'generation_scalars':{},'eos_ids':[1],'spatial_ids':list(range(256))};r={'image_sha256':'i','message_sha256':'m','processor':{},'main_visual':{},'generation_scalars':{},'eos_ids':[1],'spatial_ids':list(range(256))};inp={'image_sha256':'i','message_sha256':'m'};score={'score':.5,'independent_score':.5};require_pair(r,er,inp,score)
  for rr,ee,ii,ss in [({**r,'image_sha256':'wrong'},er,inp,score),(r,er,{**inp,'message_sha256':'wrong'},score),(r,{**er,'generation_scalars':{'top_k':9}},inp,score),(r,er,inp,{**score,'independent_score':0.})]:
   with self.assertRaises(ValueError):require_pair(rr,ee,ii,ss)
 def test_full_synthetic_384_and_admission_fail_closed(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);here=root/'e6/selected_vs_eadp';here.mkdir(parents=True);out=here/'result_001.json'
   def sha(p):return 'hash:'+str(p)
   sources={str((here/n).resolve()):sha(here/n) for n in ('plan.json','plan_002.json','plan_003.json','compare.py','test_compare.py','cpu_validation.py')}
   plan={'prior_plan_sha256':sha(here/'plan.json'),'revision_002_sha256':sha(here/'plan_002.json'),'numpy_version':m.np.__version__,'inputs_sha256':{},'strata':['ALL','DocVQA_VAL','TextVQA_VAL','AI2D_TEST'],'post_hoc_disclosure':'synthetic','constraints':[]}
   ep={'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,'GT_opened':False,'scoring_performed':False,'verifier_sha256':sha(root/'e1_analysis/completion.py'),'files':{n:{'path':str(root/n),'sha256':sha(root/n)} for n in ('journal','inputs')}}
   ei={'status':'passed','n_verified':384,'errors':[],'reviewer_sha256':sha(root/'e1_independent_review/review_full.py'),'completion_proof_sha256':sha(root/'e1/e1full001_completion.json'),'completion_started_sha256':sha(root/'e1/monitorverifye1full001/started.json'),'completion_terminal_sha256':sha(root/'e1/monitorverifye1full001/terminal.json')}
   selected=root/'e6/selected_full/panel_001/records.jsonl';sp={'status':'passed','stage':'full_dev','questions':384,'generations':1536,'GT_opened':False,'scoring':False,'verifier_sha256':sha(root/'e6/selected_full/verify.py'),'files_sha256':{str(selected):sha(selected)}}
   si={'status':'passed','independent_review':True,'questions':384,'generations':1536,'errors':[],'reviewer_sha256':sha(root/'e1_independent_review/review_selected_full_panel.py'),'formal_proof_sha256':sha(root/'e6/selected_full/panel_001_verification.json'),'formal_started_sha256':sha(root/'e6/selected_full/monitor_verify001/started.json'),'formal_terminal_sha256':sha(root/'e6/selected_full/monitor_verify001/terminal.json'),'run_started_sha256':sha(root/'e6/selected_full/monitor_panel001/started.json'),'run_terminal_sha256':sha(root/'e6/selected_full/monitor_panel001/terminal.json')}
   er={'status':'passed','n_verified':384,'arms_verified':1536,'errors':[],'score_sha256':sha(root/'e1/e1full001_score/scores.jsonl'),'e1_review_sha256':sha(root/'e1_independent_review/e1full001_review.json'),'score_terminal_sha256':sha(root/'e1/monitorscoree1full001/terminal.json'),'reviewer_sha256':sha(root/'e1_independent_review/review_score.py')}
   sr={'status':'passed','independent_review':True,'questions':384,'scores_verified':1536,'CONFIRM_FINAL_answers_parsed':False,'scores_sha256':sha(root/'e6/selected_analysis/score002/scores.jsonl'),'formal_sha256':sha(root/'e6/selected_full/panel_001_verification.json'),'panel_review_sha256':sha(root/'e1_independent_review/selected_full_panel001_review.json'),'score_started_sha256':sha(root/'e6/selected_analysis/monitor_score002/started.json'),'score_terminal_sha256':sha(root/'e6/selected_analysis/monitor_score002/terminal.json'),'reviewer_sha256':sha(root/'e1_independent_review/review_e6_selected_scores.py')}
   cp={'status':'passed','source_sha256_before':sources,'source_sha256_after':sources};rv={'status':'passed','independent_review':True,'source_sha256':sources,'reviewer_sha256':sha(root/'e1_independent_review/review_selected_vs_eadp.py'),'cpu_sha256':sha(here/'cpu_check_005/verification.json'),'cpu_started_sha256':sha(here/'monitor_cpu005/started.json'),'cpu_terminal_sha256':sha(here/'monitor_cpu005/terminal.json')}
   objs={str(here/'plan_003.json'):plan,str(here/'cpu_check_005/verification.json'):cp,str(here/'independent_prelaunch_review.json'):rv,str(root/'e1/e1full001_completion.json'):ep,str(root/'e1_independent_review/e1full001_review.json'):ei,str(root/'e6/selected_full/panel_001_verification.json'):sp,str(root/'e1_independent_review/selected_full_panel001_review.json'):si,str(root/'e1_independent_review/e1full001_score_review.json'):er,str(root/'e1_independent_review/e6_selected_scores_review.json'):sr}
   raw=[];inputs=[];chosen=[];es=[];ss=[]
   for n in range(384):
    task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[n//128];base={'task':task,'qid':str(n),'group_id':'g'+str(n)};shared={'message_sha256':'m','processor':{},'generation_scalars':{},'eos_ids':[1],'spatial_ids':list(range(256)),'answer':'','eos':True,'truncated':False}
    raw.append({**base,**shared,'arm':'P3-D1','forced':False,'visual_sha256':{}});inputs.append({**base,'message_sha256':'m','image_sha256':'i'});es.append({**base,'arms':{'P3-D1':{'score':.5,'independent_score':.5,'raw_answer':'','eos':True,'truncated':False}}});arms={}
    for method in ('div','cd'):
     chosen.append({**base,**shared,'method':method,'path':'adapted','image_sha256':'i','main_visual':{}});arms[method+':adapted']={'score':.5,'independent_score':.5,'answer':'','eos':True,'truncated':False}
    ss.append({**base,'arms':arms})
   records={str(root/'e1/e1full001/raw_predictions.jsonl'):raw,str(root/'e1/input_preparation_002/inputs.jsonl'):inputs,str(selected):chosen,str(root/'e1/e1full001_score/scores.jsonl'):es,str(root/'e6/selected_analysis/score002/scores.jsonl'):ss}
   with patch.multiple(m,ROOT=root,HERE=here,PLAN=sha(here/'plan_003.json')),patch.object(m,'sha',side_effect=sha),patch.object(m,'read',side_effect=lambda p:objs[str(p)]),patch.object(m,'lines',side_effect=lambda p:records[str(p)]),patch.object(sys,'argv',['compare.py','--out',str(out)]):
    with patch.object(m,'supervised',side_effect=ValueError('review pending')),patch.object(m,'lines',side_effect=AssertionError('prediction read before admission')):
     with self.assertRaisesRegex(ValueError,'review pending'):m.main()
    with patch.object(m,'supervised',return_value=None):m.main()
   result=json.loads(out.read_text());self.assertEqual(result['reports']['ALL']['n_questions'],384);self.assertEqual(result['reports']['ALL']['contrasts']['div-adapted_minus_eadp']['difference_pp'],0.)
if __name__=='__main__':unittest.main(verbosity=2)
