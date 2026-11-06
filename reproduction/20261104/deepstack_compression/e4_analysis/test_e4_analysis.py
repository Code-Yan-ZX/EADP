import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import paired,score_e4
PLAN=json.loads((Path(__file__).parent/'plan.json').read_text())

def sample():
 panel=[];records=[]
 for task in score_e4.TASKS:
  for i in range(128):
   q={'task':task,'qid':str(i),'group_id':task+str(i),'message_sha256':'m'+str(i),'image_sha256':'i'+str(i)};panel.append(q)
   for k in (128,256,512,1024):
    for on in (False,True):records.append(dict(q,budget=k,ds_on=on,forced=False,answer='x'))
 return panel,records
class AdmissionTests(unittest.TestCase):
 def test_invalid_proof_prevents_reading_predictions_and_truth(self):
  args=types.SimpleNamespace(run=Path('/tmp/run'),panel_proof=Path('/tmp/no-proof'),verify_started=Path('/tmp/no-start'),verify_terminal=Path('/tmp/no-terminal'))
  with patch.object(score_e4,'validate_supervisor'),patch.object(score_e4,'read',return_value={'status':'failed'}),patch.object(score_e4.completion,'read_jsonl',side_effect=AssertionError('prediction accessed')),patch.object(score_e4.e1_score,'read_truths',side_effect=AssertionError('truth accessed')):
   with self.assertRaisesRegex(ValueError,'acceptance'):score_e4.gate(args)
 def test_formal_acceptance_without_independent_review_never_reads_data(self):
  args=types.SimpleNamespace(run=Path('/tmp/run'),panel_proof=Path('/tmp/proof'),verify_started=Path('/tmp/start'),verify_terminal=Path('/tmp/term'),review_proof=Path('/tmp/review'),review_started=Path('/tmp/reviewstart'),review_terminal=Path('/tmp/reviewterm'))
  formal={'status':'passed','stage':'full','n_questions':384,'all_arms_validated':3072,'k256_reuse':False,'GT_opened':False,'scoring_performed':False,'verifier_sha256':'h'}
  def fake_read(path):return formal if str(path)=='/tmp/proof' else {'status':'failed'}
  with patch.object(score_e4,'validate_supervisor'),patch.object(score_e4,'sha',return_value='h'),patch.object(score_e4,'read',side_effect=fake_read),patch.object(score_e4.completion,'read_jsonl',side_effect=AssertionError('prediction accessed')),patch.object(score_e4.e1_score,'read_truths',side_effect=AssertionError('GT accessed')):
   with self.assertRaisesRegex(ValueError,'independent'):score_e4.gate(args)
 def test_main_gate_failure_prevents_scoring_or_prediction_parsing(self):
  import sys
  names=('run','panel-proof','verify-started','verify-terminal','run-started','run-terminal','e1-acceptance','e1-started','e1-terminal','review-proof','review-started','review-terminal','out')
  argv=['score_e4.py']+[value for name in names for value in ('--'+name,'/tmp/no-data')]
  with patch.object(sys,'argv',argv),patch.object(score_e4,'sha',return_value=score_e4.PLAN_SHA256),patch.object(score_e4,'gate',side_effect=ValueError('gate failed')),patch.object(score_e4.completion,'read_jsonl',side_effect=AssertionError('prediction parsed')),patch.object(score_e4.e1_score,'read_truths',side_effect=AssertionError('GT read')):
   with self.assertRaisesRegex(ValueError,'gate failed'):score_e4.main()
 def test_v2_formal_supervisor_and_proof_reject_old_verifier(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run=root/'run';run.mkdir()
   names=('e1_acceptance','e1_started','e1_terminal','run_started','run_terminal')
   attrs={k:root/k for k in names}
   for p in attrs.values():p.write_text('{}')
   for n in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl'):(run/n).write_text('{}')
   proof=root/'proof';started=root/'started';terminal=root/'terminal'
   args=types.SimpleNamespace(run=run,panel_proof=proof,verify_started=started,verify_terminal=terminal,review_proof=root/'review',review_started=root/'review_started',review_terminal=root/'review_terminal',**attrs)
   accepted={str(p.resolve()):score_e4.sha(p) for p in [*attrs.values(),*[run/n for n in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl')]]}
   formal={'status':'passed','stage':'full','n_questions':384,'all_arms_validated':3072,'k256_reuse':False,'GT_opened':False,'scoring_performed':False,'verifier_sha256':score_e4.sha(score_e4.V2/'panel_verify.py'),'files_sha256':accepted}
   proof.write_text(json.dumps(formal));terminal.write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':123}))
   def write_start(script):started.write_text(json.dumps({'worker_pid':123,'cwd':str(root),'command':['python',str(script),'--out',str(proof)]}))
   with patch.object(score_e4,'validate_independent_review'),patch.object(score_e4,'validate_panel_sources'),patch.object(score_e4.panel_verify,'validate_terminal'),patch.object(score_e4,'validated_dev',return_value=({'GT_opened':False,'scoring_performed':False},[{}]*384)),patch.object(score_e4.completion,'read_jsonl',side_effect=AssertionError('prediction parsed')),patch.object(score_e4.e1_score,'read_truths',side_effect=AssertionError('GT read')):
    write_start(score_e4.V2/'panel_verify.py');score_e4.gate(args)
    write_start(score_e4.E4/'panel_verify.py')
    with self.assertRaisesRegex(ValueError,'wrong supervised verifier'):score_e4.gate(args)
    write_start(score_e4.V2/'panel_verify.py');formal['verifier_sha256']=score_e4.sha(score_e4.E4/'panel_verify.py');proof.write_text(json.dumps(formal))
    with self.assertRaisesRegex(ValueError,'acceptance'):score_e4.gate(args)
 def test_runtime_sources_reject_missing_old_path_and_tampering(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);v2=root/'e4_panel_v2';e4=root/'e4';v2.mkdir();e4.mkdir()
   files=[v2/x for x in score_e4.V2_SOURCES]+[e4/x for x in score_e4.E4_SOURCES]
   for p in files:p.write_text('original '+p.name)
   bindings={str(p.resolve()):score_e4.sha(p) for p in files};frozen={'code_sha256':bindings}
   score_e4.validate_panel_sources(frozen,v2,e4)
   bad=copy.deepcopy(frozen);del bad['code_sha256'][str((v2/'panel_verify.py').resolve())];bad['code_sha256'][str((e4/'panel_verify.py').resolve())]='old'
   with self.assertRaisesRegex(ValueError,'missing or changed'):score_e4.validate_panel_sources(bad,v2,e4)
   (v2/'panel_worker.py').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'missing or changed'):score_e4.validate_panel_sources(frozen,v2,e4)
 def test_dev_answer_filter_does_not_decode_nondevelopment_answers(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source.tsv';p.write_bytes(b'index\tanswer\n17\tvalid\n18\t\xff\xfe\n')
   src={'TextVQA_VAL':{'path':str(p),'sha256':score_e4.sha(p)}}
   self.assertEqual(score_e4.e1_score.read_truths(src,{('TextVQA_VAL','17')}),{('TextVQA_VAL','17'):{'answers':['valid']}})
 def test_independent_review_rejects_missing_mismatch_or_unfinished(self):
  good={'status':'passed','n_expected':384,'n_verified':384,'arms_expected':3072,'arms_verified':3072,'errors':[],
        'GT_opened':False,'scoring_performed':False,'formal_proof_sha256':'h','formal_verifier_started_sha256':'h',
        'formal_verifier_terminal_sha256':'h','reviewer_sha256':'h'}
  args=[Path('/tmp/review'),Path('/tmp/review-start'),Path('/tmp/review-term'),Path('/tmp/proof'),Path('/tmp/start'),Path('/tmp/term')]
  with patch.object(score_e4,'sha',return_value='h'),patch.object(score_e4,'validate_supervisor') as supervisor:
   with patch.object(score_e4,'read',return_value=good):score_e4.validate_independent_review(*args)
   supervisor.assert_called_once()
   for key,value in [('status','failed'),('n_verified',383),('arms_verified',3071),('errors',['bad']),('GT_opened',True),('scoring_performed',True),('formal_proof_sha256','wrong'),('formal_verifier_started_sha256','wrong'),('formal_verifier_terminal_sha256','wrong'),('reviewer_sha256','wrong')]:
    bad=dict(good);bad[key]=value
    with patch.object(score_e4,'read',return_value=bad):
     with self.assertRaises(ValueError,msg=key):score_e4.validate_independent_review(*args)
   with patch.object(score_e4,'read',return_value=good),patch.object(score_e4,'validate_supervisor',side_effect=ValueError('review running')):
    with self.assertRaises(ValueError):score_e4.validate_independent_review(*args)
 def test_full_inventory_rejects_missing_duplicated_changed_and_extra(self):
  panel,records=sample();data=score_e4.inventory(panel,records);self.assertEqual(len(data),384);self.assertEqual(sum(len(x) for x in data.values()),3072)
  for bad in (records[:-1],records+[records[-1]],records[:1]+records,records[1:]+records[:1]):
   with self.assertRaises(ValueError):score_e4.inventory(panel,bad)
  bad=copy.deepcopy(records);bad[0]['group_id']='wrong'
  with self.assertRaises(ValueError):score_e4.inventory(panel,bad)
  bad=copy.deepcopy(records);bad[0]['forced']=True
  with self.assertRaises(ValueError):score_e4.inventory(panel,bad)
 def test_xlsx_matches_author_na_handling(self):
  panel,records=sample();records[0]['answer']='';records[1]['answer']='NA'
  result=score_e4.native_predictions(records,PLAN['configs'])
  self.assertEqual(len(result),3072);self.assertEqual(result[(panel[0]['task'],panel[0]['qid'],'K128-D0')],'nan');self.assertEqual(result[(panel[0]['task'],panel[0]['qid'],'K128-D1')],'nan')
class StatsTests(unittest.TestCase):
 def test_required_cross_budget_contrasts_and_cluster_unit(self):
  configs=PLAN['configs'];rows=[]
  for group,n in [('a',2),('b',1)]:
   for i in range(n):
    vals={a:0.0 for a in configs};vals['K128-D1']=1.0 if group=='a' else 0.0;vals['K256-D1']=1.0;vals['K512-D0']=0.25
    answers={a:'same' for a in configs};answers['K128-D1']='changed' if group=='a' else 'same';rows.append({'group_id':group,'scores':vals,'answers':answers})
  result=paired.analyze(rows,configs,PLAN['contrasts'],replicates=100,seed=20261104);self.assertEqual(result['n_image_groups'],2);self.assertEqual(result['n_questions'],3)
  a=result['contrasts']['K128_D1_minus_K256_D0'];self.assertAlmostEqual(a['difference_pp'],200/3);self.assertEqual(a['answer_changed'],2)
  self.assertEqual(result['contrasts']['K256_D1_minus_K512_D0']['difference_pp'],75.0)
  self.assertEqual([x['budget'] for x in result['quality_curves']['D1']],[128,256,512,1024]);self.assertEqual(result['quality_curves']['D1'][1]['mean_pct'],100.0)
  self.assertEqual(result,paired.analyze(rows,configs,PLAN['contrasts'],replicates=100,seed=20261104))
 def test_manual_cluster_resampling_and_percentage_point_scale(self):
  import numpy as np
  configs=PLAN['configs'];rows=[]
  for group,n in [('a',2),('b',1),('c',3)]:
   for i in range(n):
    values={k:0.0 for k in configs};values['K128-D1']={'a':1.0,'b':0.0,'c':0.25}[group]
    rows.append({'group_id':group,'scores':values,'answers':{k:'' for k in configs}})
  n=300;seed=20261104;result=paired.analyze(rows,configs,PLAN['contrasts'],replicates=n,seed=seed)
  rng=np.random.default_rng(seed);draws=rng.integers(0,3,size=(n,3));manual=[]
  for draw in draws:
   sample=[r for idx in draw for r in rows if r['group_id']==('a','b','c')[idx]]
   manual.append(100*sum(r['scores']['K128-D1']-r['scores']['K256-D0'] for r in sample)/len(sample))
  actual=result['contrasts']['K128_D1_minus_K256_D0'];self.assertAlmostEqual(actual['difference_pp'],100*(2+0.75)/6)
  np.testing.assert_allclose(actual['image_cluster_percentile_95_ci_pp'],np.quantile(manual,[.025,.975]))
 def test_missing_arm_and_nonfinite_rejected(self):
  configs=PLAN['configs'];r={'group_id':'a','scores':{c:0.0 for c in configs},'answers':{c:'' for c in configs}}
  bad=copy.deepcopy(r);bad['scores'].pop(configs[0])
  with self.assertRaises(ValueError):paired.analyze([bad],configs,PLAN['contrasts'],10)
  bad=copy.deepcopy(r);bad['scores'][configs[0]]=float('nan')
  with self.assertRaises(ValueError):paired.analyze([bad],configs,PLAN['contrasts'],10)
if __name__=='__main__':unittest.main()
