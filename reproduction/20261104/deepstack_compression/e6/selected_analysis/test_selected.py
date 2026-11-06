import copy,json,tempfile,unittest,sys,types
from pathlib import Path
from unittest.mock import patch
import score_selected
score_selected.load_scoring_modules(score_selected.admission.source_hashes())
gate=score_selected.gate;inventory=score_selected.inventory;paired=score_selected.paired

ARMS=list(inventory.ARMS)
def fake_panel():
 return [{'task':t,'qid':str(i),'group_id':t+str(i),'split':'DEV','image_sha256':'a'*64,'message_sha256':'b'*64} for t in inventory.TASKS for i in range(128)]
def fake_records(panel):
 return [{'task':r['task'],'qid':r['qid'],'group_id':r['group_id'],'image_sha256':r['image_sha256'],'message_sha256':r['message_sha256'],
          'method':a.split(':')[0],'path':a.split(':')[1],'forced':False,'answer':'','eos':False,'truncated':True,'output_ids':[1]} for r in panel for a in ARMS]
class InventoryTests(unittest.TestCase):
 def test_dev_subset_binds_accepted_input_hashes(self):
  p=fake_panel();raw=[{k:r[k] for k in ('task','qid','group_id','split')} for r in p];other=[{'task':'DocVQA_VAL','qid':'nondev','group_id':'other','split':x} for x in ('CONFIRM','FINAL')]
  self.assertEqual(inventory.build_dev_panel(raw+other,p),p)
 def test_dev_input_mismatch_rejected(self):
  p=fake_panel();raw=[{k:r[k] for k in ('task','qid','group_id','split')} for r in p];bad=copy.deepcopy(p);bad[0]['group_id']='wrong'
  with self.assertRaisesRegex(ValueError,'identity differ'):inventory.build_dev_panel(raw,bad)
 def test_complete_and_empty_answer_preserved(self):
  panel=fake_panel();result=inventory.inventory(panel,fake_records(panel))
  self.assertEqual(len(result),384);self.assertTrue(all(len(v)==4 for v in result.values()));self.assertEqual(result[(panel[0]['task'],'0')][ARMS[0]]['answer'],'')
 def test_duplicate_forced_or_wrong_input_rejected(self):
  panel=fake_panel();records=fake_records(panel)
  for kind in ('duplicate','forced','input','split'):
   p=copy.deepcopy(panel);r=copy.deepcopy(records)
   if kind=='duplicate':r[0]=copy.deepcopy(r[1])
   if kind=='forced':r[0]['forced']=True
   if kind=='input':r[-1]['image_sha256']='wrong'
   if kind=='split':p[0]['split']='FINAL'
   with self.subTest(kind=kind),self.assertRaises(ValueError):inventory.inventory(p,r)
class PairedTests(unittest.TestCase):
 def test_cluster_weighted_question_mean_and_reproducibility(self):
  def row(group,a):return {'group_id':group,'scores':{ARMS[0]:a,ARMS[1]:1,ARMS[2]:0,ARMS[3]:0},'answers':{x:x for x in ARMS}}
  rows=[row('g1',0),row('g1',1),row('g2',1)];contrasts={'a_minus_b':[ARMS[1],ARMS[0]]}
  x=paired.analyze(rows,ARMS,contrasts,replicates=300,seed=7);self.assertEqual(x,paired.analyze(rows,ARMS,contrasts,replicates=300,seed=7));self.assertEqual(x['n_questions'],3);self.assertEqual(x['n_image_groups'],2)
  self.assertAlmostEqual(x['arm_means_pct'][ARMS[0]],200/3);self.assertAlmostEqual(x['contrasts']['a_minus_b']['difference_pp'],100/3)
 def test_nonfinite_or_missing_score_rejected(self):
  r={'group_id':'g','scores':{a:0 for a in ARMS},'answers':{a:'' for a in ARMS}};r['scores'][ARMS[0]]=float('nan')
  with self.assertRaises(ValueError):paired.analyze([r],ARMS,{},replicates=2)
class GateTests(unittest.TestCase):
 def test_missing_full_proof_never_reads_metadata_or_truth(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'PROOF',Path(d)/'missing'),patch.object(gate,'read',side_effect=AssertionError('must not read')):
   with self.assertRaisesRegex(ValueError,'both acceptance'):gate.gate()
 def test_main_stops_before_prediction_or_truth_when_gate_fails(self):
  with patch('sys.argv',['score_selected.py','--out','unused']),patch.object(score_selected.admission,'admit',side_effect=ValueError('not ready')),patch.object(score_selected.completion,'read_jsonl',side_effect=AssertionError('prediction read')),patch.object(score_selected.e1_score,'read_truths',side_effect=AssertionError('truth read')):
   with self.assertRaisesRegex(ValueError,'not ready'):score_selected.main()
 def test_missing_scorer_review_is_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(score_selected.admission,'PRE',Path(d)/'missing'):
   with self.assertRaisesRegex(ValueError,'prelaunch review'):score_selected.admission.admit()
 def test_replaced_source_is_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'replacement.py';f.write_text('raise RuntimeError("executed")')
   with self.assertRaisesRegex(ValueError,'trusted source changed'):score_selected.load_exact('pollution_test',f,'0'*64)
 def test_same_name_pollution_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'candidate.py';f.write_text('value=1');sys.modules['pollution_test']=types.ModuleType('pollution_test')
   try:
    with self.assertRaisesRegex(ValueError,'preloaded module'):score_selected.load_exact('pollution_test',f,gate.sha(f))
   finally:del sys.modules['pollution_test']
 def test_e1_verified_sources_and_versions(self):
  self.assertIn('code_sha256',score_selected.admission.e1_identity())

 def test_source_manifest_detects_changed_snapshot(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);snapshot=root/'files/tmp';snapshot.mkdir(parents=True);(snapshot/'code.py').write_text('x');m=root/'manifest.json';m.write_text(json.dumps({'source_sha256':{'/tmp/code.py':gate.sha(snapshot/'code.py')}}));h=gate.sha(m)
   self.assertEqual(len(gate.manifest(m,h,1)),1);(snapshot/'code.py').write_text('changed')
   with self.assertRaisesRegex(ValueError,'snapshot changed'):gate.manifest(m,h,1)
 def test_formal_and_review_fields_do_not_accept_incomplete_or_wrong_binding(self):
  f={'status':'passed','stage':'full_dev','questions':384,'generations':1536,'pairs':768,'GT_opened':False,'scoring':False,'verifier_sha256':'hash','source_manifest_sha256':'manifest','run':str(gate.RUN)}
  r={'status':'passed','independent_review':True,'questions':384,'generations':1536,'GT_opened':False,'scoring_performed':False,'errors':[],'formal_proof_sha256':'hash','formal_started_sha256':'hash','formal_terminal_sha256':'hash','run_started_sha256':'hash','run_terminal_sha256':'hash','source_manifest_sha256':'manifest','reviewer_sha256':'hash'}
  with patch.object(gate,'sha',return_value='hash'):
   gate.validate_review_fields(f,r)
   for field,value in [('generations',1535),('GT_opened',True),('formal_proof_sha256','wrong'),('reviewer_sha256','wrong')]:
    bad=dict(r);bad[field]=value
    with self.subTest(field=field),self.assertRaises(ValueError):gate.validate_review_fields(f,bad)
class ScorerTests(unittest.TestCase):
 def test_real_author_and_independent_metrics_on_synthetic_truth(self):
  m=score_selected.metrics;author=m.author_functions();constants=m._author_constants()
  examples=[('DocVQA_VAL','abc',{'answers':['abc']}),('TextVQA_VAL','yes',{'answers':['yes']*10}),('AI2D_TEST','A',{'answer':'A','choices':{'A':'one','B':'two'}})]
  for task,pred,truth in examples:
   with self.subTest(task=task):
    value=m.score(task,pred,truth,author,constants);self.assertEqual(value['official'],1);self.assertEqual(value['independent'],1)
 def test_author_xlsx_roundtrip_preserves_missing_and_order(self):
  panel=fake_panel();records=fake_records(panel);records[0]['answer']='NA';records[1]['answer']='answer';out=score_selected.native_predictions(records,ARMS)
  self.assertEqual(out[(panel[0]['task'],'0',ARMS[0])],'nan');self.assertEqual(out[(panel[0]['task'],'0',ARMS[1])],'answer');self.assertEqual(len(out),1536)
if __name__=='__main__':unittest.main()

class TruthFilterTests(unittest.TestCase):
 def test_non_dev_undecodable_answer_is_never_decoded(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'fake.tsv';path.write_bytes(b'index\tanswer\nDEV1\t[\'ok\']\nFINAL1\t\xff\xfe\n')
   truth=score_selected.e1_score.read_truths({'DocVQA_VAL':{'path':str(path),'sha256':gate.sha(path)}},{('DocVQA_VAL','DEV1')})
   self.assertEqual(truth,{('DocVQA_VAL','DEV1'):{'answers':['ok']}})
