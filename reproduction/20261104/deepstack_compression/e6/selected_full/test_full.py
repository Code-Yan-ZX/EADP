import copy,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'selected_smoke'))
import torch
from test_stock_capture import StockTests
from identity_records import tensor_meta
from full_imports import prepare,load
full_runner=load('eadp_selected_full_runner',HERE/'full_runner.py')

class SummaryTests(unittest.TestCase):
 def test_full_stock_diff_only_changes_import_path_and_archive_contract(self):
  original=(HERE.parent/'selected_smoke/stock_capture.py').read_text()
  original=original.replace("HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E4=ROOT/'e4';BASE=HERE.parent/'selected_baselines'", "HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E4=ROOT/'e4';BASE=HERE.parent/'selected_baselines';SMOKE=HERE.parent/'selected_smoke';sys.path.insert(0,str(SMOKE))")
  original=original.replace('def run_stock(torch,w,method,row,generate,on_generated,archive,expected_scalars,forced_token=None):','def run_stock(torch,w,method,row,generate,on_generated,archive,expected_scalars,forced_token=None,expected_controls=None):')
  original=original.replace('            settings=controls(w.model.generation_config,kwargs,forced)\n',"            settings=controls(w.model.generation_config,kwargs,forced)\n            if settings!=expected_controls:raise ValueError('actual stock controls differ before generate')\n")
  a=original.index("    if (not isinstance(arrays,dict) or set(arrays)!");b=original.index("    return {'method':method",a)
  expected=original[:a]+"    if (not isinstance(arrays,dict) or set(arrays)!={'kind','tensors'} or arrays['kind']!='summary_only' or\n        arrays['tensors']!={k:tensor_meta(v) for k,v in tensors.items()}):\n        raise ValueError('complete runtime tensor summaries required')\n"+original[b:]
  self.assertEqual((HERE/'stock_capture.py').read_text(),expected)
 def test_real_stock_hooks_save_only_hashes(self):
  stock,_=prepare();w=StockTests().make();events=[];cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  row={'task':'fake','qid':'1','group_id':'g','split':'DEV','image_sha256':'x','message_sha256':'y'}
  def call(model,_):model.model.generate(**cfg);return ''
  def archive(tensors):return {'kind':'summary_only','tensors':{k:tensor_meta(v) for k,v in tensors.items()}}
  from generation_contract import controls
  control=controls(w.model.generation_config,cfg,False)
  rec=stock.run_stock(torch,w,'div',row,call,events.append,archive,cfg,expected_controls=control)
  self.assertEqual(rec['arrays']['kind'],'summary_only');self.assertEqual(rec['spatial_ids'],list(range(256)))
  self.assertTrue(all(set(x)=={'dtype','shape','sha256'} for x in rec['arrays']['tensors'].values()))
  wrong=dict(control);wrong['num_beams']=2;before=len(events)
  with self.assertRaisesRegex(ValueError,'before generate'):stock.run_stock(torch,StockTests().make(),'div',row,call,events.append,archive,cfg,expected_controls=wrong)
  self.assertEqual(len(events),before)
  def bad(tensors):
   out=archive(tensors);out['tensors']['visual']['sha256']='0'*64;return out
  with self.assertRaisesRegex(ValueError,'summaries'):stock.run_stock(torch,StockTests().make(),'div',row,call,events.append,bad,cfg,expected_controls=control)
 def test_full_runner_rejects_non384_before_generation(self):
  stock,contract=prepare()
  with patch.object(full_runner,'prepare',return_value=(stock,contract)):
   class FakePruner:pass
   class FakeWrapper:pass
   wrapper=FakeWrapper();wrapper.pruner=FakePruner()
   with self.assertRaisesRegex(ValueError,'inventory'):full_runner.run_method(torch,wrapper,FakeWrapper,FakePruner,'div',[],{},'/tmp',{})
 def test_384_questions_make_768_natural_markers_without_full_arrays(self):
  class FakePruner:pass
  class FakeModel:
   def __init__(self):self.model=types.SimpleNamespace(rope_deltas=None);self.generation_config=types.SimpleNamespace(eos_token_id=[2]);self.calls=0
   def generate(self,**kw):self.calls+=1;return [[2]]
  class FakeWrapper:
   def __init__(self):self.pruner=FakePruner();self.model=FakeModel()
   def generate_inner_transformers_fixed(self,message,dataset=None):
    self.model.generate(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.);return ''
  class FakeStock:
   def run_stock(self,torch,w,method,row,invoke,generated,archive,scalars,expected_controls=None):
    record={'method':method,'path':'author_stock','task':row['task'],'qid':row['qid'],'group_id':row['group_id'],
            'forced':False,'arrays':archive({'visual':torch.ones(2)})};generated(record);return record
  class FakeContract:
   def compare_pair(self,*args):return True
  def baseline(torch,w,method,row,invoke,generated,archive):
   invoke(w,row);record={'method':method,'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],
                         'forced':False,'arrays':archive({'visual':torch.ones(2)})};generated(record);return record
  tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');rows=[{'task':tasks[i//128],'qid':str(i),'group_id':'g'+str(i),'message':[]} for i in range(384)]
  recipe={'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.};writes=[];w=FakeWrapper()
  from generation_contract import controls
  cfg=dict(recipe,do_sample=False,max_new_tokens=2048);expected={t:controls(w.model.generation_config,cfg,False) for t in tasks}
  with patch.object(full_runner,'prepare',return_value=(FakeStock(),FakeContract())),patch.object(full_runner,'append',side_effect=lambda p,r:writes.append((p.name,r))),patch.dict(sys.modules,{'baseline_runner':types.SimpleNamespace(run_arm=baseline)}):
   result=full_runner.run_method(torch,w,FakeWrapper,FakePruner,'div',rows,recipe,'/tmp',expected)
  self.assertEqual(result['generations'],768);self.assertEqual(sum(name=='generated.jsonl' for name,_ in writes),768)
  self.assertEqual(sum(name=='pairs.jsonl' for name,_ in writes),384)
  self.assertTrue(all(r['arrays']['kind']=='summary_only' for name,r in writes if name=='records.jsonl'))
  wrong=copy.deepcopy(expected);wrong['DocVQA_VAL']['num_beams']=2;writes=[];w=FakeWrapper()
  with patch.object(full_runner,'prepare',return_value=(FakeStock(),FakeContract())),patch.object(full_runner,'append',side_effect=lambda p,r:writes.append((p.name,r))),patch.dict(sys.modules,{'baseline_runner':types.SimpleNamespace(run_arm=baseline)}):
   with self.assertRaisesRegex(ValueError,'adapted controls differ before generate'):
    full_runner.run_method(torch,w,FakeWrapper,FakePruner,'div',rows,recipe,'/tmp',wrong)
  self.assertEqual(w.model.calls,0);self.assertEqual(sum(name=='generated.jsonl' for name,_ in writes),1)
 def test_1535_generations_never_pass_completion(self):
  verify=load('eadp_selected_full_verify',HERE/'verify.py');approval={'approved':True}
  frozen={'stage':'full_dev','approval':approval,'GT_opened':False,'scoring':False,'full_tensors_archived':False,'plan_sha256':'x'}
  done={'status':'completed','questions':384,'generations':1536,'pairs':768,'GT_opened':False,'scoring':False}
  verify.require_completion(frozen,done,approval,'x')
  done['generations']=1535
  with self.assertRaisesRegex(ValueError,'completion'):verify.require_completion(frozen,done,approval,'x')
if __name__=='__main__':unittest.main()
