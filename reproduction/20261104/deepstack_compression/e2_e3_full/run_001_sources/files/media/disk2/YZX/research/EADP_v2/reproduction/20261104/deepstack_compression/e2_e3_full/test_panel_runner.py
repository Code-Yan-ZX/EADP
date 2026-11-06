import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from panel_runner import run_panel,validate_against_e1,validate_endpoint,E1_INPUT_FIELDS,ARMS

class PanelCoreTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'task':task,'qid':str(i),'group_id':task+str(i),'image_sha256':'a'*64,'message_sha256':'b'*64,'split':'DEV'} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
  self.refs={}
  for row in self.rows:
   for arm in ('P3-D0','P3-D1'):
    self.refs[(row['task'],row['qid'],arm)]={**{v:'reference_'+v for v in E1_INPUT_FIELDS.values()},'generation_scalars':{'max_new_tokens':2048},'output_ids':[2],'answer':'ok','eos_ids':[2],'eos':True,'truncated':False,'logits':['logit']}
  self.anchors=[self.rows[0],self.rows[128],self.rows[256]]
 def invoke(self,row,family,arm,journal,capture,need_logits):
  j={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'budget':256,'ds_on':True,'output_ids':[2],'eos_ids':[2],'generation_scalars':{'max_new_tokens':2048},'forced':False};journal(j)
  ref=self.refs[(row['task'],row['qid'],'P3-D1')]
  return dict(j,family=family,arm=arm,**{key:ref[value] for key,value in E1_INPUT_FIELDS.items()},image_sha256=row['image_sha256'],message_sha256=row['message_sha256'],answer='ok',eos=True,truncated=False,actual_injection_verified=True,prefill_logits='logit' if need_logits else None,arrays=capture({}))
 def test_full_matrix_anchor_retention_and_aliases(self):
  events=[]
  with tempfile.TemporaryDirectory() as root,patch('panel_runner.append',side_effect=lambda path,x:events.append((Path(path).name,x))):
   result=run_panel(rows=self.rows,e1_refs=self.refs,anchors=self.anchors,out=Path(root)/'run',invoke=self.invoke,archive=lambda t,p:{'kind':'full_arrays','path':str(p),'sha256':'a'*64},summary=lambda t:{})
   self.assertEqual(result['physical_generations'],3840)
   self.assertEqual(sum(n=='generated.jsonl' for n,_ in events),3840)
   self.assertEqual(sum(n=='records.jsonl' for n,_ in events),3840)
   self.assertEqual(sum(n=='e3_views.jsonl' for n,_ in events),384)
   records=[r for n,r in events if n=='records.jsonl'];self.assertEqual(sum(r['arrays']['kind']=='full_arrays' for r in records),30)
   views=[r for n,r in events if n=='e3_views.jsonl'];self.assertEqual(views[0]['arms']['OFF']['source_arm'],'000');self.assertEqual(views[0]['arms']['MATCH']['source_arm'],'111')
 def test_generation_journal_survives_later_failure(self):
  def bad(row,family,arm,journal,capture,need_logits):
   journal({'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'budget':256,'ds_on':True,'output_ids':[2],'eos_ids':[2],'generation_scalars':{},'forced':False});raise RuntimeError('after generation')
  with tempfile.TemporaryDirectory() as root:
   out=Path(root)/'run'
   with self.assertRaises(RuntimeError):run_panel(rows=self.rows,e1_refs=self.refs,anchors=self.anchors,out=out,invoke=bad,archive=lambda *_:None,summary=lambda _:None)
   self.assertEqual(len((out/'generated.jsonl').read_text().splitlines()),1);self.assertFalse((out/'worker_complete.json').exists())
 def test_mismatched_endpoints_inputs_and_partial_are_rejected(self):
  def bad_endpoint(row,family,arm,journal,capture,need_logits):
   x=self.invoke(row,family,arm,journal,capture,need_logits);x['prefill_logits']='wrong';return x
  with tempfile.TemporaryDirectory() as root:
   with self.assertRaisesRegex(ValueError,'first-logit'):run_panel(rows=self.rows,e1_refs=self.refs,anchors=self.anchors,out=Path(root)/'r',invoke=bad_endpoint,archive=lambda t,p:{'kind':'full_arrays'},summary=lambda t:{})
  ref=self.refs[(self.rows[0]['task'],self.rows[0]['qid'],'P3-D1')];rec=self.invoke(self.rows[0],'E2','000',lambda _:None,lambda _:{'kind':'full_arrays'},True);rec['processor']='changed'
  with self.assertRaisesRegex(ValueError,'processor'):validate_against_e1(rec,self.rows[0],ref)
  self.assertEqual(len(ARMS),10)
if __name__=='__main__':unittest.main()
