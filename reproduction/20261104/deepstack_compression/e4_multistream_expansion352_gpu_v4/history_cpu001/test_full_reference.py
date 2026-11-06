"""Replay real accepted identity schema with synthetic tensor objects."""
import copy,importlib.util,json,unittest
from pathlib import Path
import full_reference as f
D=Path(__file__).resolve().parent;B=D.parent
spec=importlib.util.spec_from_file_location('_identity_contract_schema_test',B/'e4_c_microbench_pilot/identity_contract.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
class Tensor:
 def __init__(self,meta=None,value=None):self.meta=meta;self.value=value
 def detach(self):return self
 def cpu(self):return self
 def reshape(self,*_):return self
 def tolist(self):return self.value
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  d=json.loads((D/'schema_fixture001.json').read_text());cls.selected=d['selected'];cls.e1=d['e1_fields'];cls.e4={(r['task'],str(r['qid'])):r for r in d['e4_fields']};cls.slim={}
  for x in cls.selected:
   k=(x['task'],str(x['qid']));r=next(y for y in cls.e1 if (y['task'],str(y['qid']))==k);cls.slim[k]=dict(x,processor=r['processor'],image_start=r['image_start'])
  for i in range(375):cls.slim[('_unused',str(i))]={}
 def actual(self,r):
  return {'processor':{k:Tensor(meta=v) for k,v in r['processor'].items()},'visual':(Tensor(meta=r['visual_sha256']),[Tensor(meta=x) for x in r['deepstack_full_sha256']]),'ordered':Tensor(value=r['ordered_ids']),'prefill':{'inputs_embeds':Tensor(meta=r['prefill_embed_sha256']),'attention_mask':Tensor(meta=r['attention_mask_sha256']),'position_ids':Tensor(value=r['prefill_position_ids']),'visual_pos_masks':Tensor(value=r['injected_mask']),'deepstack_visual_embeds':[Tensor(meta=x) for x in r['injected_deepstack']]},'rope_delta':Tensor(value=r['rope_delta'])}
 def full(self):return f.collect(self.selected,copy.deepcopy(self.e1),self.slim,copy.deepcopy(self.e4))
 def test_nine_actual_schema(self):
  refs=self.full();self.assertEqual(len(refs),9)
  for ref in refs.values():self.assertEqual(identity.validate(self.actual(ref),ref,lambda t:t.meta)['status'],'passed')
 def test_incomplete_duplicate_dsoff_qid(self):
  a=copy.deepcopy(self.e1);del a[0]['visual_sha256']
  with self.assertRaises(ValueError):f.collect(self.selected,a,self.slim,self.e4)
  with self.assertRaises(ValueError):f.collect(self.selected,self.e1+[self.e1[0]],self.slim,self.e4)
  a=copy.deepcopy(self.e1);a[0]['arm']='P3-D0'
  with self.assertRaises(ValueError):f.collect(self.selected,a,self.slim,self.e4)
  a=copy.deepcopy(self.e1);a[0]['qid']='not-planned'
  with self.assertRaises(ValueError):f.collect(self.selected,a,self.slim,self.e4)
 def test_all_e4_identity_mapping_tampered(self):
  key=(self.selected[0]['task'],str(self.selected[0]['qid']))
  for k in f.E4_MAP.values():
   q=copy.deepcopy(self.e4);q[key][k]='tampered'
   with self.subTest(field=k),self.assertRaises(ValueError):f.collect(self.selected,self.e1,self.slim,q)
 def test_actual_validate_tampered(self):
  r=next(iter(self.full().values()))
  for name in ('main','ds0','ds1','ds2','ordered','position','delta','mask','prefill','attention','injected0','injected1','injected2'):
   a=self.actual(r)
   if name=='main':a['visual'][0].meta='tampered'
   elif name.startswith('ds'):a['visual'][1][int(name[-1])].meta='tampered'
   elif name=='ordered':a['ordered'].value=list(reversed(r['ordered_ids']))
   elif name=='position':a['prefill']['position_ids'].value='tampered'
   elif name=='delta':a['rope_delta'].value='tampered'
   elif name=='mask':a['prefill']['visual_pos_masks'].value='tampered'
   elif name=='prefill':a['prefill']['inputs_embeds'].meta='tampered'
   elif name=='attention':a['prefill']['attention_mask'].meta='tampered'
   else:a['prefill']['deepstack_visual_embeds'][int(name[-1])].meta='tampered'
   with self.subTest(field=name),self.assertRaises(ValueError):identity.validate(a,r,lambda t:t.meta)
 def test_slim_rejected_before_capture(self):
  a=copy.deepcopy(self.e1);a[0]=dict(self.slim[(a[0]['task'],str(a[0]['qid']))],arm='P3-D1',forced=False)
  with self.assertRaises(ValueError):f.collect(self.selected,a,self.slim,self.e4)
if __name__=='__main__':unittest.main()
