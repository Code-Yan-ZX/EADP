import copy,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import extract,gate
class Tests(unittest.TestCase):
 def make(self):
  pilot=[];scores=[];panel=[]
  for task,n in [('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10)]:
   for i in range(n):
    row={'task':task,'qid':str(i),'group_id':str(i),'message_sha256':'m','image_sha256':'x'};pilot.append(row)
    scores.append({'task':task,'qid':str(i),'group_id':str(i),'arms':{'K%d-D1'%k:{'score':.3,'independent_score':.3} for k in (128,256,512)}})
    for k in (128,256,512):panel.append(dict(row,budget=k,ds_on=True,actual_injection_verified=True))
  return pilot,scores,panel
 def test_full_identity_and_tamper(self):
  pilot,scores,panel=self.make();self.assertEqual(len(extract.identities(pilot,scores,panel)),32)
  for typ in ('score','panel','missing','duplicate'):
   a=copy.deepcopy(scores);b=copy.deepcopy(panel)
   if typ=='score':a[0]['arms']['K128-D1']['independent_score']=.4
   if typ=='panel':b[0]['image_sha256']='wrong'
   if typ=='missing':b.pop()
   if typ=='duplicate':a.append(copy.deepcopy(a[0]))
   with self.assertRaises(ValueError,msg=typ):extract.identities(pilot,a,b)
 def test_prelaunch_failure_before_open(self):
  class Denied:
   def prelaunch(self):raise ValueError('prelaunch missing')
  with tempfile.TemporaryDirectory() as tmp,patch.object(extract,'D',Path(tmp)),patch.object(extract,'load',return_value=Denied()),patch.object(extract,'rows',side_effect=AssertionError('target opened')),patch.object(sys,'argv',['extract.py','--out',str(Path(tmp)/'result001.json')]):
   with self.assertRaisesRegex(ValueError,'prelaunch missing'):extract.main()
   self.assertFalse((Path(tmp)/'result001.json').exists())
 def test_selector_manifest_reordering_rejected(self):
  path=Path('/tmp/original_manifest.json');proof={'source_sha256':{str(path.resolve()):'original-sha'}}
  gate.require_manifest_binding(proof,path,'original-sha')
  with self.assertRaisesRegex(ValueError,'original manifest identity'):gate.require_manifest_binding(proof,path,'reordered-sha')
  with self.assertRaisesRegex(ValueError,'original manifest identity'):gate.require_manifest_binding(proof,Path('/tmp/other_manifest.json'),'original-sha')
 def test_changed_or_preloaded_source_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'m.py';sentinel=Path(tmp)/'executed';p.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').touch()')
   with self.assertRaises(ValueError):extract.load('_test_multistream',p,'0'*64)
   self.assertFalse(sentinel.exists())
   with patch.dict(sys.modules,{'_test_multistream':object()}):
    with self.assertRaises(ValueError):extract.load('_test_multistream',p,extract.sha(p))
if __name__=='__main__':unittest.main()
