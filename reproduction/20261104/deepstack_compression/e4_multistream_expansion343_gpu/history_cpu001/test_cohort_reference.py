import copy,json,unittest
from collections import Counter
from pathlib import Path
import cohort,full_reference as f
D=Path(__file__).resolve().parent;B=D.parent
class Tests(unittest.TestCase):
 def test_actual_frozen_identity_only(self):
  x=json.loads((B/'e4_multistream_expansion352/selection001.json').read_text());a=cohort.build(x)
  self.assertEqual(len(a),343);self.assertEqual(Counter(y['task'] for y in a),{'DocVQA_VAL':114,'TextVQA_VAL':114,'AI2D_TEST':115})
  y=copy.deepcopy(x);y['pilot'][0]['image_sha256']='bad'
  with self.assertRaises(ValueError):cohort.build(y)
  y=copy.deepcopy(x);y['remaining'][0]=y['remaining'][1]
  with self.assertRaises(ValueError):cohort.build(y)
 def test_343_real_shape_synthetic_records_and_rejection(self):
  q=json.loads((B/'e4_multistream_expansion352_gpu_v4/schema_fixture001.json').read_text());selected=[];e1=[];e4={};slim={}
  for i in range(384):
   raw=copy.deepcopy(q['e1_fields'][i%9]);other=copy.deepcopy(q['e4_fields'][i%9]);r=dict(task='T',qid=str(i),group_id='G'+str(i),message_sha256='M'+str(i),image_sha256='I'+str(i));raw.update({k:r[k] for k in ('task','qid','group_id','message_sha256')});other.update(r);key=('T',str(i));slim[key]=dict(r,processor=raw['processor'],image_start=raw['image_start'])
   if i<343:selected.append(r);e1.append(raw);e4[key]=other
  self.assertEqual(len(f.collect(selected,e1,slim,e4)),343)
  for operation in ('missing','duplicate','ds_off','wrong_ref'):
   e=copy.deepcopy(e1);d=copy.deepcopy(e4)
   if operation=='missing':e.pop()
   elif operation=='duplicate':e.append(copy.deepcopy(e[0]))
   elif operation=='ds_off':e[0]['arm']='P3-D0'
   else:d[('T','0')]['prefill_position_ids']='bad'
   with self.subTest(operation=operation),self.assertRaises(ValueError):f.collect(selected,e,slim,d)
if __name__=='__main__':unittest.main()
