import hashlib,json,tempfile,types,unittest
from pathlib import Path
from e1_refs import load,FIELDS
class Tests(unittest.TestCase):
 def test_bound_three_unique_no_answers_returned(self):
  anchors=[{'task':t,'qid':str(i),'group_id':str(i),'message_sha256':'m'} for i,t in enumerate(('D','T','A'))]
  rows=[]
  for a in anchors:
   r={k:None for k in FIELDS};r.update(a,arm='P3-D1',forced=False,answer='do not return');rows.append(r)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'j';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));receipt=types.SimpleNamespace(journal_sha=hashlib.sha256(p.read_bytes()).hexdigest());out=load(p,receipt,anchors)
   self.assertTrue(all('answer' not in x[0] for x in out.values()))
   p.write_text(p.read_text()+json.dumps(rows[0])+'\n');receipt.journal_sha=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):load(p,receipt,anchors)
   receipt.journal_sha='x'*64
   with self.assertRaises(ValueError):load(p,receipt,anchors)
if __name__=='__main__':unittest.main()
