"""Small unlabeled ordering and identity gate tests."""
import json,tempfile,unittest
from pathlib import Path
from reorder import reorder
class Tests(unittest.TestCase):
 def setUp(self):self.items=[dict(position=i,index=str(11+i)) for i in range(3)]
 def lines(self,order):return [dict(position=i,index=str(11+i),arm='ds_off',response='r'+str(i)) for i in order]
 def run_case(self,rows):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'in';b=Path(d)/'out';a.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   n=reorder(a,b,self.items,'ds_off');return n,[json.loads(s) for s in b.read_text().splitlines()]
 def test_restores_order_without_changing_rows(self):
  source=self.lines([2,0,1]);n,out=self.run_case(source);self.assertEqual(n,3);self.assertEqual(out,self.lines([0,1,2]))
 def test_duplicate_missing_and_wrong_id_fail(self):
  for rows in (self.lines([0,0,1]),self.lines([0,1]),[dict(position=0,index='wrong',arm='ds_off')]+self.lines([1,2])):
   with self.assertRaises(ValueError):self.run_case(rows)
