import unittest
from core import compute
def row(q,d,g,s='fit'):return {'question_id':q,'document_id':d,'group_id':g,'split':s}
class Tests(unittest.TestCase):
 def test_eval_group_transitive_chain(self):
  rows=[row(1,'a','g1','excluded_eval'),row(2,'b','g1','excluded_eval'),row(3,'b','g2'),row(4,'c','g2'),row(5,'isolated','g3')]
  out,s=compute(rows,set());d={r['question_id']:r['split'] for r in out}
  self.assertTrue(all(d[x]=='excluded_eval' for x in (1,2,3,4)));self.assertNotEqual(d[5],'excluded_eval');self.assertEqual(s['max_component_rows']['excluded_eval'],4)
 def test_eval_document_transitive_chain(self):
  rows=[row(1,'evaldoc','g1'),row(2,'other','g1'),row(3,'other','g2')]
  out,_=compute(rows,{'evaldoc'});self.assertTrue(all(x['split']=='excluded_eval' for x in out))
 def test_order_and_atomic_documents(self):
  rows=[row(1,'a','g1'),row(2,'a','g2'),row(3,'b','g3'),row(4,'c','g4'),row(5,'d','g5')]
  a,s=compute(rows,set());b,t=compute(list(reversed(rows)),set());self.assertEqual((a,s),(b,t));self.assertEqual(a[0]['split'],a[1]['split']);self.assertEqual(sum(s['row_counts'].values()),5)
 def test_original_group_conflict_and_bad_inputs(self):
  for rows in ([row(1,'a','g','fit'),row(2,'b','g','excluded_eval')],[row(1,'a','g'),row(1,'b','h')],[row(1,'','g')],[row(1,'a','g','unknown')]):
   with self.assertRaises(ValueError):compute(rows,set())
  with self.assertRaises(ValueError):compute([],{' '})
if __name__=='__main__':unittest.main(verbosity=2)
