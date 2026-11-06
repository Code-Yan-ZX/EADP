import unittest
import design
class Tests(unittest.TestCase):
 def test_nine_unique_three_arms_and_seed(self):
  anchors=[{'task':str(i),'qid':str(i),'group_id':str(i)} for i in range(3)]
  jobs=design.make(anchors);self.assertEqual(jobs,design.make(anchors));self.assertEqual([j['order'] for j in jobs],list(range(9)))
  self.assertEqual(len({(j['task'],j['arm']) for j in jobs}),9)
  for i in range(3):self.assertEqual({j['arm'] for j in jobs if j['task']==str(i)},{'t020_author','t020_wavefront','t0_author'})
 def test_duplicate_task_rejected(self):
  with self.assertRaises(ValueError):design.make([{'task':'x','qid':str(i),'group_id':str(i)} for i in range(3)])
if __name__=='__main__':unittest.main()
