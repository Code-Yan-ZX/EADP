import copy,unittest
from design import panel,TASKS
def example():
 return [dict(task=t,qid=str(i),group_id=t+str(i),message_sha256='a'*64,image_sha256='b'*64,processor={k:{} for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},image_start=2) for t in TASKS for i in range(128)]
class TestPanel(unittest.TestCase):
 def test_complete_and_order_preserved(self):
  rows=example();self.assertEqual(panel(rows),rows)
 def test_missing_or_duplicate_rejected(self):
  rows=example()
  with self.assertRaises(ValueError):panel(rows[:-1])
  rows[-1]=copy.deepcopy(rows[0])
  with self.assertRaises(ValueError):panel(rows)
 def test_wrong_task_count_or_bad_identity(self):
  rows=example();rows[-1]['task']=TASKS[0]
  with self.assertRaises(ValueError):panel(rows)
  rows=example();rows[0]['image_sha256']='bad'
  with self.assertRaises(ValueError):panel(rows)
if __name__=='__main__':unittest.main()
