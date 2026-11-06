import copy,unittest
import selection
class SelectionTests(unittest.TestCase):
 def fixtures(self):
  a=[];b=[]
  for t in selection.TASKS:
   for i in range(128):
    r=dict(task=t,qid=str(i),group_id=t+str(i),image_sha256='im'+str(i),message_sha256='msg'+str(i));a.append(r)
    if i<(10 if t=='AI2D_TEST' else 11):b.append(copy.deepcopy(r))
  return a,b
 def test_full_and_order(self):
  a,b=self.fixtures();x=selection.select(a,b);y=selection.select(a[::-1],b[::-1]);self.assertEqual(x,y);self.assertEqual(len(x['remaining']),352);self.assertEqual(len(x['pilot']),9);self.assertEqual(x['remaining_rows_in_pilot_groups'],0)
 def test_identity_tamper(self):
  a,b=self.fixtures();b[0]['image_sha256']='different'
  with self.assertRaises(ValueError):selection.select(a,b)
 def test_duplicate_row_and_group(self):
  a,b=self.fixtures();a[50]=copy.deepcopy(a[51])
  with self.assertRaises(ValueError):selection.select(a,b)
  a,b=self.fixtures();a[50]['group_id']=a[51]['group_id']
  with self.assertRaises(ValueError):selection.select(a,b)
 def test_cross_task_group_not_dropped(self):
  a,b=self.fixtures();a[128+50]['group_id']=a[50]['group_id'];x=selection.select(a,b)
  self.assertEqual(x['cross_task_shared_group_count'],1);self.assertEqual(len(x['remaining']),352)
 def test_missing_anchor(self):
  a,b=self.fixtures()
  with self.assertRaises(ValueError):selection.select(a,b[:-1])
if __name__=='__main__':unittest.main()
