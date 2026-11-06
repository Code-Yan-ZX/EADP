import copy,unittest
from collections import Counter
from unittest.mock import patch
import contract as c
def fixture():
 rows=[];old=[];cohort={}
 for t,q in zip(c.TASKS,(11,11,10)):
  group=[]
  for i in range(128):
   r={'task':t,'qid':str(i),'group_id':f'{t}-{i:03d}','split':'DEV','image_path':'/fake','image_sha256':'a'*64,'message_sha256':'b'*64};rows.append(r);group.append(r)
  warm=group[:10];old.extend(warm+group[10:10+q]);cohort[t]={'warmup':[x['group_id'] for x in warm]}
 return rows,{'input_identities':old,'cohort':cohort}
class Tests(unittest.TestCase):
 def test_counts_pairing_deterministic_and_no_pilot_overlap(self):
  rows,p=fixture();a=c.build(rows,p);b=c.build(list(reversed(rows)),p)
  self.assertEqual(a,b);self.assertEqual((len(a['measurement_identities']),len(a['warmup_identities']),len(a['jobs'])),(192,30,7920));self.assertEqual(Counter(j['phase'] for j in a['jobs']),{'warmup':240,'measure':7680});self.assertFalse({(r['task'],r['group_id']) for r in a['measurement_identities']}&{(r['task'],r['group_id']) for r in p['input_identities']})
  count=Counter((j['task'],j['group_id'],j['budget'],j['protocol']) for j in a['jobs'] if j['phase']=='measure');self.assertEqual(len(count),192*8);self.assertEqual(set(count.values()),{5});self.assertEqual([j['order'] for j in a['jobs']],list(range(7920)))
 def test_reject_bad_identity_and_inventory(self):
  for change in ('duplicate_group','duplicate_qid','confirm','pilot_duplicate','pilot_identity','warm_missing','short'):
   r,p=copy.deepcopy(fixture())
   if change=='duplicate_group':r[1]['group_id']=r[0]['group_id']
   if change=='duplicate_qid':r[1]['qid']=r[0]['qid']
   if change=='confirm':r[20]['split']='CONFIRM'
   if change=='pilot_duplicate':p['input_identities'][1]=p['input_identities'][0]
   if change=='pilot_identity':p['input_identities'][1]=dict(p['input_identities'][1],image_sha256='c'*64)
   if change=='warm_missing':p['cohort'][c.TASKS[0]]['warmup'].pop()
   if change=='short':r.pop()
   with self.subTest(change=change),self.assertRaises(ValueError):c.build(r,p)
 def test_tie_breaks_are_deterministic(self):
  rows,p=fixture()
  with patch.object(c,'digest',return_value='0'*64):a=c.build(rows,p);b=c.build(rows[::-1],p)
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
