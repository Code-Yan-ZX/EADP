import copy,unittest
from inventory_contract import expected,validate
class Tests(unittest.TestCase):
 def fixture(self):
  anchors=[{'task':t,'qid':str(i),'group_id':str(i)} for i,t in enumerate(('D','T','A'))];jobs=expected(anchors);journal=[];generated=[];records=[]
  for job in jobs:
   rec={**job,'output_ids':[1],'eos_ids':[1],'generation_scalars':{},'protocol':'forced33' if job['forced'] else 'natural'};journal.extend([{'event':'started',**job},{'event':'recorded',**job}]);generated.append({'scheduled':job,'raw':rec});records.append(rec)
  complete={'status':'completed','arms':18,'natural':9,'forced33':9,'GT_opened':False,'scoring_performed':False,'timing_eligible':False}
  return anchors,journal,generated,records,complete
 def test_full_inventory(self):self.assertTrue(validate(*self.fixture()))
 def test_reject_missing_extra_changed_generation_and_bad_order(self):
  for kind in ('missing','extra','changed','order','incomplete'):
   a,j,g,r,c=self.fixture()
   if kind=='missing':g.pop()
   if kind=='extra':r.append(copy.deepcopy(r[0]))
   if kind=='changed':g[0]=copy.deepcopy(g[0]);g[0]['raw']['output_ids']=[999]
   if kind=='order':j[0],j[1]=j[1],j[0]
   if kind=='incomplete':c['arms']=17
   with self.subTest(kind=kind),self.assertRaises(ValueError):validate(a,j,g,r,c)
if __name__=='__main__':unittest.main()
