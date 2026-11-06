import copy,unittest
from full_contract import split_schedule,COMPONENTS

def payload():
 images=[['D','g'+str(i)] for i in range(32)];identities=[{'task':a,'group_id':b} for a,b in images];jobs=[]
 for i,(a,b) in enumerate(images):
  for j in range(60):jobs.append({'task':a,'group_id':b,'order':60*i+j,'component':COMPONENTS[j%3]})
 return {'schedule':{'images':images,'jobs':jobs,'components':list(COMPONENTS)},'input_identities':identities}
class Tests(unittest.TestCase):
 def test_all_groups_and_original_global_order(self):
  groups=split_schedule(payload());self.assertEqual(len(groups),32);self.assertEqual(groups[1]['jobs'][0]['order'],0);self.assertEqual(groups[1]['jobs'][0]['global_order'],60);self.assertEqual(groups[-1]['jobs'][-1]['global_order'],1919)
 def test_bad_order_and_group_rejected(self):
  for mutate in ('order','group','identity','missing'):
   p=payload()
   if mutate=='order':p['schedule']['jobs'][100]['order']=3
   elif mutate=='group':p['schedule']['jobs'][100]['group_id']='wrong'
   elif mutate=='identity':p['input_identities'][1]['group_id']='wrong'
   else:p['schedule']['jobs'].pop()
   with self.assertRaises(ValueError):split_schedule(p)
if __name__=='__main__':unittest.main()
