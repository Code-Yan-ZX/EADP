import copy,unittest
from identity_contract import validate
class T:
 def __init__(self,name,values=None):self.name=name;self.values=values
 def detach(self):return self
 def cpu(self):return self
 def reshape(self,*_):return self
 def tolist(self):return self.values
def meta(t):return {'name':t.name}
class Tests(unittest.TestCase):
 def test_real_input_fields_and_corruption(self):
  p={x:T(x) for x in ('input_ids','attention_mask','pixel_values','image_grid_thw')};v=[T('ds0'),T('ds1'),T('ds2')];streams=[T('sel0'),T('sel1'),T('sel2')];order=list(reversed(range(256)))
  actual={'processor':p,'visual':(T('main'),v),'ordered':T('ordered',order),'rope_delta':T('delta',[[5]]),'prefill':{'inputs_embeds':T('embed'),'attention_mask':T('mask'),'position_ids':T('pos',[[[1]]]),'visual_pos_masks':T('vmask',[[True]]),'deepstack_visual_embeds':streams}}
  ref={'processor':{k:meta(t) for k,t in p.items()},'visual_sha256':meta(T('main')),'deepstack_full_sha256':[meta(t) for t in v],'ordered_ids':order,'spatial_ids':sorted(order),'prefill_embed_sha256':meta(T('embed')),'attention_mask_sha256':meta(T('mask')),'prefill_position_ids':[[[1]]],'rope_delta':[[5]],'injected_mask':[[True]],'injected_deepstack':[meta(t) for t in streams]}
  self.assertEqual(validate(actual,ref,meta)['status'],'passed')
  for field,value in [('visual_sha256',{'name':'wrong'}),('rope_delta',[[6]]),('injected_deepstack',[{'name':'bad'}]*3),('ordered_ids',sorted(order))]:
   q=copy.deepcopy(ref);q[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):validate(actual,q,meta)
if __name__=='__main__':unittest.main(verbosity=2)
