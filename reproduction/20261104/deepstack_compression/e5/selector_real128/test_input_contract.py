import unittest,torch
from input_contract import compact_indices,compare_real
class Tests(unittest.TestCase):
 def test_text_before_after_and_unambiguous_mapping(self):
  ids=[1,2]+[9]*1024+[3,4,5];self.assertEqual(compact_indices(ids,[3,4],2,261),[258,259])
 def test_mapping_ambiguous_missing_and_cross_visual(self):
  ids=[1,2]+[9]*1024+[3,4,3,4]
  for ins in ([3,4],[8],[2,9]):
   with self.assertRaises(ValueError):compact_indices(ids,ins,2,262)
 def test_tensor_mapping_and_corruption(self):
  ids=[1,2]+[9]*1024+[3,4];prefill=torch.arange(260*3,dtype=torch.float32).reshape(1,260,3);mask=torch.zeros((1,260),dtype=torch.bool);mask[:,2:258]=True
  actual={'processor':{'input_ids':torch.tensor([ids])},'prefill':{'inputs_embeds':prefill,'visual_pos_masks':mask}};seq=prefill[:,258:260,:].clone();args=(None,seq.mean(dim=1),seq,None);r=compare_real(torch,actual,args,[3,4],2);self.assertTrue(r['sequence_equal']);self.assertTrue(r['mean_equal'])
  bad=list(args);bad[2]=seq+1;r=compare_real(torch,actual,bad,[3,4],2);self.assertFalse(r['sequence_equal'])
if __name__=='__main__':unittest.main()
