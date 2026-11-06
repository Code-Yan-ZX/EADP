import copy,unittest
from gates import SYNTHETIC,synthetic_eligibility,endpoint_eligible,reconstruction_claim
class Tests(unittest.TestCase):
 def test_synthetic_and_e4_failures_reject_before_timing(self):
  ks=(4,4,4,8,1,8);rows=[{'case':c,'k':k,'author_ids':list(range(k)),'fused_ids':list(range(k)),'graph_ids':list(range(k)),'fused_equal':True,'graph_equal':True} for c,k in zip(SYNTHETIC,ks)]+[{'case':'graph_changed_input','k':4,'author_ids':[1,0,2,3],'graph_ids':[1,0,2,3],'before_graph_ids':[0,1,2,3],'graph_equal':True,'different_from_original':True}]
  self.assertEqual(synthetic_eligibility(rows),{'fused':True,'graph':True});bad=copy.deepcopy(rows);bad[1]['fused_ids']=[1,0,2,3];bad[1]['fused_equal']=False;self.assertEqual(synthetic_eligibility(bad),{'fused':False,'graph':True})
  author={x:True for x in ('e4_common_metadata_equal','e4_order_equal','e4_spatial_equal','e4_pruned_hash_equal')};cross={'available':True,'importance_equal_256':True,'similarity_equal_256':True};identity={'status':'passed'};recon={'status':'passed'}
  self.assertTrue(endpoint_eligible(identity,recon,True,author,cross,True));author['e4_order_equal']=False;self.assertFalse(endpoint_eligible(identity,recon,True,author,cross,True))
 def test_successful_mapping_cannot_be_falsely_reported_failed(self):
  actual={'sequence_equal':True,'mapped_indices':[2,3]};good={'status':'passed','sequence_equal':True,'mean_equal':True,'mapped_indices':[2,3]};self.assertTrue(reconstruction_claim(actual,good));bad=dict(good,status='failed')
  with self.assertRaises(ValueError):reconstruction_claim(actual,bad)
if __name__=='__main__':unittest.main()
