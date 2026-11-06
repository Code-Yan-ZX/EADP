import copy,unittest
from identity_contract import compare_pair,admit_timing,validate_pair_set
H='a'*64
M={'dtype':'torch.bfloat16','shape':[1,2,3],'sha256':H}
KV={'dtype':'torch.bfloat16','shape':[1,1,4,2],'sha256':H}
INPUT={'dtype':'torch.int64','shape':[1,1026],'sha256':H}
ANCHOR={'task':'DocVQA_VAL','qid':'16539','group_id':H,'image_sha256':H,'message_sha256':H}
RECIPE={'max_new_tokens':2048,'do_sample':False}
def record(path):
 x={**ANCHOR,'path':path,'status':'completed','GT_opened':False,'scoring_performed':False,'generation_scalars':RECIPE,'eos_ids':[2],'output_ids':[2],'selected_indices':[0,3],
    'original_sequence_length':1026,'original_max_position':1025,'prefill_mask':[[1,1,1,1]],'prefill_position_ids':[[[0,1,3,1025]]]*3,'rope_delta':[[0]],'visual_mask':[[False,True,True,False]],
    'main_visual':M,'injected_deepstack':[M,M,M],'prefill_embeddings':M,'prefill_logits':M,
    'processor':{'input_ids':INPUT,**{k:M for k in ('attention_mask','pixel_values','image_grid_thw')}},'decode_trace':[{'position_ids':[[[0,1,3,1025]]]*3,'cache_position':[0,1,2,3],'rope_delta':[[0]]}],
    'cache_fingerprints':[[{'key':KV,'value':KV} for _ in range(36)]], 'archive_path':'example.npz','archive_sha256':H}
 return copy.deepcopy(x)
class TestIdentity(unittest.TestCase):
 def test_positive_pair_and_three_pairs(self):
  a,w=record('author'),record('wavefront');self.assertTrue(compare_pair(a,w,ANCHOR,RECIPE));anchors=[{**ANCHOR,'qid':str(i),'group_id':str(i)} for i in range(3)];pairs=[]
  for anchor in anchors:
   x,y=record('author'),record('wavefront');x.update(anchor);y.update(anchor);pairs.append((x,y))
  self.assertTrue(validate_pair_set(pairs,anchors,RECIPE))
 def test_missing_both_sides_is_not_equal(self):
  a,w=record('author'),record('wavefront');del a['prefill_logits'];del w['prefill_logits']
  with self.assertRaises(ValueError):compare_pair(a,w,ANCHOR,RECIPE)
 def test_position_token_and_ds_changes(self):
  for key,value in [('output_ids',[1,2]),('selected_indices',[0,4]),('rope_delta',[[9]]),('injected_deepstack',[M,M]),('generation_scalars',{'max_new_tokens':2048,'do_sample':True})]:
   a,w=record('author'),record('wavefront');w[key]=value
   with self.subTest(key=key):
    with self.assertRaises(ValueError):compare_pair(a,w,ANCHOR,RECIPE)
 def test_cache_truncation(self):
  a,w=record('author'),record('wavefront');w['cache_fingerprints'][0].pop()
  with self.assertRaises(ValueError):compare_pair(a,w,ANCHOR,RECIPE)
 def test_identical_but_wrong_cache_and_position_are_rejected(self):
  for defect in ('cache','position','mask'):
   a,w=record('author'),record('wavefront')
   for x in (a,w):
    if defect=='cache':x['cache_fingerprints'][0][35]['key']['shape'][-2]=3
    if defect=='position':x['decode_trace'][0]['cache_position']=[0,1,2,7]
    if defect=='mask':x['prefill_mask'][0][2]=0
   with self.subTest(defect=defect),self.assertRaises(ValueError):compare_pair(a,w,ANCHOR,RECIPE)
 def test_no_review_or_partial_pairs(self):
  a,w=record('author'),record('wavefront')
  with self.assertRaises(RuntimeError):admit_timing([(a,w)]*3,[ANCHOR]*3,{**RECIPE},{'status':'passed','independent_review':True,'images':3,'GT_opened':False})
  with self.assertRaises(ValueError):validate_pair_set([(a,w)],[ANCHOR]*3,RECIPE)
  with self.assertRaises(ValueError):validate_pair_set([(a,w)]*3,[ANCHOR]*3,RECIPE)
if __name__=='__main__':unittest.main(verbosity=2)
