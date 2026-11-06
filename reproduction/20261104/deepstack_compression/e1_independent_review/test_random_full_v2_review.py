import copy,unittest
import review_random_full_v2_result as v
def fixture():
 image='12'*32;seed=20261104;derived,ids=v.choose(image,seed);start=1;n=1026;positions=[[[i for i in range(n)]] for _ in range(3)];ret=list(range(start))+[start+x for x in ids]+list(range(start+1024,n));mask=[False]*len(ret);mask[start:start+256]=[True]*256
 r={'task':'T','qid':'1','group_id':'g','image_sha256':image,'message_sha256':'m','seed':seed,'derived_seed64':derived,'ordered_ids':ids,'spatial_ids':ids,'selection_mode':'e6_random_fixed_seed','budget':256,'ds_on':True,'greedy_calls':1,'extra_author_greedy':True,'actual_injection_verified':True,'timing_eligible':False,'forced':False,'protocol':'natural','processor':{},'main_visual':{},'full_deepstack':[{}, {}, {}],'full_position_ids':positions,'full_delta':[[0]],'image_start':start,'original_length':n,'retained_ids':ret,'prefill_position_ids':[[[x[0][i] for i in ret]] for x in positions],'adjusted_delta':[[768]],'injected_mask':[mask],'selected_deepstack':[{}, {}, {}],'injected_deepstack':[{}, {}, {}],'arrays':{'kind':'summary_only'},'output_ids':[9],'eos_ids':[9],'generation_scalars':{},'answer':'ok','eos':True,'truncated':False}
 r['decode_trace']=[{'position_ids':r['prefill_position_ids'],'cache_position':list(range(len(ret))),'rope_delta':[[768]],'layer_cache_lengths':[],'layer_cache_lengths_after':[len(ret)]*36}]
 c={k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256','seed','derived_seed64','spatial_ids')};e={k:r[k] for k in ('task','qid','group_id','message_sha256','processor','eos_ids','generation_scalars')};e.update(visual_sha256=r['main_visual'],deepstack_full_sha256=r['full_deepstack'],original_position_ids=positions,full_rope_delta=[[0]])
 return r,c,e
class Tests(unittest.TestCase):
 def test_valid_and_seed_determinism(self):
  r,c,e=fixture();self.assertIsNone(v.check(r,c,e,lambda _: 'ok'));self.assertEqual(v.choose(c['image_sha256'],c['seed']),(c['derived_seed64'],c['spatial_ids']))
 def test_tampering_is_rejected(self):
  for field,val in (('adjusted_delta',[[1]]),('seed',42),('answer','bad'),('timing_eligible',True),('injected_deepstack',[{}])):
   r,c,e=copy.deepcopy(fixture());r[field]=val
   with self.subTest(field=field),self.assertRaises(ValueError):v.check(r,c,e,lambda _: 'ok')
  r,c,e=copy.deepcopy(fixture());r['decode_trace'][0]['layer_cache_lengths_after'][2]=1
  with self.assertRaises(ValueError):v.check(r,c,e,lambda _: 'ok')
if __name__=='__main__':unittest.main()
