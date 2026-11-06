import copy,hashlib,tempfile,unittest
from pathlib import Path
import torch
from verify import verify_identity,sha

class IdentityTests(unittest.TestCase):
 def test_archive_and_tampering(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);a=run/'processor_inputs.pt'; keys=('input_ids','attention_mask','pixel_values','image_grid_thw');data={k:torch.tensor([i, i+1]) for i,k in enumerate(keys)};torch.save(data,a)
   meta={k:{'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()} for k,t in data.items()}
   ident={'task':'DocVQA_VAL','qid':'63709','group_id':'g','image_sha256':'i','message_sha256':'m','ordered_ids':[4,2],'processor':meta,'processor_archive':{'path':str(a),'sha256':sha(a)}}
   ref={k:copy.deepcopy(v) for k,v in ident.items() if k!='processor_archive'};ref.update(output_ids=[1,2],generation_scalars={'top_p':0.5},eos_ids=[2]);returned={k:ref[k] for k in ('output_ids','generation_scalars','eos_ids')}
   self.assertEqual(verify_identity(run,ident,returned,ref),meta)
   for field in ('ordered_ids','group_id','processor'):
    broken=copy.deepcopy(ident);broken[field]=None
    with self.assertRaises(ValueError):verify_identity(run,broken,returned,ref)
   bad=copy.deepcopy(returned);bad['output_ids']=[2]
   with self.assertRaises(ValueError):verify_identity(run,ident,bad,ref)
   data['pixel_values'][0]+=1;torch.save(data,a);ident['processor_archive']['sha256']=sha(a)
   with self.assertRaises(ValueError):verify_identity(run,ident,returned,ref)
if __name__=='__main__':unittest.main(verbosity=2)
