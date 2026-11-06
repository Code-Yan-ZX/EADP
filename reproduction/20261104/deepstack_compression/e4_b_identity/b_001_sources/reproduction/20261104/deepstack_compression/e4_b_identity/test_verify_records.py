import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from verify_records import verify_arrays

def meta(a,dtype,shape):return {'dtype':dtype,'shape':shape,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}

class ArrayTests(unittest.TestCase):
    def test_real_array_readback_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'arrays').mkdir();path=root/'arrays/a.npz'
            visual=np.arange(1024,dtype=np.float32).view(np.uint8);selected=np.arange(256,dtype=np.float32).view(np.uint8)
            val=np.array([1.,2.],dtype=np.float32).view(np.uint8);inp=np.array([3],dtype=np.int64).view(np.uint8)
            data={'visual':visual.copy(),'pruned':selected.copy(),'prefill':selected.copy(),'prefill_logits':val.copy()}
            data.update({k:visual.copy() for k in ('ds0','ds1','ds2')})
            for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'):data['processor_'+k]=inp.copy()
            np.savez(path,**data);m=meta(visual,'torch.float32',[1024,1]);pm=meta(selected,'torch.float32',[256,1]);em=meta(selected,'torch.float32',[1,256,1]);lm=meta(val,'torch.float32',[2]);im=meta(inp,'torch.int64',[1])
            r={'mode':'C-DROP','main_visual':m,'pruned_visual':pm,'prefill_embeds':em,'first_logits':lm,
               'ordered_ids':list(range(256)),'spatial_ids':list(range(256)),'retained_ids':list(range(256)),
               'image_start':0,'original_length':1024,
               'processor':{k:im for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
               'full_deepstack':[m,m,m], 'arrays':{'kind':'full_arrays','path':str(path),
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            verify_arrays(r,root)
            data['visual'][0]^=1;np.savez(path,**data);r['arrays']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'array contents mismatch'):verify_arrays(r,root)

if __name__=='__main__':unittest.main()
