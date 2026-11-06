import json
from pathlib import Path
import tempfile
import unittest
import torch
from local_imports import local
C=local('contract')

class ContractTests(unittest.TestCase):
    def test_target_ids_and_native_prefix_position(self):
        C.check_targets([100,200,300],512,[511]);p=torch.tensor([[[0,1,3,4]],[[0,1,3,4]],[[0,1,3,4]]])
        ref={'decode_trace':[{}, {'position_ids':[[[3]],[[3]],[[3]]]}, {'position_ids':[[[4]],[[4]],[[4]]]}]}
        C.check_prefix_positions(p,2,1,ref,2)
        with self.assertRaises(ValueError):C.check_targets([100,200,301],512,[])
        with self.assertRaises(ValueError):C.check_targets([100,200,300],512,[200])
        ref['decode_trace'][1]['position_ids'][0][0][0]=4
        with self.assertRaises(ValueError):C.check_prefix_positions(p,2,1,ref,2)
    def test_native_logit_tolerance_and_argmax(self):
        a=torch.tensor([[1.,2.,3.]]);b=torch.tensor([[1.,2.,3.125]])
        self.assertTrue(C.compare_logits(a,b,.125)['same_argmax'])
        with self.assertRaises(ValueError):C.compare_logits(a,b,.1)
        with self.assertRaises(ValueError):C.compare_logits(a,torch.tensor([[3.,2.,1.]]),4.)
        with self.assertRaises(ValueError):C.compare_logits(a,torch.tensor([[float('nan'),2.,3.]]),4.)
    def test_m3_anchor_rejects_modified_array_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);arr=d/'a';arr.write_bytes(b'array');row={'task':'x','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m'}
            sample=d/'sample';sample.write_text(json.dumps({'GT_opened':False,'predictions_opened':False,'examples':[row]}))
            ref=dict(row,path='stock_native',forced=False,arrays={'kind':'full_arrays','path':str(arr),'sha256':C.sha(arr)})
            records=d/'records';records.write_text(json.dumps(ref)+'\n');C.anchor(sample,records,'x','1','g')
            arr.write_bytes(b'changed')
            with self.assertRaises(ValueError):C.anchor(sample,records,'x','1','g')
            records.write_text((json.dumps(ref)+'\n')*2)
            with self.assertRaises(ValueError):C.anchor(sample,records,'x','1','g')

if __name__=='__main__':unittest.main()
