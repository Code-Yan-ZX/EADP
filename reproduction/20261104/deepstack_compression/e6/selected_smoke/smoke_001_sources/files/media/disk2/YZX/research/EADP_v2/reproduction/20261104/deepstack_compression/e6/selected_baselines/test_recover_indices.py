import importlib.util,unittest
from pathlib import Path
import torch
from recover_indices import recover
ROOT=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/Qwen_vl/model')
class RecoverTests(unittest.TestCase):
 def test_exact_bf16_recovery_and_ambiguous_rejection(self):
  x=torch.arange(48,dtype=torch.float32).reshape(12,4).to(torch.bfloat16);self.assertEqual(recover(torch,x,x[[1,4,9]]),[1,4,9])
  for bad in (x[[4,1]],x[[1,1]],torch.full((1,4),999,dtype=x.dtype)):
   with self.assertRaises(ValueError):recover(torch,x,bad)
  dup=x.clone();dup[3]=dup[1]
  with self.assertRaisesRegex(ValueError,'ambiguous'):recover(torch,dup,dup[[1]])
 def test_unmodified_actual_pruners_return_recoverable_spatial_rows(self):
  torch.manual_seed(7);features=torch.randn(16,8);text=torch.randn(1,8);grid=torch.tensor([[1,8,8]])
  for file,klass in (('divpruner.py','DivPruner'),('cdpruner.py','CDPruner')):
   spec=importlib.util.spec_from_file_location('_eadp_test_'+klass,ROOT/file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);pruner=getattr(mod,klass)(visual_token_num=4,spatial_merge_size=2)
   output,sizes=pruner(features,text,grid);ids=recover(torch,features,output)
   self.assertEqual(sizes,[4]);self.assertEqual(len(ids),4);self.assertTrue(torch.equal(output,features[ids]))
if __name__=='__main__':unittest.main()
