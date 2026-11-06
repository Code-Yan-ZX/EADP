import sys,unittest
from unittest import mock
from pathlib import Path
import torch
import worker
class TestThresholdProbe(unittest.TestCase):
 def test_reject_scope_before_input(self):
  with self.assertRaises(ValueError):worker.source_gate({'GT_opened':True})
 def test_author_selector_single_merged_patch(self):
  sys.path.insert(0,str(worker.PIXEL));from pixelprune.core import compute_merged_keep_indices
  from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
  pixels=torch.zeros((4,1536));grid=torch.tensor([[1,2,2]]);norm=normalize_pixel_values_for_selector(pixels)
  for t in (0.,.1):self.assertEqual(compute_merged_keep_indices(norm,grid,spatial_merge_size=2,method='pred_2d',metric='max',threshold=t,anchored=True)[0].tolist(),[0])
if __name__=='__main__':unittest.main()
