import importlib.util
import sys
import unittest
from pathlib import Path
import torch

B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'e6/external_sources/pixelprune_8d347eb'))
from pixelprune.methods.pred_2d import Pred2DSelector
spec=importlib.util.spec_from_file_location('_tested_candidate',B/'e6/pixel_wavefront_cpu/candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)

class SelectorTests(unittest.TestCase):
    def test_actual_author_and_candidate_tiny_cpu_grids(self):
        for seed in range(4):
            gen=torch.Generator().manual_seed(seed)
            for h,w in ((1,1),(2,5),(4,3)):
                x=torch.randint(0,5,(h*w,3),generator=gen).float()/10
                author=Pred2DSelector(method='max',threshold=.2,anchored=True)
                expected=author._select_2d_loco_anchored(x,h,w,x.device)
                self.assertTrue(torch.equal(expected,candidate.select(x,h,w,.2)))

if __name__=='__main__':unittest.main()
