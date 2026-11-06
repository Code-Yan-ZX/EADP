import unittest
import numpy as np
from PIL import Image
from extract import features

class TestFeatures(unittest.TestCase):
 def test_blank_and_checkerboard(self):
  blank=Image.fromarray(np.zeros((256,256),dtype=np.uint8))
  entropy,gradient=features(blank)
  self.assertAlmostEqual(entropy,0);self.assertAlmostEqual(gradient,0)
  checker=((np.indices((256,256)).sum(axis=0)%2)*255).astype(np.uint8)
  entropy,gradient=features(Image.fromarray(checker))
  self.assertAlmostEqual(entropy,1);self.assertAlmostEqual(gradient,2)
if __name__=='__main__':unittest.main()
