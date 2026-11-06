import importlib.util,io,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from e1_independent_review import review_train_audit_full as r
class IndependentReviewTests(unittest.TestCase):
 def test_independent_image_digest_matches_original(self):
  p=r.M/'build.py';s=importlib.util.spec_from_file_location('original_image_digest',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  rng=np.random.default_rng(42)
  for mode in ('PNG','JPEG'):
   raw=io.BytesIO();Image.fromarray(rng.integers(0,256,size=(19,27,3),dtype=np.uint8)).save(raw,format=mode);data=raw.getvalue()
   w,h,pix,bits=r.describe(data);size,original,b=m.describe(data);self.assertEqual(((w,h),pix,bits),(size,original,b))
 def test_transitive_components(self):
  d=r.DSU(5);d.union(0,1);d.union(3,4);d.union(1,4);self.assertEqual(d.find(0),d.find(3));self.assertNotEqual(d.find(0),d.find(2))
if __name__=='__main__':unittest.main()
