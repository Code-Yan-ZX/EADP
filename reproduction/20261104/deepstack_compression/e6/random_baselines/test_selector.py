import hashlib,json,unittest
from selector import selection
class SelectorTests(unittest.TestCase):
 def test_fixed_seed_is_deterministic_unique_and_spatial(self):
  for seed in (20261104,20261105,20261106):
   x=selection('01'*32,seed);self.assertEqual(x,selection('01'*32,seed));ids=x['spatial_ids']
   self.assertEqual(len(ids),256);self.assertEqual(ids,sorted(set(ids)));self.assertTrue(all(0<=i<1024 for i in ids))
 def test_image_and_seed_change_sample(self):
  self.assertNotEqual(selection('01'*32,20261104)['spatial_ids'],selection('01'*32,20261105)['spatial_ids'])
  self.assertNotEqual(selection('01'*32,20261104)['spatial_ids'],selection('02'*32,20261104)['spatial_ids'])
 def test_frozen_portable_vectors(self):
  expected={20261104:'7a8145042881455de8e9ddb624c401175e5e03296afe02b33d2f9cea753e8a12',20261105:'49b8c6ff58c57ba5abbaa50a01aaa13210a347e4314fcd698641effd558f84a2',20261106:'bbd375f973a9c64699a81db4d91876829ae91f5fb4f38924712510dce9da1b71'}
  for seed,digest in expected.items():self.assertEqual(hashlib.sha256(json.dumps(selection('01'*32,seed)['spatial_ids'],separators=(',',':')).encode()).hexdigest(),digest)
 def test_bad_identities_fail(self):
  for image,seed in [('z'*64,20261104),('ab',20261104),('ab'*32,1),('AB'*32,20261104)]:
   with self.assertRaises(ValueError):selection(image,seed)
if __name__=='__main__':unittest.main()
