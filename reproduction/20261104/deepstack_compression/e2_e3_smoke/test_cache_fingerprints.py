import types,unittest,torch
from local_imports import load_local
CacheFingerprints=load_local('cache_fingerprints').CacheFingerprints
class LM(torch.nn.Module):
    def forward(self,value):
        layers=[types.SimpleNamespace(keys=value+i,values=value-i) for i in range(36)]
        return types.SimpleNamespace(past_key_values=types.SimpleNamespace(layers=layers))
class CacheTests(unittest.TestCase):
    def test_actual_36_layer_content_captured_each_forward_and_hook_removed(self):
        lm=LM();w=types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(language_model=lm)))
        with CacheFingerprints(w,True) as cache:
            lm(torch.zeros((1,1,3,2)));lm(torch.ones((1,1,4,2)))
            self.assertEqual(len(cache.steps),2);self.assertEqual(len(cache.steps[0]),36)
            self.assertEqual(cache.steps[1][0]['key']['shape'],[1,1,4,2])
            self.assertNotEqual(cache.steps[0][0]['key']['sha256'],cache.steps[1][0]['key']['sha256'])
            self.assertNotEqual(cache.steps[0][1]['key']['sha256'],cache.steps[0][1]['value']['sha256'])
        self.assertEqual(len(lm._forward_hooks),0)
    def test_disabled_registers_no_hook(self):
        lm=LM();w=types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(language_model=lm)))
        with CacheFingerprints(w,False) as cache:lm(torch.zeros((1,1,3,2)))
        self.assertEqual(cache.steps,[])
if __name__=='__main__':unittest.main()
