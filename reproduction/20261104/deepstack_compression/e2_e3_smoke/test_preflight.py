import copy,unittest
from pathlib import Path
from local_imports import load_local
preflight=load_local('preflight')
class PreflightTests(unittest.TestCase):
    def test_import_paths_cuda_versions_fail_closed(self):
        root=Path('/tmp/author');plan={'author_root':str(root),'expected_versions':{'torch':'test'}}
        paths={'model_adapter':preflight.BASE/'e2_e3_contract/model_adapter.py','intervention_runner':preflight.HERE/'intervention_runner.py',
               'cache_fingerprints':preflight.HERE/'cache_fingerprints.py','native_pruner':root/'model/pruner.py',
               'author_wrapper':root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
        good={'status':'preflight_passed','model_loaded':False,'predictions':False,'cuda_initialized':False,
              'versions':{'torch':'test'},'imports':{k:str(v.resolve()) for k,v in paths.items()}}
        self.assertTrue(preflight.valid_result(good,plan))
        for field in ('model_loaded','predictions','cuda_initialized'):
            bad=copy.deepcopy(good);bad[field]=True;self.assertFalse(preflight.valid_result(bad,plan))
        for field in ('model_adapter','intervention_runner','cache_fingerprints','native_pruner','author_wrapper'):
            bad=copy.deepcopy(good);bad['imports'][field]='/tmp/wrong.py';self.assertFalse(preflight.valid_result(bad,plan))
        bad=copy.deepcopy(good);bad['versions']['torch']='wrong';self.assertFalse(preflight.valid_result(bad,plan))
if __name__=='__main__':unittest.main()
