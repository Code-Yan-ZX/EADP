import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('oracle_preflight_test',Path(__file__).resolve().parent/'preflight.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PreflightTests(unittest.TestCase):
    def test_interrupted_unknown_or_unclean_never_pass(self):
        good={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'container_stopped_verified':True,'client_exited_while_container_running':False,'errors':[]}
        self.assertTrue(m.clean_exit(good))
        for k,v in [('interrupted',15),('timed_out',True),('container_stopped_verified',False),('errors',['stop failed']),('returncode',None)]:
            bad=dict(good);bad[k]=v;self.assertFalse(m.clean_exit(bad),k)
if __name__=='__main__':unittest.main()
