import unittest
from types import SimpleNamespace
from unittest.mock import patch
import worker
import preflight_gate


class WorkerPreflightTest(unittest.TestCase):
    def test_missing_independent_cpu_review_cannot_authorize(self):
        with patch.object(preflight_gate.gate,'admit_cpu',return_value={}), \
             patch.object(preflight_gate.rt,'supervisor',side_effect=ValueError('review missing')):
            with self.assertRaisesRegex(ValueError,'review missing'):preflight_gate.admit()

    def kwargs(self):
        prepared={'source_sha256':{'/s':'hash'},'rows':['r'],'excluded':['x'],
                  'schedule':{'jobs':[1]},'weights_before':dict.fromkeys('abcd')}
        return {'stage':'smoke','mode':'C-ON','approved_sources':{'/s':'hash'},
                'require_identity':lambda:{'status':'passed'},'prepare':lambda _:prepared,
                'load_runtime':lambda _:(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),1,2,{}),
                'build':lambda *args:{'jobs':[1]},'sha':lambda _:'hash'}
    def test_preparation_and_locks(self):
        k=self.kwargs();v=worker.prepare_cpu(**k);self.assertIn('existing DEV prediction journal opened',v['scope'])
        for key,value in [('stage','full'),('mode','C-SKIP'),('approved_sources',{}),('build',lambda *_:{'jobs':[]}),('sha',lambda _:'bad')]:
            k=self.kwargs();k[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.prepare_cpu(**k)
        k=self.kwargs();k['load_runtime']=lambda _:(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True)),1,2,{})
        with self.assertRaises(ValueError):worker.prepare_cpu(**k)

if __name__=='__main__':unittest.main()
