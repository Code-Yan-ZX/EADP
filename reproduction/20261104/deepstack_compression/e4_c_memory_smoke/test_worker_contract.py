import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import worker_contract as w
import worker

class TestWorkerContract(unittest.TestCase):
    def test_worker_fails_before_preparation_without_matching_approval(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);nonce='a'*32;sha='b'*64;allocation='c'*64;sources={'/x':'d'*64};calls=[]
            launch={'stage':'smoke','mode':'C-ON','run_dir':str(d.resolve()),'nonce':nonce,'worker_sha256':sha,
                    'source_sha256':sources,'allocation_sha256':allocation,'gpu_authorized':True}
            (d/'launch.json').write_text(json.dumps(launch))
            approval={'scope':'C-ON isolated memory smoke','stage':'smoke','mode':'C-ON',
                      'gpu_authorized':True,'source_sha256':sources,'allocation_sha256':allocation}
            def prepare(_):calls.append('prepare');return {'stage':'smoke','mode':'C-ON',
                      'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))}
            def continuation(*_):calls.append('continue');return 'synthetic'
            def digest(p):return sha if str(p)==str(Path(worker.__file__)) else 'd'*64
            self.assertEqual(worker.run(d,nonce,admit=lambda:approval,prepare=prepare,continue_run=continuation,sha=digest),'synthetic')
            self.assertEqual(calls,['prepare','continue']);calls.clear()
            with self.assertRaises(ValueError):worker.run(d,nonce,admit=lambda:dict(approval,gpu_authorized=False),prepare=prepare,continue_run=continuation,sha=digest)
            self.assertEqual(calls,[])
            with self.assertRaises(ValueError):worker.run(d,'f'*32,admit=lambda:approval,prepare=prepare,continue_run=continuation,sha=digest)
            self.assertEqual(calls,[])
            def initialized(_):return {'stage':'smoke','mode':'C-ON','torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))}
            with self.assertRaises(ValueError):worker.run(d,nonce,admit=lambda:approval,prepare=initialized,continue_run=continuation,sha=digest)
            with self.assertRaises(ValueError):worker.run(d,nonce,admit=lambda:approval,prepare=prepare,continue_run=continuation,sha=lambda _:sha)
            self.assertEqual(calls,[])

    def test_mismatch_never_passes(self):
        d=Path('/tmp/example');nonce='a'*32;sha='b'*64;allocation='c'*64;sources={'/x':'d'*64}
        launch={'stage':'smoke','mode':'C-ON','run_dir':str(d),'nonce':nonce,'worker_sha256':sha,
                'source_sha256':sources,'allocation_sha256':allocation,'gpu_authorized':True}
        self.assertEqual(w.validate(launch,directory=d,nonce=nonce,worker_sha=sha,reviewed_source_sha=sources,allocation_sha=allocation)['mode'],'C-ON')
        for field,value in [('stage','full'),('mode','C-SKIP'),('nonce','f'*32),('run_dir','/other'),
                            ('worker_sha256','0'*64),('source_sha256',{}),('allocation_sha256','0'*64),('gpu_authorized',False)]:
            bad=dict(launch);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):w.validate(bad,directory=d,nonce=nonce,worker_sha=sha,reviewed_source_sha=sources,allocation_sha=allocation)

if __name__=='__main__':unittest.main()
