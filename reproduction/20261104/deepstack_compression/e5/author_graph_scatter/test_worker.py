import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
import primitive

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scatter_worker_test',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class TestWorker(unittest.TestCase):
    def test_primitives_on_cpu(self):
        for name in primitive.NAMES:
            mask=torch.zeros((1,4),dtype=torch.bool);idx=torch.tensor([2]);batch=torch.arange(1)
            sim=torch.arange(16,dtype=torch.float32).reshape(1,4,4);gain=torch.ones((1,4))
            if name=='boolean_mask_assign':mask[0,2]=True
            result=primitive.apply(name,mask,idx,batch,sim,gain)
            if name in ('index_assign','scatter_assign'):self.assertEqual(result.tolist(),[[False,False,True,False]])
            if name=='index_read':self.assertTrue(torch.equal(result,sim[:,2,:]))
            if name=='boolean_mask_assign':self.assertEqual(result.tolist(),[[1,1,-1e9,1]])
    def test_budget_and_case_constraints(self):
        good={'stage':'author scatter graph diagnosis','cases':list(w.CASES),'child_timeout_seconds':300,
              'allocator_fraction':.6,'k512_min_free_bytes':32*1024**3,'other_min_free_bytes':12*1024**3,'sha256':{}}
        for key,value in [('allocator_fraction',.9),('child_timeout_seconds',301),('k512_min_free_bytes',1),('cases',['k_512'])]:
            bad=dict(good);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):w.validate_plan(bad)
        with self.assertRaises(ValueError):w.validate_plan(good) # source closure absent
    def test_bounded_child_and_event(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'progress';w.event(path,{'event':'started'});w.event(path,{'event':'failed'})
            self.assertEqual([json.loads(x)['event'] for x in path.read_text().splitlines()],['started','failed'])
            p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            try:w.stop_child(p);self.assertIsNotNone(p.poll())
            finally:
                if p.poll() is None:p.kill();p.wait()
    def test_reference_and_candidate_are_separate_inputs(self):
        import probe
        self.assertEqual(Path(probe.__file__).resolve(),HERE/'probe.py')
        import inspect
        self.assertEqual(list(inspect.signature(probe.run_one).parameters),['k','case','author','candidate','emit'])

if __name__=='__main__':unittest.main(verbosity=2)
