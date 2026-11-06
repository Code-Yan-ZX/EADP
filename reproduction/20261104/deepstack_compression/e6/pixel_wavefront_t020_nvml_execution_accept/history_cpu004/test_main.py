"""Run the real verifier main on nine fake jobs, without predictions or GPU."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contract,verify
class MainTest(unittest.TestCase):
 def test_nine_jobs_and_failure_are_not_silently_dropped(self):
  jobs=[{'order':i,'task':'fake','qid':str(i),'arm':'fake'} for i in range(9)];results=[{'status':'passed','job':j} for j in jobs];journal=[x for i,j in enumerate(jobs) for x in ({'event':'started','job':j},{'event':'completed','job':j,'result':results[i]})]
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);d=root/'formal';d.mkdir();e=root/'execution';e.mkdir();run=e/'run_001';run.mkdir();out=d/'verification001.json';source={'dummy':'digest'};agreement={'jobs':jobs,'plan_sha256':'digest','source_sha256':source};references={('fake',str(i),'fake'):{} for i in range(9)}
   def read(path):
    p=Path(path)
    if p.name=='verification_plan001.json':return {'gpu_plan_sha256':'digest','GPU_used':False,'GT_opened':False,'source_sha256':source}
    if p.name=='final.json':return {'status':'passed','jobs':9}
    if p.name=='memory_summary.json':return results[int(p.parent.name[-3:])]
    if p.name=='expected.json':return {'start_ticks':int(p.parent.name[-3:])+1}
    if p.name=='run_plan.json':return {'container_image':'fake'}
    raise AssertionError('unplanned fake read '+str(path))
   def module(name,*args):
    if name.endswith('_gate'):return SimpleNamespace(admit_gpu=lambda:agreement)
    if name.endswith('_refs'):return SimpleNamespace(load_refs=lambda:(references,{}))
    if name.endswith('_inventory'):return contract
    return SimpleNamespace()
   def job(job,*args,**kw):return {'job':job,'container_id':str(job['order']),'host_pid':job['order']+1}
   def execute(events):
    with contextlib.ExitStack() as st:
     for obj,name,value in ((verify,'D',d),(verify,'E',e),(verify,'sha',lambda x:'digest'),(verify,'read',read),(verify,'lines',lambda x:events),(verify,'module',module),(verify,'terminal',lambda *x:None),(verify,'verify_job',job),(sys,'argv',['verify.py','--out',str(out)])):st.enter_context(patch.object(obj,name,value))
     verify.main()
   execute(journal);self.assertTrue(out.exists());out.unlink()
   with self.assertRaises(ValueError):execute(journal[:-2])
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
