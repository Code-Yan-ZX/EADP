import json,tempfile,unittest
from pathlib import Path
from panel_worker import HERE,validate_launch_binding
import gate
class WorkerTests(unittest.TestCase):
 def test_binding_and_tampering_fail(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);approval={'review':'a'}
   l={'approval':approval,'worker_sha256':gate.sha(HERE/'panel_worker.py'),'launcher_sha256':gate.sha(HERE/'launch.py'),'budget_sha256':gate.sha(HERE/'resource_plan.json'),'controller':{'pid':123},'command':['docker',str(HERE/'panel_worker.py'),'--execute','--out',str(out)]}
   path=out/'launch.json';path.write_text(json.dumps(l));self.assertEqual(validate_launch_binding(out,approval),l)
   for key,value in (('approval',{}),('worker_sha256','wrong'),('launcher_sha256','wrong'),('budget_sha256','wrong'),('controller',None)):
    x=dict(l);x[key]=value;path.write_text(json.dumps(x))
    with self.assertRaises(ValueError):validate_launch_binding(out,approval)
   for cmd in ([str(HERE/'panel_worker.py'),'--execute','--out','/wrong'],[str(HERE/'panel_worker.py'),'--execute','--preflight-only','--out',str(out)]):
    x=dict(l,command=cmd);path.write_text(json.dumps(x))
    with self.assertRaises(ValueError):validate_launch_binding(out,approval)
if __name__=='__main__':unittest.main()

class ImportIdentityTests(unittest.TestCase):
 def test_loaded_class_has_real_file_and_collisions_fail(self):
  import inspect,sys
  from panel_worker import load_exact
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'example.py';path.write_text('class Example:\n    pass\n');name='_priortr_dev384_test_unique'
   try:
    mod=load_exact(name,path);self.assertEqual(Path(inspect.getfile(mod.Example)).resolve(),path.resolve())
    with self.assertRaises(ImportError):load_exact(name,path)
   finally:sys.modules.pop(name,None)
