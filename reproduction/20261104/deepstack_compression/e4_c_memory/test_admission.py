import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import admission

class AdmissionTests(unittest.TestCase):
 def test_missing_review_rejects_before_hash_and_gpu(self):
  with patch.object(admission,'REVIEW_PROOF',Path('/definitely-missing-memory-review')):
   with self.assertRaisesRegex(ValueError,'independent review absent'):admission.admit()
 def test_actual_launcher_rejects_missing_review_before_output(self):
  import preflight_launch
  with patch('sys.argv',['preflight_launch.py','--run-id','notcreated']),patch.object(preflight_launch,'admit',side_effect=ValueError('review absent')):
   with self.assertRaisesRegex(ValueError,'review absent'):preflight_launch.main()
  self.assertFalse((admission.HERE/'container_preflightnotcreated').exists())
 def test_actual_launcher_no_gpu_at_mock_container_boundary(self):
  import preflight_launch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);worker=root/'preflight_entry.py';worker.write_text('worker');budget=root/'budget_plan.json';budget.write_text(json.dumps({'budget_source':'../unused'}));source=[worker,budget];approval={'source_sha256':{str(p):preflight_launch.sha(p) for p in source}}
   monitor=root/'monitor_container_preflightx';monitor.mkdir();(monitor/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'timeout_seconds':1500,'grace_seconds':90}))
   calls=[];mod=types.ModuleType('run_panel');mod.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu)) or ['docker','run','--rm','image',*args]);mod.identity=lambda pid:{'pid':pid}
   old=list(sys.path)
   try:
    with patch.object(preflight_launch,'HERE',root),patch.object(preflight_launch,'LATENCY',root),patch.object(preflight_launch,'sources',return_value=source),patch.object(preflight_launch,'admit',return_value=approval),patch.object(preflight_launch,'validate_budget_window'),patch.object(preflight_launch,'run_bounded',return_value=0) as bounded,patch.dict(sys.modules,{'run_panel':mod}),patch('sys.argv',['preflight_launch.py','--run-id','x']):
     self.assertEqual(preflight_launch.main(),0);self.assertIs(calls[0][1],False);self.assertNotIn('--rm',bounded.call_args.args[0]);self.assertIs(bounded.call_args.kwargs['require_gpu_idle'],False)
   finally:sys.path[:]=old
if __name__=='__main__':unittest.main()
