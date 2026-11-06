import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wiring

class WiringTest(unittest.TestCase):
    def test_preloaded_same_name_modules_fail_before_use(self):
        for name in ('latency_worker','latency_gate','real_worker','schedule','contract','runner',
                     'worker_candidate','worker_barrier','bridge','light_adapter','skip_deepstack','plan'):
            with self.subTest(name=name),patch.dict(sys.modules,{name:SimpleNamespace(__file__='/tmp/wrong.py')}):
                with self.assertRaisesRegex(ValueError,'source mismatch'):wiring.prepare({})
                with self.assertRaisesRegex(ValueError,'source mismatch'):wiring.continue_run({},Path('/tmp'),'a'*32,{})

    def test_preflight_gate_source_and_supervisor_mismatch(self):
        launch=wiring.load_named('eadp_test_smoke_preflight_launch',wiring.BASE/'e4_c_memory_smoke/preflight_launch.py')
        self.assertEqual(Path(launch.preflight_gate.__file__).resolve(),(wiring.BASE/'e4_c_memory_smoke/preflight_gate.py').resolve())
        plan=json.loads((wiring.BASE/'e4_c_memory_smoke/preflight_plan.json').read_text());cmd=plan['command'];child=cmd[cmd.index('--')+1:];base=str(wiring.BASE)
        started={'worker_pid':42,'process_group':42,'cwd':base,'timeout_seconds':2100.0,'grace_seconds':120.0,
                 'snapshot_seconds':3600.0,'poll_seconds':5.0,'stale_seconds':None,'progress_paths':[],'command':child}
        launch.validate_supervisor(plan,started,42,42,42,base,cmd,child,False)
        for field,value in [('worker_pid',1),('timeout_seconds',999),('grace_seconds',1),('command',['wrong'])]:
            bad=dict(started);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):launch.validate_supervisor(plan,bad,42,42,42,base,cmd,child,False)
        with self.assertRaises(ValueError):launch.validate_supervisor(plan,started,42,1,42,base,cmd,child,False)
        with self.assertRaises(ValueError):launch.validate_supervisor(plan,started,42,42,42,base,['wrong'],child,False)

    def test_actual_fixed_continuation_wiring_without_gpu(self):
        sys.path.insert(0,str(wiring.RUNTIME));sys.path.insert(0,str(wiring.LATENCY))
        import worker_candidate
        context={'prepared':{'marker':'p'},'expected_schedule':{'marker':'s'},'identity':{'marker':'i'},
                 'torch':object(),'native':object(),'wrapper_class':object()}
        with patch.object(worker_candidate,'run_after_preflight',side_effect=lambda **kw:kw):
            result=wiring.continue_run(context,Path('/tmp/example'),'a'*32,{})
        self.assertEqual(result['stage'],'smoke');self.assertEqual(result['mode'],'C-ON')
        self.assertEqual(result['prepared'],context['prepared'])
        self.assertEqual(result['expected_schedule'],context['expected_schedule'])
        self.assertEqual(Path(sys.modules['plan'].__file__).resolve(),(wiring.MEMORY/'plan.py').resolve())
        self.assertEqual(Path(result['run_memory_jobs'].__code__.co_filename).resolve(),(wiring.MEMORY/'worker.py').resolve())

if __name__=='__main__':unittest.main()
