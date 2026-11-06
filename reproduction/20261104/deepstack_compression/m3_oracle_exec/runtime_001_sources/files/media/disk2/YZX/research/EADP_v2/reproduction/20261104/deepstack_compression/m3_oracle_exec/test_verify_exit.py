import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent/'e4')]
import verify
class ExitTests(unittest.TestCase):
    def test_only_clean_identified_container_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'runtime_001';run.mkdir();events=str((run/'panel/panel_events.jsonl').resolve())
            plan={'outer_timeout_seconds':300,'progress_stale_seconds':50,'snapshot_seconds':3600}
            ident={'pid':20,'start_ticks':123,'boot_id':'boot'};container={'container_id':'abc','host_init':{'pid':21,'start_ticks':124,'boot_id':'boot'}}
            start={'worker_pid':20,'process_group':20,'timeout_seconds':300,'stale_seconds':50,'snapshot_seconds':3600,'progress_paths':[events],
                   'cwd':str(verify.HERE),'command':['python',str(verify.HERE/'runtime_launch.py'),'--run-id','001']}
            end={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':20}
            launch={'controller':ident,'command':['docker',str(verify.HERE/'runtime_worker.py'),'--execute','--out',str(run)],'worker_sha256':'h','launcher_sha256':'h'}
            ex={'controller':ident,'container_identity':container,'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,
                'container_stopped_verified':True,'client_exited_while_container_running':False,'errors':[],'gpu_pids_after':[]}
            inspect=types.SimpleNamespace(stdout=json.dumps([{'Id':'abc','State':{'Running':False,'Pid':0,'ExitCode':0}}]))
            module=types.ModuleType('run_panel');module.alive=lambda _:False
            with patch.dict(sys.modules,{'run_panel':module}),patch.object(verify.runtime_gate,'sha',return_value='h'),patch.object(verify.subprocess,'run',return_value=inspect),patch.object(verify.container_identity,'validate',return_value={}):
                verify.check_terminal(run,start,end,launch,ex,plan)
                ex['interrupted']=15
                with self.assertRaisesRegex(ValueError,'container exit'):verify.check_terminal(run,start,end,launch,ex,plan)
                ex['interrupted']=None;end['worker_pid']=99
                with self.assertRaisesRegex(ValueError,'supervisor'):verify.check_terminal(run,start,end,launch,ex,plan)
if __name__=='__main__':unittest.main()
