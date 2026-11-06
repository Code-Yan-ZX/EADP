import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import b_gate,b_verify

class ExitTests(unittest.TestCase):
    def test_only_clean_bound_stopped_container_can_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);here=root/'b';here.mkdir();run=here/'b_001';run.mkdir()
            budget={'budget_source_sha256':'h','budget_end_unix':99999999,'outer_timeout_seconds':3720,
                'inner_timeout_seconds':3600,'journal_stale_seconds':900,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
            (here/'budget_plan.json').write_text(json.dumps(budget));(root/'initial_plan.json').write_text(json.dumps({'budget_end_unix':99999999}))
            ident={'pid':7,'start_ticks':10,'boot_id':'b'};child={'pid':8,'start_ticks':11,'boot_id':'b'}
            started={'worker_pid':7,'command':['python',str(here/'b_launch.py'),'--run-id','001'],'cwd':str(root),
                'timeout_seconds':3720,'stale_seconds':900,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90,
                'progress_paths':[str(run/'generated.jsonl')]}
            ended={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':7}
            launch={'controller':ident,'command':[str(here/'b_worker.py'),'--out',str(run)],'worker_sha256':'h',
                'launcher_sha256':'h','timeout_seconds':3600,'started_unix':1,'approval_bundle':{'ok':True}}
            receipt=run/'host_panel_gate.json';receipt.write_text(json.dumps({'ok':True}))
            launch['host_panel_gate']={'path':str(receipt),'sha256':'h'}
            exited={'controller':ident,'container_identity':{'container_id':'id','host_init':child},'returncode':0,
                'container_exit_code':0,'timed_out':False,'interrupted':None,'container_stopped_verified':True,
                'client_exited_while_container_running':False,'gpu_pids_after':[]}
            docker=SimpleNamespace(stdout=json.dumps([{'Id':'id','State':{'Running':False,'ExitCode':0}}]))
            with patch.object(b_verify,'HERE',here),patch.object(b_verify,'ROOT',root),patch.object(b_verify,'sha',lambda _: 'h'), \
                 patch.object(b_verify,'alive',return_value=False),patch.object(b_verify.local('b_gate'),'check_approval',return_value={'ok':True}), \
                 patch.object(b_verify.local('b_gate'),'host_validate_panel',return_value={'ok':True}), \
                 patch.object(b_verify.subprocess,'run',return_value=docker):
                b_verify.validate_terminal(run,started,ended,launch,exited)
                exited['container_stopped_verified']=False
                with self.assertRaises(ValueError):b_verify.validate_terminal(run,started,ended,launch,exited)

if __name__=='__main__':unittest.main()
