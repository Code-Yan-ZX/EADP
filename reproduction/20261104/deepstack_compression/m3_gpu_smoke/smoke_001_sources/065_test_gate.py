import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from local_imports import local,teacher
launcher=local('launcher')

class GateTests(unittest.TestCase):
    def test_missing_approval_and_conflicting_teacher_import_are_rejected(self):
        g=local('gate')
        with patch.object(g,'APPROVAL',Path('/nonexistent/m3-approval.json')):
            with self.assertRaisesRegex(ValueError,'locked'):g.check_approval()
        fake=types.ModuleType('teacher_contract');fake.__file__='/tmp/unapproved_teacher.py'
        with patch.dict(sys.modules,{'teacher_contract':fake}):
            with self.assertRaisesRegex(ValueError,'conflicting module source'):teacher()
    def test_bounded_supervisor_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);here=base/'m';here.mkdir();end=int(time.time())+10000
            (base/'initial_plan.json').write_text(json.dumps({'budget_end_unix':end}));(here/'plan.json').write_text(json.dumps({
                'inner_timeout_seconds':1800,'outer_timeout_seconds':1920,'budget_source_sha256':'h','budget_end_unix':end,
                'progress_stale_seconds':900,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}))
            mon=here/'monitor_smoke001';mon.mkdir();start={'worker_pid':os.getpid(),'timeout_seconds':1920,'stale_seconds':900,
                'progress_paths':[str((here/'smoke_001/events.jsonl').resolve())],'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
            (mon/'started.json').write_text(json.dumps(start))
            with patch.object(launcher,'HERE',here),patch.object(launcher,'sha',lambda _:'h'):
                launcher.check_budget('001',1800);start['stale_seconds']=0;(mon/'started.json').write_text(json.dumps(start))
                with self.assertRaises(ValueError):launcher.check_budget('001',1800)

    def test_review_source_and_clean_exit_are_required(self):
        g=local('gate')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);here=root/'m3_gpu_smoke';here.mkdir();review=root/'e1_independent_review';review.mkdir()
            script=review/'review_m3gpu002_preflight.py';script.write_text('review')
            mon=review/'monitor_m3gpu002_preflight';mon.mkdir();proof=here/'independent_review_002_preflight.json'
            start={'cwd':str(root),'worker_pid':123,'command':['python',str(script),'--out',str(proof)]}
            end={'worker_pid':123,'worker_returncode':0,'reason':'completed','remaining_process_group':[]}
            (mon/'started.json').write_text(json.dumps(start));(mon/'terminal.json').write_text(json.dumps(end))
            b={'review_started':str(mon/'started.json'),'review_terminal':str(mon/'terminal.json'),'review_proof':str(proof)}
            r={'reviewer_path':str(script),'reviewer_sha256':g.sha(script)}
            with patch.object(g,'ROOT',root),patch.object(g,'HERE',here):
                g._review_binding(b,r,True)
                end['worker_pid']=124;(mon/'terminal.json').write_text(json.dumps(end))
                with self.assertRaises(ValueError):g._review_binding(b,r,True)
                end['worker_pid']=123;(mon/'terminal.json').write_text(json.dumps(end));script.write_text('changed')
                with self.assertRaises(ValueError):g._review_binding(b,r,True)
    def test_preflight_receipt_log_and_command_are_bound(self):
        g=local('gate')
        with tempfile.TemporaryDirectory() as tmp:
            here=Path(tmp);out=here/'preflight_001';out.mkdir()
            for name in ('worker.py','preflight.py'): (here/name).write_text(name)
            receipt=out/'host_upstream.json';receipt.write_text(json.dumps({'native':'ok'}))
            command=['docker','--name','eadp-m3-preflight-001','CUDA_VISIBLE_DEVICES=',str(here/'worker.py'),'--execute','--preflight-only','--out',str(out)]
            launch=out/'launch.json';launch.write_text(json.dumps({'command':command,'host_upstream':{'path':str(receipt),'sha256':g.sha(receipt)}}))
            (out/'stdout.log').write_text('ok');(out/'stderr.log').write_text('')
            result={'status':'preflight_passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions':False,'target_ids':[100,200,300]}
            paths={'preflight_sha256':here/'preflight.py','launch_sha256':launch,'receipt_sha256':receipt,'stdout_sha256':out/'stdout.log','stderr_sha256':out/'stderr.log','worker_sha256':here/'worker.py'}
            d={k:g.sha(v) for k,v in paths.items()};d.update(status='passed',returncode=0,container_running=False,container_exit_code=0,timed_out=False,container_id='abc',result=result)
            proof=here/'proof.json';proof.write_text(json.dumps(d))
            with patch.object(g,'HERE',here),patch.object(g,'upstream',return_value={'native':'ok'}):
                g._check_preflight_evidence(proof)
                (out/'stdout.log').write_text('tampered')
                with self.assertRaises(ValueError):g._check_preflight_evidence(proof)
                (out/'stdout.log').write_text('ok');command.append('/dev/nvidia0');launch.write_text(json.dumps({'command':command,'host_upstream':{'path':str(receipt),'sha256':g.sha(receipt)}}));d['launch_sha256']=g.sha(launch);proof.write_text(json.dumps(d))
                with self.assertRaises(ValueError):g._check_preflight_evidence(proof)

if __name__=='__main__':unittest.main()
