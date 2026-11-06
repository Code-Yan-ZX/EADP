import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import prepare_panel003_acceptance as prepare

def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class PrepareTests(unittest.TestCase):
    def test_requires_clean_exit_and_snapshots_exact_approved_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';base.mkdir();panel=base/'e4_panel_v2';panel.mkdir();run=panel/'panel_003';run.mkdir()
            sup=panel/'monitor_panel003';sup.mkdir();src=root/'source.py';src.write_text('approved source');verifier=panel/'panel_verify.py';verifier.write_text('verified')
            sources={str(src):sha(src),str(verifier):sha(verifier)};put(panel/'cpu_check_005/verification.json',{'status':'passed','source_sha256_before':sources,'source_sha256_after':sources})
            put(panel/'independent_review_005.json',{'status':'passed'});put(run/'frozen.json',{'code_sha256':sources});put(run/'worker_complete.json',{})
            names=('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal')
            command=['python']+[x for n in names for x in ('--'+n,str(root/n))]
            put(sup/'started.json',{'worker_pid':7,'command':command});put(sup/'terminal.json',{'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':7})
            put(run/'launch.json',{});exitdata={'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,
                'interrupted':None,'gpu_pids_after':[],'container_identity':{'container_id':'id'}};put(run/'exit.json',exitdata)
            docker=SimpleNamespace(stdout=json.dumps([{'Id':'id','State':{'Running':False,'ExitCode':0}}]))
            with patch.object(prepare,'BASE',base),patch.object(prepare,'PROJECT',root),patch.object(prepare,'PANEL',panel), \
                 patch.object(prepare,'RUN',run),patch.object(prepare,'SUP',sup),patch.object(prepare.subprocess,'run',return_value=docker), \
                 patch.object(prepare.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen:
                exitdata['container_stopped_verified']=False;put(run/'exit.json',exitdata)
                with self.assertRaises(ValueError):prepare.main()
                self.assertFalse((panel/'panel_003_sources').exists());popen.assert_not_called()
                exitdata['container_stopped_verified']=True;put(run/'exit.json',exitdata);prepare.main()
                manifest=json.loads((panel/'panel_003_sources/manifest.json').read_text())
                self.assertEqual(manifest['sources'][str(src)]['sha256'],sha(src));popen.assert_called_once()
                cmd=popen.call_args.args[0];self.assertIn(str(panel/'panel_verify.py'),cmd)
                self.assertIn(str(panel/'panel_003_verification.json'),cmd)

if __name__=='__main__':unittest.main()
