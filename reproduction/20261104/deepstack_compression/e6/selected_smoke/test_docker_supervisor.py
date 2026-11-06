import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import docker_supervisor as launcher

class BadClient:
    def poll(self): raise OSError('client vanished')
    def kill(self): raise OSError('kill client failed')
    def wait(self,timeout=None): raise OSError('wait failed')

class LauncherTests(unittest.TestCase):
    def test_unknown_container_and_failed_cleanup_still_writes_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher.subprocess,'Popen',return_value=BadClient()), \
             patch.object(launcher.subprocess,'run',side_effect=OSError('docker unavailable')):
            r=launcher.supervise(['docker','run'],'own-container',Path(tmp),{'pid':1},1,
                                 lambda: (_ for _ in ()).throw(OSError('gpu query failed')),lambda p:p,[None])
            saved=json.loads((Path(tmp)/'exit.json').read_text())
            self.assertEqual(saved,r);self.assertFalse(r['container_stopped_verified'])
            self.assertIsNone(r['gpu_pids_after']);self.assertTrue(any(x.startswith('kill:') for x in r['errors']))
    def test_stop_exception_attempts_kill_and_keeps_failure(self):
        calls=[]
        def run(args,**kwargs):
            calls.append(args[1])
            if args[1]=='stop':raise subprocess.TimeoutExpired(args,30)
            if args[1]=='kill':return SimpleNamespace(returncode=0)
            state={'Running':('kill' not in calls),'Pid':0,'ExitCode':137}
            return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':'abc','State':state}]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher.subprocess,'Popen',return_value=BadClient()), \
             patch.object(launcher.subprocess,'run',side_effect=run):
            r=launcher.supervise(['docker','run'],'own-container',Path(tmp),{},1,lambda:[],lambda p:p,[None])
            self.assertIn('kill',calls);self.assertTrue(any(x.startswith('stop:') for x in r['errors']))
            self.assertFalse(r['container_stopped_verified']);self.assertTrue((Path(tmp)/'exit.json').is_file())
