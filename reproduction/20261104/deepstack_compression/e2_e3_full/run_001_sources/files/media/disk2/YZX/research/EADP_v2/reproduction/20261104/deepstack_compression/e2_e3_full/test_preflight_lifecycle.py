import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from preflight import supervise_container
class PreflightLifecycleTests(unittest.TestCase):
 def proc(self):
  p=MagicMock();p.poll.return_value=None;p.wait.return_value=0;return p
 def test_term_stops_real_container_before_return(self):
  stopped={'Id':'abc','State':{'Running':False,'Pid':0,'ExitCode':0}}
  def docker(cmd,**kw):
   if cmd[1]=='inspect':return subprocess.CompletedProcess(cmd,0,json.dumps([stopped]),'')
   return subprocess.CompletedProcess(cmd,0,'','')
  with tempfile.TemporaryDirectory() as d,patch('preflight.subprocess.Popen',return_value=self.proc()),patch('preflight.subprocess.run',side_effect=docker) as run:
   r=supervise_container(['docker','run'],'name',Path(d),[signal.SIGTERM],timeout=10)
   self.assertEqual(r['interrupted'],signal.SIGTERM);self.assertFalse(r['timed_out']);self.assertTrue(r['container_stopped_verified']);self.assertTrue(any(c.args[0][1]=='stop' for c in run.call_args_list));self.assertTrue((Path(d)/'stdout.log').exists())
 def test_timeout_failed_stop_attempts_kill_and_records_failure(self):
  running={'Id':'abc','State':{'Running':True,'Pid':22,'ExitCode':0}};stopped={'Id':'abc','State':{'Running':False,'Pid':0,'ExitCode':137}};states=iter((running,stopped))
  def docker(cmd,**kw):
   if cmd[1]=='inspect':return subprocess.CompletedProcess(cmd,0,json.dumps([next(states)]),'')
   return subprocess.CompletedProcess(cmd,1 if cmd[1]=='stop' else 0,'','')
  with tempfile.TemporaryDirectory() as d,patch('preflight.subprocess.Popen',return_value=self.proc()),patch('preflight.subprocess.run',side_effect=docker) as run:
   r=supervise_container(['docker','run'],'name',Path(d),[None],timeout=0)
   self.assertTrue(r['timed_out']);self.assertTrue(r['container_stopped_verified']);self.assertTrue(r['errors']);self.assertTrue(any(c.args[0][1]=='kill' for c in run.call_args_list))
 def test_real_external_sigterm_stops_child_container(self):
  fake_docker="""#!/usr/bin/env python3
import json,os,signal,sys,time
p=os.environ['E2E3_TEST_STATE'];action=sys.argv[1]
if action=='run':
 open(p,'w').write(json.dumps({'Id':'synthetic','State':{'Running':True,'Pid':os.getpid(),'ExitCode':0}}))
 while True:time.sleep(.1)
elif action=='inspect':
 try:print(json.dumps([json.load(open(p))]))
 except OSError:sys.exit(1)
elif action in ('stop','kill'):
 try:
  v=json.load(open(p));os.kill(v['State']['Pid'],signal.SIGTERM);v['State'].update(Running=False,Pid=0,ExitCode=143);open(p,'w').write(json.dumps(v))
 except (OSError,KeyError):sys.exit(1)
"""
  harness="""import json,pathlib,signal,sys
from preflight import supervise_container
out=pathlib.Path(sys.argv[1]);flag=[None]
def handle(sig,frame):flag[0]=sig
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,handle)
r=supervise_container(['docker','run'],'synthetic',out,flag,timeout=60)
(out/'result.json').write_text(json.dumps(r))
"""
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);docker=root/'docker';docker.write_text(fake_docker);docker.chmod(0o755);state=root/'state';env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],E2E3_TEST_STATE=str(state),PYTHONPATH=str(Path(__file__).resolve().parent))
   p=subprocess.Popen([sys.executable,'-c',harness,d],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   try:
    for _ in range(100):
     if state.exists():break
     if p.poll() is not None:self.fail('harness exited early: '+p.stderr.read())
     time.sleep(.05)
    self.assertTrue(state.exists(),'fake container never started');p.send_signal(signal.SIGTERM);stdout,stderr=p.communicate(timeout=15);self.assertEqual(p.returncode,0,stderr)
    r=json.loads((root/'result.json').read_text());self.assertEqual(r['interrupted'],signal.SIGTERM);self.assertTrue(r['container_stopped_verified']);self.assertFalse(json.loads(state.read_text())['State']['Running'])
   finally:
    if p.poll() is None:p.kill();p.wait()
 def test_unknown_inspect_and_cleanup_cannot_report_stopped(self):
  def docker(cmd,**kw):raise subprocess.TimeoutExpired(cmd,1)
  with tempfile.TemporaryDirectory() as d,patch('preflight.subprocess.Popen',return_value=self.proc()),patch('preflight.subprocess.run',side_effect=docker):
   r=supervise_container(['docker','run'],'name',Path(d),[None],timeout=0);self.assertTrue(r['timed_out']);self.assertFalse(r['container_stopped_verified']);self.assertTrue(r['errors'])
if __name__=='__main__':unittest.main()
