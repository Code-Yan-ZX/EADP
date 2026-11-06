"""Real controller primitives against fake Docker/NVML and disposable child PIDs."""
import json,os,subprocess,sys,tempfile,textwrap,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[2];SUP=B/'monitor/supervise.py';E4=B/'e4_c_latency'
class Tests(unittest.TestCase):
 def test_outer_stale_term_drives_stop_kill_and_reinspect(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);tools=d/'bin';tools.mkdir();state=d/'state';state.write_text('running');calls=d/'calls';art=d/'artifacts';art.mkdir();run=d/'monitor';progress=d/'no_progress'
   docker=tools/'docker';docker.write_text('#!'+sys.executable+'\n'+textwrap.dedent('''
import json,os,sys,time
from pathlib import Path
state=Path(os.environ['FAKE_CONTAINER_STATE']);log=Path(os.environ['FAKE_CONTAINER_CALLS']);action=sys.argv[1]
with log.open('a') as f:f.write(action+'\\n')
if action=='run':
 while state.read_text()=='running':time.sleep(.05)
 sys.exit(137)
if action=='stop':sys.exit(1)
if action=='kill':state.write_text('stopped');sys.exit(0)
if action=='inspect':
 running=state.read_text()=='running';data={'Id':'fake-container','Image':'fake-image','State':{'Running':running,'Pid':321 if running else 0,'ExitCode':137},'HostConfig':{'Devices':[],'DeviceRequests':None,'PidMode':''},'Config':{'Entrypoint':['python'],'Cmd':['fake-worker']}}
 print(json.dumps([data]));sys.exit(0)
sys.exit(9)
'''));docker.chmod(0o755)
   nvml=tools/'nvidia-smi';nvml.write_text('#!'+sys.executable+'\nprint("fake NVML, no device access")\n');nvml.chmod(0o755)
   worker=d/'worker.py';worker.write_text(textwrap.dedent('''
import os,sys
from pathlib import Path
sys.path.insert(0,os.environ['FAKE_E4_DIR'])
import timing_launch
timing_launch.process_attestation=lambda pid,cid:{'host_argv':['fake-worker'],'host_cgroup':cid}
identity=lambda pid:{'pid':pid,'start_ticks':1,'boot_id':'fake-boot'}
Path(os.environ['FAKE_PROGRESS']).write_text('worker ready')
code=timing_launch.run_bounded(['docker','run'],'fake-container',60,lambda:[],identity,Path(os.environ['FAKE_ARTIFACTS']),require_gpu_idle=True)
raise SystemExit(code)
'''))
   env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ.get('PATH',''),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',FAKE_CONTAINER_STATE=str(state),FAKE_CONTAINER_CALLS=str(calls),FAKE_ARTIFACTS=str(art),FAKE_E4_DIR=str(E4),FAKE_PROGRESS=str(progress))
   command=[sys.executable,str(SUP),'--run-dir',str(run),'--timeout','20','--stale','5','--progress',str(progress),'--snapshot-seconds','60','--poll-seconds','.1','--grace-seconds','6','--',sys.executable,str(worker)]
   proc=subprocess.run(command,cwd=B,env=env,capture_output=True,text=True,timeout=25)
   terminal=json.loads((run/'terminal.json').read_text());started=json.loads((run/'started.json').read_text());exit_=json.loads((art/'exit.json').read_text());actions=calls.read_text().splitlines()
   self.assertNotEqual(proc.returncode,0);self.assertEqual(terminal['reason'],'progress_stale');self.assertEqual(started['worker_pid'],terminal['worker_pid']);self.assertEqual(started['process_group'],started['worker_pid']);self.assertEqual(terminal['remaining_process_group'],[])
   self.assertIn('stop',actions);self.assertIn('kill',actions);self.assertLess(actions.index('stop'),actions.index('kill'));self.assertIn('inspect',actions[actions.index('kill')+1:]);self.assertEqual(state.read_text(),'stopped')
   self.assertTrue(exit_['container_stopped_verified']);self.assertFalse(exit_['last_inspect']['running']);self.assertEqual(exit_['last_inspect']['pid'],0);self.assertEqual(exit_['gpu_pids_after'],[]);self.assertEqual(exit_['interrupted'],15)
if __name__=='__main__':unittest.main()
