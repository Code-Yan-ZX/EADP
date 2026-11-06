"""One-shot R-res postprocessing. Never launch during an incomplete GPU run."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon,worker,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if cwd.resolve()!=B or s['worker_pid']!=t['worker_pid'] or type(s['worker_pid']) is not int or s['worker_pid']<=0 or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or len(cmd)<2 or (cwd/cmd[1]).resolve()!=worker.resolve():raise ValueError('unclean supervisor '+str(mon))
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('output argument')
 return s
def generation():
 run=B/'e6/rres_k256_dev384'; e=B/'e6/rres_k256_dev384_exec'; s=clean(run/'monitor_run001',e/'launch.py');cmd=s['command']
 if cmd[-2:]!=['--run-id','001'] or s.get('timeout_seconds')!=21600 or s.get('grace_seconds')!=300 or s.get('process_group')!=s['worker_pid']:raise ValueError('generation supervision identity')
 l=read(run/'run001/launch.json'); x=read(run/'run001/exit.json');c=x['controller'];ident=x['container_identity']
 if c!=l['controller'] or c['pid']!=s['worker_pid'] or type(c.get('start_ticks')) is not int or c['start_ticks']<=0 or not c.get('boot_id') or l.get('gpu_pids_before')!=[] or x.get('gpu_pids_after')!=[] or x.get('returncode')!=0 or x.get('container_exit_code')!=0 or x.get('timed_out') is not False or x.get('interrupted') is not None or x.get('failure_type') is not None or x.get('container_stopped_verified') is not True:raise ValueError('generation controller/exit')
 cid=ident['container_id'];host=ident['host_init']
 if len(cid)!=64 or cid not in ident['host_cgroup'] or type(host.get('pid')) is not int or host['pid']<=0 or type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id'):raise ValueError('container parent identity')
 obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]; st=obj['State']
 if obj['Id']!=cid or obj['Image']!=ident['image'] or st['Running'] is not False or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] is not False:raise ValueError('actual stopped container')
 return {'run_started_sha256':sha(run/'monitor_run001/started.json'),'run_terminal_sha256':sha(run/'monitor_run001/terminal.json'),'launch_sha256':sha(run/'run001/launch.json'),'exit_sha256':sha(run/'run001/exit.json'),'container_id':cid}
def assert_product(step):
 q=read(Path(step['product']))
 if q.get('status')!=step['status']:raise ValueError('step status '+step['name'])
 for k,v in step.get('required',{}).items():
  if q.get(k)!=v:raise ValueError('step field '+step['name']+':'+k)
 return sha(step['product'])
def active_parent(mon):
 """Attest the current running generation child and its real supervisor."""
 s=read(mon/'started.json');pid=s['worker_pid'];p=Path('/proc')/str(pid)
 raw=(p/'stat').read_text();rest=raw[raw.rfind(')')+2:].split();parent=int(rest[1]);cmd=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 supervisor=Path('/proc')/str(parent);pcmd=[x.decode() for x in (supervisor/'cmdline').read_bytes().split(b'\0') if x];pr=(supervisor/'stat').read_text();parts=pr[pr.rfind(')')+2:].split()
 if cmd!=s['command'] or s['process_group']!=pid or parent<=0 or not any(Path(x).name=='supervise.py' for x in pcmd) or pcmd.count('--run-dir')!=1 or Path(pcmd[pcmd.index('--run-dir')+1]).resolve()!=mon.resolve():raise ValueError('live supervisor identity')
 return {'worker_pid':pid,'worker_start_ticks':int(rest[19]),'supervisor_pid':parent,'supervisor_start_ticks':int(parts[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan-sha256',required=True);a=ap.parse_args();plan_path=D/'plan001.json'
 if len(a.plan_sha256)!=64 or sha(plan_path)!=a.plan_sha256:raise ValueError('unapproved plan')
 plan=read(plan_path)
 if plan['coordinator_sha256']!=sha(__file__) or any(sha(k)!=v for k,v in plan['sources'].items()):raise ValueError('source drift')
 if Path.cwd().resolve()!=B:raise ValueError('working directory')
 # The exclusive lock coordinates only instances of this program. The operator
 # must reserve the canonical step paths against non-cooperating launchers.
 with (D/'coordinator.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (D/'audit001.jsonl').open('x',buffering=1) as log:
   def event(kind,**extra):
    log.write(json.dumps({'event':kind,'utc_unix':time.time(),'monotonic_ns':time.monotonic_ns(),**extra},sort_keys=True)+'\n');log.flush();os.fsync(log.fileno())
   try:
    event('started',pid=os.getpid(),plan_sha256=a.plan_sha256)
    run=B/'e6/rres_k256_dev384/monitor_run001';limit=plan['budget_end_unix']
    live=None
    while not (run/'terminal.json').exists():
     if time.time()>=limit:raise TimeoutError('generation did not exit within project budget')
     try:
      current=active_parent(run)
      if live is None:live=current;event('live_generation_supervisor',**live)
      elif current!=live:raise ValueError('generation process identity changed')
     except (FileNotFoundError,ProcessLookupError):
      # The child can exit shortly before its supervisor writes terminal.json.
      time.sleep(15)
      if not (run/'terminal.json').exists():raise RuntimeError('generation supervisor or child vanished without terminal record')
      break
     time.sleep(15)
    event('generation_accepted',**generation())
    for step in plan['steps']:
     if time.time()+step['timeout']+step['grace']>=limit:raise TimeoutError('insufficient time for '+step['name'])
     if any(sha(k)!=v for k,v in plan['sources'].items()):raise ValueError('source drift before '+step['name'])
     mon=Path(step['monitor']);product=Path(step['product']);reserved=Path(step['reserved'])
     if mon.exists() or product.exists() or reserved.exists():raise FileExistsError('step previously started or output already exists: '+step['name'])
     command=step['command']; event('launch',step=step['name'],command=command)
     child=subprocess.Popen(command,cwd=B,start_new_session=True)
     event('supervisor_started',step=step['name'],supervisor_pid=child.pid)
     try:ret=child.wait(timeout=step['timeout']+step['grace']+120)
     except subprocess.TimeoutExpired:
      event('supervisor_watchdog_timeout',step=step['name'],supervisor_pid=child.pid)
      try:os.killpg(child.pid,signal.SIGTERM)
      except ProcessLookupError:pass
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:event('supervisor_still_running',step=step['name'],supervisor_pid=child.pid)
      raise TimeoutError('supervisor exceeded bound; inspect cleanup: '+step['name'])
     if ret!=0:raise RuntimeError('supervisor failed '+step['name']+' exit '+str(ret))
     clean(mon,Path(step['worker']),Path(step['out_arg']) if step['out_arg'] else None)
     event('step_completed',step=step['name'],product_sha256=assert_product(step),started_sha256=sha(mon/'started.json'),terminal_sha256=sha(mon/'terminal.json'))
    event('completed')
   except BaseException as e:
    event('stopped',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
