"""CPU-only subprocess isolation rehearsal; the real GPU controller is locked.

No Docker or model launch is implemented here. Each fake job is a new process,
serially paired according to the frozen plan. GPU admission is always rejected.
"""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from pilot_contract import validate_schedule

def identity():
 p=Path('/proc/self/stat').read_text();start=int(p[p.rfind(')')+2:].split()[19])
 return {'pid':os.getpid(),'ppid':os.getppid(),'start_ticks':start,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def run_fake(jobs,anchors,emit,timeout=30,child=None):
 validate_schedule(jobs,anchors)
 if type(timeout) is not int or not 1<=timeout<=60:raise ValueError('fake timeout')
 seen=set()
 for job in jobs:
  argv=child or [sys.executable,str(HERE/'controller_candidate.py'),'--fake-child']
  emit({'event':'started','job':job})
  try:
   proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
   try:out,err=proc.communicate(timeout=timeout)
   except BaseException:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.communicate();raise
   if proc.returncode!=0:raise ValueError('fake process failure: '+err[-1000:])
   x=json.loads(out);key=(x['pid'],x['start_ticks'],x['boot_id'])
   if x['pid']!=proc.pid or x['ppid']!=os.getpid() or key in seen:raise ValueError('fake process identity mismatch')
   seen.add(key);emit({'event':'completed','job':job,'child_identity':x})
  except BaseException as exc:
   emit({'event':'failed','job':job,'error_type':type(exc).__name__,'error':str(exc)})
   raise
 return len(seen)

def main():
 p=argparse.ArgumentParser();p.add_argument('--fake-child',action='store_true');p.add_argument('--real-gpu',action='store_true');a=p.parse_args()
 if a.real_gpu:raise ValueError('real controller locked: Docker identity, device exclusivity, endpoint and prelaunch review not implemented')
 if not a.fake_child:raise ValueError('controller supports only isolated CPU fake-child test')
 print(json.dumps(identity()))
if __name__=='__main__':main()
