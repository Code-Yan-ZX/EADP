"""Actual no-GPU disposable sleep container stop/inspect probe."""
import json,os,subprocess,time
from pathlib import Path
from light_preflight_launch import cleanup_container
from real_worker import PLAN,sha,HERE

plan=json.loads(PLAN.read_text());name='eadp-light-cleanup-probe-'+str(os.getpid());out=HERE/'cpu_cleanup_probe001.json'
command=['docker','run','-d','--network','none','--cap-drop','ALL','--name',name,'--entrypoint','/bin/sh',plan['container_image'],'-c','sleep 120']
record={'command':command,'name':name,'gpu_requested':False,'cleanup_source_sha256':sha(HERE/'light_preflight_launch.py'),'started_unix':time.time(),'status':'failed'}
try:
 x=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True);record['container_id']=x.stdout.strip()
 observed={};last=cleanup_container(name,observed);record['observed']=observed
 if last is None or last['id']!=record['container_id'] or last['state']['Running'] is not False or last['state']['Pid']!=0 or last['host_devices'] or last['device_requests']:raise ValueError('container did not stop cleanly')
 record['status']='passed'
except Exception as exc:record['error']={'type':type(exc).__name__,'message':str(exc)}
finally:
 if record['status']!='passed' and record.get('container_id'):
  try:
   cleanup=subprocess.run(['docker','kill',name],capture_output=True,text=True,timeout=15,check=False)
   record['emergency_kill']={'returncode':cleanup.returncode,'stderr':cleanup.stderr[-1000:]}
  except Exception as exc:record['emergency_kill']={'error_type':type(exc).__name__,'message':str(exc)}
 record['finished_unix']=time.time()
 with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if record['status']!='passed':raise SystemExit(1)
