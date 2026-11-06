"""CPU container preflight retaining identity, command and stopped container."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PLAN=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path.insert(0,str(PLAN.parent))
from run_panel import container_command,identity
from smoke_launch import configure_container
plan=json.loads(PLAN.read_text());cache=HERE/'container_preflight003_cache';cache.mkdir(exist_ok=False)
name='eadp-e6prior-cpu-preflight003-'+str(os.getpid())
cmd=container_command(plan,[str(HERE/'container_preflight003.py')],False,name)
idx=cmd.index('--entrypoint');cmd[idx+1]=str(HERE.parent/'install001/env/bin/python')
cmd=configure_container(cmd,cache);at=cmd.index('--workdir');cmd[at:at]=['--env','CUDA_VISIBLE_DEVICES=']
if '--gpus' in cmd or cmd.count('--rm')!=1:raise ValueError('CPU container command policy')
cmd.remove('--rm')
def save(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def inspect():
 p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10,check=True)
 x=json.loads(p.stdout)[0]
 return {'id':x['Id'],'image_id':x['Image'],'path':x['Path'],'args':x['Args'],'state':{k:x['State'][k] for k in ('Status','Running','Pid','ExitCode','StartedAt','FinishedAt')}}
def alive(ident):
 pid=ident['pid'];p=Path('/proc')/str(pid)/'stat'
 try:
  raw=p.read_text();return int(raw[raw.rfind(')')+2:].split()[19])==ident['start_ticks']
 except (OSError,ValueError,IndexError):return False
save('container_preflight003_launch.json',{'command':cmd,'scope':'CPU only; retained container','controller':identity(os.getpid())})
received=[None]
def interrupted(sig,_frame):received[0]=sig
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
container=None;begin=time.monotonic();rc=None;failure=None;after=None
with (HERE/'container_preflight003.stdout').open('x') as out,(HERE/'container_preflight003.stderr').open('x') as err:
 client=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
 try:
  while client.poll() is None:
   if container is None:
    try:
     current=inspect()
     if current['state']['Pid']:
      host=identity(current['state']['Pid'])
      if host:container={'initial':current,'host_init':host};save('container_preflight003_identity.json',container)
    except subprocess.CalledProcessError:pass
   if received[0] is not None or time.monotonic()-begin>600:
    subprocess.run(['docker','stop','--time','10',name],stdout=err,stderr=err,timeout=30,check=False)
    try:client.wait(timeout=30)
    except subprocess.TimeoutExpired:client.kill();client.wait()
    raise RuntimeError('CPU container interrupted or timed out')
   time.sleep(1)
  rc=client.wait();after=inspect()
  if container is None or after['id']!=container['initial']['id'] or after['state']['Running'] or after['state']['Pid']!=0 or after['state']['ExitCode']!=0 or rc!=0 or alive(container['host_init']):raise ValueError('CPU container exit evidence invalid')
  if after['image_id']!=plan['container_image'] or after['path']!=str(HERE.parent/'install001/env/bin/python') or after['args']!=[str(HERE/'container_preflight003.py')]:raise ValueError('actual image or argv changed')
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)}
  subprocess.run(['docker','stop','--time','10',name],stdout=err,stderr=err,timeout=30,check=False)
  if client.poll() is None:client.kill();client.wait()
  try:after=inspect()
  except Exception:pass
 finally:
  if received[0] is not None and failure is None:failure={'type':'Interrupted','message':str(received[0])}
  save('container_preflight003_exit.json',{'status':'passed' if failure is None else 'failed','returncode':rc,'identity':container,'final_inspect':after,'failure':failure,'finished_unix':time.time()})
if failure is not None:raise SystemExit(1)
result=json.loads((HERE/'container_preflight003.stdout').read_text().strip().splitlines()[-1])
if result.get('status')!='passed' or result.get('cuda_initialized') is not False:raise ValueError('CPU runtime proof failed')
print(json.dumps({'status':'passed','container_id':after['id'],'scope':'retained CPU-only container, no GPU'}))
