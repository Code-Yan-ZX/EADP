"""Bounded retained container imports of full worker; no GPU device."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PILOT=HERE.parent/'e4_c_latency';sys.path.insert(0,str(PILOT))
from real_worker import PLAN,sha
from timing_launch import run_bounded
from full_admission import sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('bad run ID')
 out=HERE/('import_preflight'+a.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-latency-full-import-'+a.run_id+'-'+str(os.getpid());cmd=container_command(plan,[str(HERE/'import_probe.py')],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only import command invalid')
 cmd.remove('--rm');source={str(p):sha(p) for p in sources()};launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'source_sha256':source,'gpu_requested':False,'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,180,None,identity,out,require_gpu_idle=False)
 if any(sha(p)!=d for p,d in source.items()):raise ValueError('source changed during import probe')
 return result
if __name__=='__main__':sys.exit(main())
