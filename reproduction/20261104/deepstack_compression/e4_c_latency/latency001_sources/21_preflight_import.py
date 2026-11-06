"""Bounded retained-container import probe; never expose or query GPU."""
import argparse,json,os,sys,time
from pathlib import Path
from real_worker import PLAN,HERE,sha
from latency_worker import sources
from timing_launch import run_bounded

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid preflight identity')
 out=HERE/('import_preflight'+args.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-latency-import-'+args.run_id+'-'+str(os.getpid());cmd=container_command(plan,[str(HERE/'import_probe.py')],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only command invalid')
 cmd.remove('--rm');source=sources()+[HERE/'import_probe.py',HERE/'preflight_import.py'];hashes={str(p):sha(p) for p in source}
 record={'command':cmd,'controller':identity(os.getpid()),'source_sha256':hashes,'gpu_requested':False,'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,180,None,identity,out,require_gpu_idle=False)
 if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('source changed during CPU probe')
 return result
if __name__=='__main__':sys.exit(main())
