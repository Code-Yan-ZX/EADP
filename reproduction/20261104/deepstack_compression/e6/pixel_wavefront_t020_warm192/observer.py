"""Bounded low-frequency host observation, no GPU query or model access."""
import argparse,json,os,subprocess,time
from pathlib import Path
PS=['ps','-eo','user,pid,pcpu,pmem,stat,etime,comm','--sort=-pcpu']
DOCKER=['docker','ps','--no-trunc','--format','{{.ID}}\t{{.Image}}\t{{.Status}}']
def query(argv,*,run=subprocess.run,clock=time.monotonic_ns):
 start=clock()
 try:
  p=run(argv,capture_output=True,text=True,timeout=10,check=False);data={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except Exception as exc:data={'returncode':None,'error_type':type(exc).__name__}
 end=clock()
 if end<start:raise ValueError('nonmonotonic observer clock')
 return {'query_start_ns':start,'query_end_ns':end,**data}
def write_sync(stream,value):
 stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop',type=Path,required=True);p.add_argument('--interval',type=int,required=True);p.add_argument('--duration',type=int,required=True);a=p.parse_args();d=Path(__file__).resolve().parent
 if a.output.resolve()!=(d/'observations001.jsonl').resolve() or a.stop.resolve()!=(d/'stop001').resolve() or a.stop.exists() or a.interval!=60 or a.duration!=23000:raise ValueError('unplanned observer arguments')
 os.nice(10);namespace=os.readlink('/proc/self/ns/time');begin=time.monotonic();index=0
 with a.output.open('x',encoding='utf8') as f:
  while time.monotonic()-begin<a.duration and not a.stop.exists():
   ps=query(PS);docker=query(DOCKER);write_sync(f,{'event':'snapshot','index':index,'time_namespace':namespace,'ps':ps,'docker':docker});end=time.monotonic_ns()
   write_sync(f,{'event':'logging_complete','index':index,'time_namespace':namespace,'first_fsync_end_ns':end})
   index+=1;target=time.monotonic()+a.interval
   while time.monotonic()<target and not a.stop.exists():time.sleep(min(1,max(0,target-time.monotonic())))
  write_sync(f,{'event':'observer_end','time_namespace':namespace,'end_ns':time.monotonic_ns(),'samples':index,'reason':'stop' if a.stop.exists() else 'duration_expired'})
 return 0
if __name__=='__main__':raise SystemExit(main())
