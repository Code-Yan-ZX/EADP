"""Host-side per-process NVML samples via nvidia-smi; memory-only runs.

This samples MiB-rounded values and can miss short-lived peaks. A valid process
identity (host PID + start ticks) and a separate process/GPU exclusion audit are
required; a sampled maximum is never called an exact peak.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def parse_sample(output,uuid,pid):
 matched=[];others=[]
 for line in output.splitlines():
  if not line.strip():continue
  parts=[x.strip() for x in line.split(',')]
  if len(parts)!=3:raise ValueError('malformed nvidia-smi record')
  gpu,raw_pid,raw_mem=parts
  if gpu!=uuid:continue
  if not raw_pid.isdecimal() or not raw_mem.isdecimal():raise ValueError('missing process memory')
  item={'pid':int(raw_pid),'used_bytes':int(raw_mem)*1048576}
  (matched if item['pid']==pid else others).append(item)
 if len(matched)>1:raise ValueError('duplicate process')
 return (matched[0]['used_bytes'] if matched else None),others

def identity_alive(pid,start_ticks):
 try:
  text=Path('/proc') .joinpath(str(pid),'stat').read_text();tail=text[text.rfind(')')+2:].split()
  return int(tail[19])==start_ticks and tail[0]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--pid',type=int,required=True);p.add_argument('--start-ticks',type=int,required=True);p.add_argument('--gpu-uuid',required=True)
 p.add_argument('--interval',type=float,default=0.5);p.add_argument('--timeout',type=float,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.pid<=0 or a.start_ticks<=0 or not a.gpu_uuid.startswith('GPU-') or not 0.1<=a.interval<=10 or not 0<a.timeout<=86400:raise ValueError('invalid sampler arguments')
 if not identity_alive(a.pid,a.start_ticks):raise ValueError('target process identity absent')
 deadline=time.monotonic()+a.timeout;seen=False;peak=0;count=0;other_seen=False
 with a.out.open('x') as f:
  while identity_alive(a.pid,a.start_ticks):
   if time.monotonic()>deadline:raise TimeoutError('sampler deadline expired')
   run=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
   used,others=parse_sample(run.stdout,a.gpu_uuid,a.pid);seen|=used is not None;other_seen|=bool(others);count+=1
   if used is not None:peak=max(peak,used)
   f.write(json.dumps({'type':'sample','monotonic_ns':time.monotonic_ns(),'used_bytes':used,'other_gpu_processes':others})+'\n');f.flush();os.fsync(f.fileno())
   time.sleep(a.interval)
  result={'type':'summary','pid':a.pid,'start_ticks':a.start_ticks,'gpu_uuid':a.gpu_uuid,'samples':count,'target_observed':seen,'other_gpu_process_observed':other_seen,'nvml_process_sampled_peak_bytes':peak if seen else None,'limitation':'MiB-rounded samples can miss transient peaks'}
  f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
 if not seen or other_seen:raise ValueError('process memory missing or concurrent GPU process observed')
if __name__=='__main__':main()
