"""Supplemental manual health overlap, preserving all complete-run jobs."""
import argparse,hashlib,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent;FULL=BASE/'e4_c_latency_full';RUN=FULL/'full_001';MON=FULL/'monitor_verify_full001';PROOF=FULL/'full_001_verification.json';MANUAL=FULL/'manual_health_windows.jsonl'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):
 with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]
def validate_windows(items,namespace):
 if not items:raise ValueError('manual health windows missing')
 result=[]
 for i,x in enumerate(items):
  if x.get('time_namespace')!=namespace or 'nvml_window_monotonic_ns' not in x:raise ValueError('manual clock or NVML status unknown')
  def pair(v):
   if not isinstance(v,list) or len(v)!=2 or any(type(t) is not int or t<0 for t in v) or v[0]>v[1]:raise ValueError('invalid manual time window')
   return v
  overall=pair(x.get('manual_health_window_monotonic_ns'));nvml=x['nvml_window_monotonic_ns']
  if nvml is not None:
   nvml=pair(nvml)
   if not overall[0]<=nvml[0]<=nvml[1]<=overall[1]:raise ValueError('NVML outside manual window')
  elif 'no additional NVML query' not in x.get('gpu_check',''):raise ValueError('null NVML is not documented as no query')
  result.append({'index':i,'manual':overall,'nvml':nvml})
 return result

def annotate(events,windows,namespace):
 output=[]
 for e in events:
  if e.get('status')!='generated':continue
  job=e['job'];b=e['result']['boundaries'];a=b.get('t0_before_image_read_ns');z=b.get('generation_complete_ns')
  if (b.get('time_namespace')!=namespace or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or type(a) is not int or type(z) is not int or a<0 or a>z or job.get('phase') not in ('warmup','measure')):raise ValueError('worker time boundary/phase')
  overlaps=lambda w:w[0]<=z and a<=w[1]
  health=[x['index'] for x in windows if overlaps(x['manual'])];nvml=[x['index'] for x in windows if x['nvml'] is not None and overlaps(x['nvml'])]
  if any(i not in health for i in nvml):raise ValueError('NVML overlap cannot exceed manual overlap')
  output.append({'job':job,'manual_health_overlap':bool(health),'manual_health_indices':health,'manual_nvml_overlap':bool(nvml),'manual_nvml_indices':nvml})
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();out=a.out_dir.resolve()
 if out!=(HERE/'manual_overlap001').resolve():raise ValueError('unexpected output')
 started=read(MON/'started.json');terminal=read(MON/'terminal.json');proof=read(PROOF);journal=(RUN/'journal.jsonl').resolve()
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or started.get('worker_pid')!=terminal.get('worker_pid') or [(Path(started['cwd'])/v).resolve() for v in started['command'] if Path(v).name=='full_verify.py']!=[(FULL/'full_verify.py').resolve()] or (proof.get('status'),proof.get('n_jobs'),proof.get('n_generated'),proof.get('GT_opened'),proof.get('scoring_performed'),proof.get('errors'))!=('passed',5940,5940,False,False,[]):raise ValueError('formal acceptance not complete')
 if proof['verifier_sha256']!=sha(FULL/'full_verify.py') or proof['evidence_sha256'].get(str(journal))!=sha(journal):raise ValueError('accepted journal/verifier changed')
 hashes={str(p.resolve()):sha(p) for p in (PROOF,MON/'started.json',MON/'terminal.json',journal,MANUAL)};manual=lines(MANUAL);events=lines(journal)
 generated=[e for e in events if e.get('status')=='generated']
 if len(events)!=11880 or len(generated)!=5940:raise ValueError('complete journal required')
 namespace=generated[0]['result']['boundaries']['time_namespace'];windows=validate_windows(manual,namespace);output=annotate(events,windows,namespace)
 if len(output)!=5940 or [r['job']['order'] for r in output]!=list(range(5940)):raise ValueError('incomplete job order')
 counts={phase:{'n':sum(x['job']['phase']==phase for x in output),'manual_health':sum(x['job']['phase']==phase and x['manual_health_overlap'] for x in output),'manual_nvml':sum(x['job']['phase']==phase and x['manual_nvml_overlap'] for x in output)} for phase in ('warmup','measure')}
 if (counts['warmup']['n'],counts['measure']['n'])!=(180,5760):raise ValueError('phase denominator')
 if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('evidence changed during annotation')
 out.mkdir(exist_ok=False)
 with (out/'annotations.jsonl').open('x') as f:
  for row in output:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 result={'status':'passed','jobs':len(output),'window_count':len(windows),'counts':counts,'time_namespace':namespace,'input_sha256':hashes,'annotations_sha256':sha(out/'annotations.jsonl'),'annotator_sha256':sha(__file__),'all_samples_retained':True,'GT_opened':False,'scoring_performed':False,'limitations':['Manual recorded windows do not include short trailing log writes or residual effects before/after checks.','About 07:03:30 two short journal/status scans have no precise monotonic bounds and cannot be assigned to individual jobs.','Other brief project-side stat/ps, small source reads, notes, and remote queries occurred without per-event monotonic bounds; no claim of zero interference.','Hourly supervisor health windows are separately recorded in formal annotations.']}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','jobs':len(output),'counts':counts}))
if __name__=='__main__':main()
