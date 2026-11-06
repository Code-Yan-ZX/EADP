"""Verify health-window annotations without computing or reporting durations."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]; A=R/'e4_c_latency_analysis'; F=R/'e4_c_latency_full'; I=R/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]
def checkmon(d):
 s=load(d/'started.json');t=load(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 return s
formal=F/'full_001_verification.json'; ann=A/'manual_overlap001/annotations.jsonl'; vp=A/'manual_overlap001/verification.json'; journal=F/'full_001/journal.jsonl'; manual=F/'manual_health_windows.jsonl'
v=load(vp); f=load(formal); s=checkmon(A/'monitor_manual001');checkmon(F/'monitor_verify_full001');checkmon(I/'monitor_e4c_latency_full001')
assert (v['status'],v['jobs'],v['all_samples_retained'],v['GT_opened'],v['scoring_performed'])==('passed',5940,True,False,False)
assert f['status']=='passed' and v['annotator_sha256']==sha(A/'manual_overlap.py') and v['annotations_sha256']==sha(ann)
assert all(sha(path)==h for path,h in v['input_sha256'].items())
assert s['command'][-3:]==['e4_c_latency_analysis/manual_overlap.py','--out-dir','e4_c_latency_analysis/manual_overlap001'] or [Path(x).name for x in s['command'][-3:]]==['manual_overlap.py','--out-dir','manual_overlap001']
windows=lines(manual); events=lines(journal); annotations=lines(ann); assert len(windows)==v['window_count'] and len(events)==11880 and len(annotations)==5940
for w in windows:
 a,b=w['manual_health_window_monotonic_ns']; assert isinstance(a,int) and isinstance(b,int) and 0<=a<=b and w['time_namespace']==v['time_namespace']
 if w['nvml_window_monotonic_ns'] is not None:
  c,d=w['nvml_window_monotonic_ns'];assert a<=c<=d<=b
 else:assert 'no additional NVML query' in w.get('gpu_check','')
counts={k:{'n':0,'manual_health':0,'manual_nvml':0} for k in ('warmup','measure')}; generated=[e for e in events if e.get('status')=='generated'];assert len(generated)==5940
for i,(event,got) in enumerate(zip(generated,annotations)):
 j=event['job'];b=event['result']['boundaries'];start=b['t0_before_image_read_ns'];end=b['generation_complete_ns'];assert j==got['job'] and j['order']==i and b['time_namespace']==v['time_namespace'] and b['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)' and 0<=start<=end
 hs=[];ns=[]
 for n,w in enumerate(windows):
  x,y=w['manual_health_window_monotonic_ns']
  if start<=y and x<=end:hs.append(n)
  if w['nvml_window_monotonic_ns'] is not None:
   x,y=w['nvml_window_monotonic_ns']
   if start<=y and x<=end:ns.append(n)
 assert got['manual_health_indices']==hs and got['manual_nvml_indices']==ns and got['manual_health_overlap']==bool(hs) and got['manual_nvml_overlap']==bool(ns) and set(ns)<=set(hs)
 counts[j['phase']]['n']+=1;counts[j['phase']]['manual_health']+=bool(hs);counts[j['phase']]['manual_nvml']+=bool(ns)
assert counts==v['counts']
out=I/'e4c_manual_overlap001_review.json';proof={'status':'passed','independent_review':True,'jobs_verified':len(annotations),'counts':counts,'proof_sha256':sha(vp),'annotations_sha256':sha(ann),'manual_windows_sha256':sha(manual),'journal_sha256':sha(journal),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'latency_values_reported':False,'limitations':v['limitations'],'errors':[]}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
