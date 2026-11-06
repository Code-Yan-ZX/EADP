"""Independent three-anchor timing and observer window arithmetic."""
import argparse,hashlib,json,math,os,statistics
from collections import Counter
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5';AN=B/'e6/pixel_wavefront_t020_timing_analysis_v5';C=B/'e6/pixel_wavefront_t020_timing_coverage_v3';O=B/'e6/pixel_wavefront_t020_timing_observer_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def hit(ws,a,b):return [i for i,(x,y) in enumerate(ws) if x<=b and a<=y]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();analysis=AN/'analysis001.json';cov=C/'coverage001.json';formal=A/'verification001.json';proof=I/'pixel_wavefront_t020_timing_result_review.json';terminal(AN/'monitor_analysis001',AN/'analysis_entry.py');terminal(C/'monitor_coverage001',C/'coverage_entry.py',cov);terminal(A/'monitor_verify001',A/'verify_entry.py',formal);terminal(I/'monitor_pixel_wavefront_t020_timing_result',I/'review_pixel_wavefront_t020_timing_result.py',proof);terminal(O/'monitor_watch001',O/'observe.py')
 x=read(analysis);v=read(cov);f=read(formal);q=read(proof);plan=read(R/'gpu_plan001.json');jobs=plan['jobs'];assert x['status']==v['status']==f['status']==q['status']=='passed' and x['GT_opened'] is False and x['scoring_performed'] is False and v['GT_opened'] is False and q['independent_review'] is True and q['formal_sha256']==sha(formal) and q['reviewer_sha256']==sha(I/'review_pixel_wavefront_t020_timing_result.py') and q['samples_excluded']==0 and v['formal_sha256']==sha(formal) and v['independent_review_sha256']==sha(proof) and v['observer_sha256']==sha(O/'observations001.jsonl')
 assert len(jobs)==len(f['rows'])==len(f['background_annotations'])==len(x['all_jobs'])==len(v['annotations'])==24 and x['warmups']==6 and x['measured']==18
 ev=lines(O/'observations001.jsonl');assert len(ev)>=3 and len(ev)%2==1 and ev[-1]['event']=='observer_end' and ev[-1]['reason']=='stop';ns=ev[-1]['time_namespace'];ps=[];docker=[];first=[]
 for i in range((len(ev)-1)//2):
  s,l=ev[2*i:2*i+2];assert s['event']=='snapshot' and l['event']=='logging_complete' and s['index']==l['index']==i and s['time_namespace']==l['time_namespace']==ns;z,w=s['ps'],s['docker'];assert z['query_start_ns']<=z['query_end_ns']<=w['query_start_ns']<=w['query_end_ns']<=l['first_fsync_end_ns'];ps.append((z['query_start_ns'],z['query_end_ns']));docker.append((w['query_start_ns'],w['query_end_ns']));first.append((z['query_start_ns'],l['first_fsync_end_ns']))
 assert v['jobs']==24 and v['samples']==len(ps)
 cells={};bounds=[];phase=Counter()
 for i,j in enumerate(jobs):
  record=read(R/'run_001'/f'job_{i:03d}'/'record.json')['record'];start,firsttoken,end=(record[k] for k in ('start_monotonic_ns','first_token_monotonic_ns','completion_monotonic_ns'));assert j['order']==i and record['boundaries']['time_namespace']==ns and 0<start<firsttoken<=end and f['rows'][i]=={'order':i,'phase':j['phase'],'ttft_ns':firsttoken-start};bounds.append((start,firsttoken,end));phase[j['phase']]+=1
  assert x['all_jobs'][i]=={'job':j,'ttft_ns':firsttoken-start,'background':f['background_annotations'][i]['background']}
  expected={'order':i,'windows':{'ps':{'status':'observed' if all(z['ps']['returncode']==0 for z in ev if z.get('event')=='snapshot') else 'unknown','overlap_indices':hit(ps,start,firsttoken)},'docker':{'status':'observed' if all(z['docker']['returncode']==0 for z in ev if z.get('event')=='snapshot') else 'unknown','overlap_indices':hit(docker,start,firsttoken)},'first_fsync_window_overlap_indices':hit(first,start,firsttoken)}};assert v['annotations'][i]==expected
  old=q['observer_window_annotations'][i];assert old['order']==i and old['ps_query_overlap_indices']==hit(ps,start,firsttoken) and old['docker_query_overlap_indices']==hit(docker,start,firsttoken) and old['query_through_first_fsync_overlap_indices']==hit(first,start,firsttoken)
  if j['phase']=='measurement':
   key=(j['task'],str(j['qid']),j['arm']);rep=j['repeat'];assert rep not in cells.setdefault(key,{});cells[key][rep]=firsttoken-start
 assert phase=={'warmup':6,'measurement':18} and len(cells)==6 and all(set(t)=={0,1,2} for t in cells.values()) and v['observer_uptime']==('covered' if ps[0][0]<=min(z[0] for z in bounds) and ev[-1]['end_ns']>=max(z[1] for z in bounds) else 'unknown_or_partial')
 expected=[]
 for task,qid in sorted({(k[0],k[1]) for k in cells}):
  author=cells[(task,qid,'author')];wave=cells[(task,qid,'wavefront')];expected.append({'task':task,'qid':qid,'author_ns':[author[i] for i in range(3)],'wavefront_ns':[wave[i] for i in range(3)],'author_median_ns':statistics.median(author.values()),'wavefront_median_ns':statistics.median(wave.values()),'wavefront_over_author_paired_ratios':[wave[i]/author[i] for i in range(3)]})
 assert len(expected)==3 and x['per_anchor']==expected
 out={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'jobs':24,'warmups':6,'measured':18,'anchors':3,'samples_excluded':0,'analysis_sha256':sha(analysis),'analysis_started_sha256':sha(AN/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(AN/'monitor_analysis001/terminal.json'),'coverage_sha256':sha(cov),'coverage_started_sha256':sha(C/'monitor_coverage001/started.json'),'coverage_terminal_sha256':sha(C/'monitor_coverage001/terminal.json'),'formal_sha256':sha(formal),'result_review_sha256':sha(proof),'reviewer_sha256':sha(__file__),'scope':'independent per-anchor three-repeat medians, matched-index ratios and observed-query windows','limitations':['Three posthoc anchors, three samples per arm and separate processes do not support population inference.','Equal repeat indices pair labels, not simultaneous measurement.','Second fsync, observer supervisor health and other unrecorded background have unknown overlap; all samples retained.'],'errors':[]}
 with open(a.out,'x') as fp:json.dump(out,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
