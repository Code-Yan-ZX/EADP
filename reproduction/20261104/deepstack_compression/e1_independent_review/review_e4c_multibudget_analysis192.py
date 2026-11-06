"""Independent 192-image paired-bootstrap and observation-window recomputation."""
import argparse,hashlib,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_c_multibudget_worker192_v6';A=B/'e4_c_multibudget_analysis192_exec';F=B/'e4_c_multibudget_worker192_v6_accept';E=B/'e4_c_multibudget_environment192_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()] and c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def eq(a,b):
 if isinstance(a,dict):assert isinstance(b,dict) and a.keys()==b.keys();[eq(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert isinstance(b,list) and len(a)==len(b);[eq(x,y) for x,y in zip(a,b)]
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(a,b)
 else:assert a==b,(a,b)
def overlap(w,a,b):return [i for i,(x,y) in enumerate(w) if x<=b and a<=y]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();assert np.__version__=='2.2.6'
 analysis=A/'analysis001.json';formal=F/'verification001.json';proof=I/'e4c_multibudget_worker192_v6_result_review.json';terminal(A/'monitor_analysis001',A/'report.py',analysis);terminal(F/'monitor_verify001',F/'verify.py',formal);terminal(I/'monitor_e4c_multibudget_worker192_v6_result',I/'review_e4c_multibudget_worker192_v6_result.py',proof)
 result=read(analysis);f=read(formal);review=read(proof);assert result['status']=='completed' and result['GT_opened'] is False and result['scoring_performed'] is False and result['samples_excluded']==0 and f['status']==review['status']=='passed' and review['formal_sha256']==sha(formal) and review['reviewer_sha256']==sha(I/'review_e4c_multibudget_worker192_v6_result.py')
 journal=D/'run_001/journal.jsonl';assert review['journal_sha256']==f['journal_sha256']==sha(journal) and review['health_sha256']==f['health_sha256']==sha(D/'monitor_run001/health.jsonl')
 proposal=read(B/'e4_c_multibudget_expansion192/proposal002.json');jobs=proposal['jobs'];events=lines(journal);assert len(jobs)==7920 and len(events)==23760
 namespace=read(D/'run_001/launch.json')['time_namespace'];health=lambda path:[x['health_snapshot_window_monotonic_ns'] for x in lines(path) if x['time_namespace']==namespace]
 obs=lines(E/'samples001.jsonl');assert obs[-1]['event']=='watch_end' and obs[-1]['reason']=='stop_marker';query=[];through=[]
 for i in range((len(obs)-1)//2):
  a,b=obs[2*i:2*i+2];assert a['event']=='snapshot' and b['event']=='logging_complete' and a['index']==b['index']==i and a['time_namespace']==b['time_namespace']==namespace and a['returncode']==0 and a['query_start_ns']<=a['query_end_ns']<=b['first_fsync_end_ns'];query.append([a['query_start_ns'],a['query_end_ns']]);through.append([a['query_start_ns'],b['first_fsync_end_ns']])
 manual=[]
 for x in lines(E/'manual_checks001.jsonl'):
  assert x['event']=='manual_health_check' and x['time_namespace']==namespace and x['start_monotonic_ns']<=x['end_monotonic_ns'];manual.append([x['start_monotonic_ns'],x['end_monotonic_ns']])
 windows={'model_health':health(D/'monitor_run001/health.jsonl'),'query':query,'query_first_fsync':through,'watcher_health':health(E/'monitor_watch001/health.jsonl'),'manual_health':manual};assert windows['model_health'] and windows['query'] and windows['watcher_health']
 vals=defaultdict(list);outcomes=defaultdict(Counter);counts={};phases=Counter();detail=[]
 for i,j in enumerate(jobs):
  segment=events[3*i:3*i+3];assert [e['event'] for e in segment]==['job_started','raw_generation','job_completed'] and all(e['job']==j for e in segment)
  rec=segment[2]['record'];raw=segment[1]['raw'];v=rec['boundaries'];a,b,c=(v[x] for x in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns'));assert v['time_namespace']==namespace and 0<=a<b<=c and raw['generation_complete_ns']==c
  hits={k:{'ttft':overlap(ws,a,b),'generation':overlap(ws,a,c)} for k,ws in windows.items()};row={k:j[k] for k in ('order','phase','task','qid','group_id','budget','protocol','replicate')};row.update(t0_before_image_read_ns=a,first_token_selected_ns=b,generation_complete_ns=c,token_count=len(rec['output_ids']),eos=rec['eos'],truncated=rec['truncated'],overlap_indices=hits);detail.append(row);phases[j['phase']]+=1
  if j['phase']=='measure':
   key=(j['task'],j['protocol'],j['group_id'],str(j['qid']),j['budget']);vals[key].append((j['replicate'],(b-a)/1e6,(c-a)/1e6));outcomes[(j['task'],j['protocol'],j['budget'])].update({'eos':int(bool(rec['eos'])),'truncated':int(bool(rec['truncated'])),'tokens':len(rec['output_ids'])})
   ck='/'.join((j['task'],j['protocol'],str(j['budget'])));entry=counts.setdefault(ck,{'jobs':0,'overlaps':{k:{'ttft':0,'generation':0} for k in windows}});entry['jobs']+=1
   for name in windows:
    for kind in ('ttft','generation'):entry['overlaps'][name][kind]+=bool(hits[name][kind])
 assert phases=={'warmup':240,'measure':7680} and len(vals)==1536 and all(len(x)==5 and {z[0] for z in x}==set(range(5)) for x in vals.values())
 eq(detail,result['jobs_detail']);eq(detail,f['environment_job_details']);eq(counts,result['health_overlap']);assert result['hourly_health_coverage']==f['hourly_health_coverage'] and result['unknown']==f['environment_limits'] and result['manual_check_evidence']==('recorded' if manual else 'unknown')
 summary=result['summary'];assert (summary['jobs'],summary['warmups'],summary['measured'],summary['images'])==(7920,240,7680,192) and summary['bootstrap']=={'replicates':10000,'seed':20261105,'unit':'image group','group_order':'ascending group_id then qid','numpy_version':np.__version__}
 conditions=0
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  for protocol in ('natural','fixed32'):
   ids=sorted({(g,q) for t,p,g,q,k in vals if t==task and p==protocol});assert len(ids)==64 and len({g for g,q in ids})==64;draw=np.random.default_rng(20261105).integers(0,64,size=(10000,64));arrays={}
   for k in (128,256,512,1024):arrays[k]=np.array([[float(np.median([v[1] for v in vals[(task,protocol,g,q,k)]])),float(np.median([v[2] for v in vals[(task,protocol,g,q,k)]]))] for g,q in ids])
   report=summary['tasks'][task][protocol]
   medians=[{'group_id':g,'qid':q,'budgets':{str(k):{'ttft_ms':float(arrays[k][i,0]),'generation_complete_ms':float(arrays[k][i,1])} for k in (128,256,512,1024)}} for i,(g,q) in enumerate(ids)];eq(medians,report['image_medians'])
   for k in (128,256,512,1024):
    conditions+=1;entry=report[str(k)];eq(entry['counts'],{'images':64,'runs':320,**dict(outcomes[(task,protocol,k)])})
    for col,label in enumerate(('ttft_ms','generation_complete_ms')):
     value=arrays[k][:,col];ref=arrays[256][:,col];assert np.isfinite(value).all() and (value>0).all() and (ref>0).all()
     def summarize(x):return {'median':float(np.median(x)),'min':float(np.min(x)),'max':float(np.max(x)),'bootstrap_ci95':[float(y) for y in np.quantile(np.median(x[draw],axis=1),[.025,.975])]}
     eq(entry[label],{'time':summarize(value),'paired_difference_vs_k256_ms':summarize(value-ref),'paired_ratio_vs_k256':summarize(value/ref)})
 assert conditions==24
 out={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'jobs':7920,'warmups':240,'measured':7680,'images':192,'conditions_verified':24,'samples_excluded':0,'analysis_sha256':sha(analysis),'analysis_started_sha256':sha(A/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(A/'monitor_analysis001/terminal.json'),'formal_sha256':sha(formal),'result_review_sha256':sha(proof),'journal_sha256':sha(journal),'reviewer_sha256':sha(__file__),'scope':'independent full image-median, paired bootstrap and observed-window recalculation','limitations':['Posthoc exploratory cohort; no confirmatory or universal conclusion.','Unknown unrecorded background and write tails remain unknown; no samples excluded.','Generate completion excludes later copy, persistence and decoding.'],'errors':[]}
 with open(args.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
