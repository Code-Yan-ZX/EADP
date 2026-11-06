"""Independent complete nine-process NVML analysis recomputation."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');I=B/'e1_independent_review';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept';X=B/'e6/pixel_wavefront_t020_nvml_execution_analysis';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert len(c)>=2 and (cwd/c[1]).resolve()==script.resolve() and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();assert out==I/'pixel_t020_nvml_execution_analysis_review.json' and not out.exists()
 p=X/'analysis001.json';formal=A/'verification001.json';review=I/'pixel_t020_nvml_execution_result_review.json';reviewer=I/'review_pixel_t020_nvml_execution_result.py';terminal(X/'monitor_analysis001',X/'analysis.py',p);terminal(A/'monitor_verify001',A/'verify.py',formal);terminal(I/'monitor_pixel_t020_nvml_execution_result',reviewer,review)
 q=read(review);v=read(formal);x=read(p);assert q['status']=='passed' and q['independent_review'] is True and q['jobs']==9 and q['formal_sha256']==sha(formal) and q['reviewer_sha256']==sha(reviewer) and q['errors']==[] and v['status']=='passed' and v['jobs']==9 and x['status']=='passed' and x['jobs']==9 and x['formal_sha256']==sha(formal) and x['independent_result_sha256']==sha(review) and x['analysis_plan_sha256']==sha(X/'analysis_plan001.json') and x['GT_opened'] is False and x['scoring_performed'] is False and x['timing_claims'] is False
 sp=importlib.util.spec_from_file_location('_independent_nvml_result_review',reviewer);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
 jobs=read(E/'gpu_plan001.json')['jobs'];assert len(jobs)==len(v['results'])==len(x['results'])==9
 for i,job in enumerate(jobs):
  d=E/'run_001'/('job_%03d'%i);expected=read(d/'expected.json');raw=read(d/'memory_summary.json');ms=lines(d/'device_metadata.jsonl');ev=lines(d/'worker_events.jsonl');names=('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end');marks={e['event']:e for e in ev if e.get('event') in names};assert set(marks)==set(names)
  bounds={'release':marks['barrier_released']['monotonic_ns'],**{n:marks[n]['monotonic_ns'] for n in names[1:]}};calc=m.reconstruct(ms,expected['host_pid'],expected['gpu_uuid'],bounds,expected['max_sample_gap_ns']);f=v['results'][i];r=x['results'][i]
  assert job==r['job']==f['job']==raw['job'] and r['selected_count']==f['identity']['selected_count'] and r['prompt_length']==f['identity']['prompt_length'] and r['generated_length']==f['identity']['generated_length'] and r['sample_count']==f['sample_count']==len(ms) and r['container_id']==f['container_id']==expected['container_id'] and r['host_pid']==f['host_pid']==expected['host_pid'] and r['gpu_uuid']==f['gpu_uuid']==expected['gpu_uuid']
  assert all(calc[k]==r['stage_summary'][k]==f['stage_summary'][k] for k in calc) and r['stage_summary']==raw['stage_summary'] and r['allocator_model_load']==f['allocator_model_load']==raw['allocator_model_load'] and r['allocator_request']==f['allocator_request']==raw['allocator_request']
  for event,name in (('model_load_end','allocator_model_load'),('request_end','allocator_request')):assert r[name]=={k:marks[event][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')}
 proof={'status':'passed','independent_review':True,'jobs':9,'stages':36,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'analysis_sha256':sha(p),'analysis_started_sha256':sha(X/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(X/'monitor_analysis001/terminal.json'),'formal_sha256':sha(formal),'independent_result_sha256':sha(review),'result_reviewer_sha256':sha(reviewer),'reviewer_sha256':sha(__file__),'errors':[],'limits':['MiB sampled peaks can miss transient peaks; interval windows may extend across stage boundaries.','Target process, whole device and PyTorch allocator are distinct; absent/unknown are not zero.','Three posthoc DEV anchors with different dynamic K do not establish a population effect or equal compute; no TTFT claim.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
