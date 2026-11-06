#!/usr/bin/env python3
"""Pre-measurement independent review of the nine-job memory analysis."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_nvml_execution_analysis';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept';P=B/'e6/pixel_wavefront_t020_nvml_gpu_plan'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'pixel_t020_nvml_execution_analysis_prelaunch_review.json' and not out.exists() and not (E/'run_001').exists()
 term(D/'monitor_cpu002',D/'cpu_check.py');term(P/'monitor_analysis_freeze001',P/'freeze_analysis.py')
 p=read(D/'analysis_plan001.json');c=read(D/'cpu_proof002.json');g=read(E/'gpu_plan001.json')
 assert p['scope']=='nine_process_pixel_nvml_descriptive' and p['GPU_used'] is False and p['GT_opened'] is False and p['scoring_performed'] is False and p['gpu_plan_sha256']==sha(E/'gpu_plan001.json') and p['formal_plan_sha256']==sha(A/'verification_plan001.json') and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after'] and all(sha(k)==v for k,v in c['source_sha256_after'].items()) and c['GT_opened'] is False and c['GPU_used'] is False and c['timing_results_opened'] is False
 jobs=g['jobs'];assert len(jobs)==9 and set((j['task'],j['arm']) for j in jobs)=={(t,a) for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for a in ('t0_author','t020_author','t020_wavefront')} and [j['order'] for j in jobs]==list(range(9))
 z=subprocess.run([sys.executable,str(D/'analysis.py'),'--out',str(D/'analysis001.json')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=90)
 assert z.returncode!=0 and not (D/'analysis001.json').exists() and 'monitor_pixel_t020_nvml_execution_analysis_prelaunch' in z.stderr,z.stderr[-1000:]
 spec=importlib.util.spec_from_file_location('independent_nvml_analysis',D/'analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 fake=[{'job':j,'status':'passed','stage_summary':{'unknown':None},'allocator_model_load':{},'allocator_request':{}} for j in jobs]
 formal={'results':[{'job':j,'identity':{'selected_count':i+1,'prompt_length':1,'generated_length':1},'sample_count':2,'stage_summary':{'unknown':None},'allocator_model_load':{},'allocator_request':{},'container_id':str(i),'host_pid':i+1,'gpu_uuid':'g'} for i,j in enumerate(jobs)]}
 assert len(m.rows(jobs,formal,lambda i:fake[i]))==9 and m.rows(jobs,formal,lambda i:fake[i])[0]['stage_summary']['unknown'] is None
 try:m.rows(jobs,{'results':formal['results'][:-1]},lambda i:fake[i]);raise AssertionError('partial accepted')
 except ValueError:pass
 value={'status':'passed','independent_review':True,'GT_opened':False,'timing_results_opened':False,'GPU_used':False,'errors':[],'plan_sha256':sha(D/'analysis_plan001.json'),'reviewer_sha256':sha(__file__),'checks':['three anchors by three methods; nine frozen processes','actual entry refuses before independent prelaunch and accepted complete run','source hash, plan and completed supervisors','unknown remains distinct from zero; incomplete nine rejected','formal and independent result proofs before reading raw memory results'],'limits':['sampled MiB can miss transient peaks','no TTFT or equal-compute or population claim','memory domains and stage-boundary uncertainties remain distinct']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
