"""Independent GPU prelaunch audit; does not query GPU or read timing results."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final';S=B/'e6/pixel_wavefront_t020_timing_analysis_v4';X=B/'e6/pixel_wavefront_t020_timing_gate_extra_tests';OUT=I/'pixel_wavefront_t020_timing_final_v4_prelaunch_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()!=OUT.resolve() or OUT.exists() or (D/'run_001').exists():raise ValueError('unplanned output or run already started')
 planfile=D/'gpu_plan001.json';formal=A/'verification_plan001.json';stats=S/'analysis_plan001.json'
 if sha(planfile)!='196f4b82fc9a0e533189a108fd0ea585cf06ceeadc0a5a6398e9a091a396f16d' or sha(formal)!='7b966368d0f86b13af72db7be01fe3a1efbad8b0a42f74bbdc11175755bfec38':raise ValueError('unexpected frozen plans')
 p=read(planfile);f=read(formal);source=p['source_sha256'];cpu=read(A/'cpu_proof001.json')
 if len(source)!=463 or any(sha(k)!=v for k,v in source.items()) or p.get('scope')!='pixel_t020_isolated_process_ttft_24' or len(p.get('jobs',[]))!=24 or p.get('inner_timeout_seconds')!=1200 or p.get('outer_timeout_seconds')!=29400 or p.get('grace_seconds')!=120 or p.get('snapshot_seconds')!=3600 or p.get('GT_opened') is not False or p.get('GPU_authorized') is not False:raise ValueError('GPU plan/source')
 if f.get('gpu_plan_sha256')!=sha(planfile) or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=f.get('source_sha256') or cpu.get('source_sha256_after')!=f.get('source_sha256') or any(sha(k)!=v for k,v in f['source_sha256'].items()) or f.get('GT_opened') is not False:raise ValueError('formal verifier freeze')
 gatepath=D/'gpu_gate.py'
 if source.get(str(gatepath))!=sha(gatepath):raise ValueError('GPU gate source')
 spec=importlib.util.spec_from_file_location('_independent_pixel_t020_gpu_gate',gatepath);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 g.terminal(A/'monitor_cpu001',A/'cpu_check.py');g.terminal(B/'e6/pixel_wavefront_t020_timing_formal_plan/monitor_freeze001',B/'e6/pixel_wavefront_t020_timing_formal_plan/freeze.py')
 weights=g.weight_preflight();g.prerequisites()
 if p.get('weight_reference_sha256')!=weights['worker_preflight_sha256'] or p.get('weight_files')!=[x['path'] for x in weights['expected']]:raise ValueError('weight identity')
 design=read(B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json')
 if p.get('design_sha256')!=sha(B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json') or p['jobs']!=design['jobs']:raise ValueError('job design')
 cpu_extra=read(X/'proof001.json');g.terminal(X/'monitor_cpu001',X/'cpu_check.py')
 if cpu_extra.get('status')!='passed' or cpu_extra.get('GT_opened') is not False or cpu_extra.get('GPU_used') is not False or cpu_extra.get('predictions_opened') is not False or cpu_extra.get('source_sha256_before')!=cpu_extra.get('source_sha256_after') or any(sha(k)!=v for k,v in cpu_extra['source_sha256_after'].items()):raise ValueError('synthetic complete gate test')
 statproof=I/'pixel_wavefront_t020_timing_analysis_v4_prelaunch_review.json';statreviewer=I/'review_pixel_wavefront_t020_timing_analysis_v4_prelaunch.py';statmon=I/'monitor_pixel_wavefront_t020_timing_analysis_v4_prelaunch';g.terminal(statmon,statreviewer,statproof);sq=read(statproof)
 if sq.get('status')!='passed' or sq.get('independent_review') is not True or sq.get('GT_opened') is not False or sq.get('timing_results_opened') is not False or sq.get('plan_sha256')!=sha(stats) or sq.get('gpu_plan_sha256')!=sha(planfile) or sq.get('formal_plan_sha256')!=sha(formal) or sq.get('reviewer_sha256')!=sha(statreviewer) or sq.get('errors')!=[]:raise ValueError('independent statistics freeze')
 tests=[(D,['test_locks','test_raw_prefix','test_preflight_gate','test_final_main','test_weights'],18),(A,['test_formal_locks','test_pipeline','test_docker_contract','test_completion_contract','test_verify_candidate','test_window_overlap'],10),(X,['test_gate_full'],1)]
 for cwd,names,count in tests:
  run=subprocess.run([sys.executable,'-m','unittest','-v',*names],cwd=cwd,capture_output=True,text=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  if run.returncode or ('Ran %s test'%count) not in run.stderr:raise ValueError('CPU independent retest '+str(cwd)+': '+run.stderr[-1000:])
 if any(sha(k)!=v for k,v in source.items()) or any(sha(k)!=v for k,v in f['source_sha256'].items()):raise ValueError('sources changed during audit')
 proof={'status':'passed','independent_review':True,'scope':'pixel_t020_timing_gpu_prelaunch','GT_opened':False,'scoring_performed':False,'source_sha256':source,'plan_sha256':sha(planfile),'reviewer_sha256':sha(__file__),'runtime_review_sha256':sha(I/'pixel_wavefront_t020_timing_runtime_v4_preflight_result_review.json'),'run_preflight_review_sha256':sha(I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json'),'final_preflight_review_sha256':sha(I/'pixel_wavefront_t020_timing_final_v4_preflight_result_review.json'),'formal_plan_sha256':sha(formal),'formal_freeze_started_sha256':sha(B/'e6/pixel_wavefront_t020_timing_formal_plan/monitor_freeze001/started.json'),'formal_freeze_terminal_sha256':sha(B/'e6/pixel_wavefront_t020_timing_formal_plan/monitor_freeze001/terminal.json'),'analysis_plan_sha256':sha(stats),'analysis_review_sha256':sha(statproof),'analysis_review_terminal_sha256':sha(statmon/'terminal.json'),'extra_gate_test_sha256':sha(X/'proof001.json'),'extra_gate_test_terminal_sha256':sha(X/'monitor_cpu001/terminal.json'),'errors':[],'limitations':['3 posthoc DEV anchors, 3 sequential independent processes per arm; warmup does not warm later process.','Selector observation overhead included for both arms.','Background outside recorded windows remains unknown.','Endpoint weight stat checks cannot exclude transient replacement; no GPU run or timing result was inspected.']}
 with OUT.open('x') as fp:json.dump(proof,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
