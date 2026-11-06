"""Independent CPU/code review, permitting only model-free M3 preflight."""
import argparse,hashlib,json,os,pathlib,subprocess,sys
B=pathlib.Path(__file__).resolve().parents[1];D=B/'m3_gpu_smoke';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();assert not a.out.exists()
cpu=D/'cpu_check_006/verification.json';cs=D/'monitor_cpu006/started.json';ct=D/'monitor_cpu006/terminal.json';v=read(cpu);s=read(cs);t=read(ct)
assert v['status']=='passed' and v['returncode']==0 and not v['timeout'] and v['source_sha256_before']==v['source_sha256_after'] and v['cuda_visible_devices']==''
assert all(sha(p)==d for p,d in v['source_sha256_after'].items()) and v['stdout_sha256']==sha(D/'cpu_check_006/stdout.log') and v['stderr_sha256']==sha(D/'cpu_check_006/stderr.log')
assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and any(pathlib.Path(x).name=='cpu_checks.py' for x in s['command'])
assert (pathlib.Path(s['cwd'])/s['command'][s['command'].index('--out')+1]).resolve()==cpu.parent
sys.path.insert(0,str(D));from local_imports import local
up=local('gate').upstream(host=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
run=subprocess.run([sys.executable,str(D/'cpu_tests.py')],cwd=D,env=env,capture_output=True,text=True,timeout=120);assert run.returncode==0 and 'Ran 10 tests' in run.stderr and run.stderr.rstrip().endswith('OK')
assert all(sha(p)==d for p,d in v['source_sha256_after'].items())
result={'status':'passed','independent_review':True,'scope':'TEMPORARY no-GPU/no-model M3 full container preflight only; not authorization for GPU','reviewer_path':str(pathlib.Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'source_sha256':v['source_sha256_after'],'oracle_review_sha256':up['oracle_review_sha256'],'native_review_sha256':up['native_review_sha256'],'sample_review_sha256':up['sample_review_sha256'],'checks':['independent re-run 10 CPU tests','fixed synthetic target, full visual input, P3/autograd semantics','native and teacher first logits compared to real accepted native anchor under frozen threshold','four input gradient leaves and full tensor archive','Docker failure stop/kill and exit proof','reviewer supervisor and preflight log binding'],'limitations':['real 8B GPU forward/backward remains unrun','teacher vs native tolerance is prospective and must fail closed if exceeded','synthetic target no oracle or accuracy claim'], 'GT_opened':False,'scoring_performed':False,'gpu_used':False}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed',sha(a.out))
