"""Source-bound CPU tests; no GPU, real model, predictions or GT."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from freeze_plan import sha
from local_imports import HERE,ROOT,E4,BASE
def sources():
 old=json.loads((BASE/'cpu_check_003/verification.json').read_text());previous=json.loads((E4/'cpu_check_023/verification.json').read_text())
 files=list(HERE.glob('*.py'))+[HERE/'plan.json',BASE/'baseline_runner.py',ROOT/'m3_oracle_exec/e1_admission.py']
 for entry in (old['source_sha256_after'],previous['source_sha256_after']):
  for name,digest in entry.items():
   if sha(name)!=digest:raise ValueError('previously reviewed source changed: '+name)
   files.append(Path(name))
 plan=json.loads((ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())
 for group in ('source_sha256','implementation_sha256'):
  for name,digest in plan[group].items():
   if sha(name)!=digest:raise ValueError('author source changed: '+name)
   files.append(Path(name))
 return sorted(set(p.resolve() for p in files))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 paths=sources();before={str(p):sha(p) for p in paths};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 commands=[([sys.executable,'-m','unittest','-v','test_stock_capture','test_smoke_contract','test_local_imports','test_gate',
             'test_docker_supervisor','test_container_identity','test_container_mount','test_generation_contract','test_arm_runner'],HERE),
           ([sys.executable,'-m','unittest','-v','test_adapter','test_recover_indices'],BASE)]
 reports=[];begin=time.time()
 for i,(command,cwd) in enumerate(commands):
  try:
   r=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=180);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
  except subprocess.TimeoutExpired as exc:rc=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '');timed=True
  (out/f'{i}.stdout').write_text(stdout);(out/f'{i}.stderr').write_text(stderr)
  reports.append({'command':command,'cwd':str(cwd),'returncode':rc,'timeout':timed,'stdout_sha256':sha(out/f'{i}.stdout'),'stderr_sha256':sha(out/f'{i}.stderr')})
 after={str(p):sha(p) for p in paths};ok=before==after and all(x['returncode']==0 and not x['timeout'] for x in reports)
 proof={'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests':reports,
        'started_unix':begin,'finished_unix':time.time(),'cuda_visible_devices':'','GT_opened':False,'real_model_loaded':False,
        'generation_performed':False,'scope':'synthetic CPU only; real GPU and model execution remain locked'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
