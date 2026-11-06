import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=[HERE/x for x in ('contract.py','timer.py','light_adapter.py','light_capture.py','light_arm.py','light_compare.py','light_smoke_plan.py','light_smoke_core.py','light_smoke_worker.py','light_smoke_launch.py','light_smoke_budget.json','light_admission.py','light_verify.py','light_verify_records.py','light_verify_core.py','light_reference_gate.py','light_container_probe.py','light_preflight_launch.py','identity_gate.py','runner.py','real_worker.py','timing_launch.py','memory_probe.py','nvml_sampler.py','input_preflight.py','e1_reference_gate.py','timing_supervise.py','interference.py','microbench_contract.py','test_contract.py','test_runner.py','test_light_verify.py','run_cpu_checks.py','container_probe.py')]
from light_admission import sources
FILES=list(dict.fromkeys(FILES+sources()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 out=HERE/'cpu_check056';out.mkdir(exist_ok=False);before={str(x):sha(x) for x in FILES};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');start=time.time()
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_contract','test_runner','test_light_verify'],cwd=HERE,env=env,text=True,capture_output=True,timeout=120);(out/'stdout.log').write_text(r.stdout);(out/'stderr.log').write_text(r.stderr);after={str(x):sha(x) for x in FILES}
 ok=r.returncode==0 and before==after and 'Ran 50 tests' in r.stderr and r.stderr.rstrip().endswith('OK')
 proof={'status':'passed' if ok else 'failed','returncode':r.returncode,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'started_unix':start,'finished_unix':time.time(),'scope':'synthetic schedule, real tiny CPU generation and process isolation only; no GPU, real data, scoring or performance measurements'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':proof['status'],'returncode':r.returncode}));return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
