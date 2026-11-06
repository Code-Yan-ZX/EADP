"""Run bounded, synthetic sampler tests and freeze source identities."""
import json,os,subprocess,sys
import admission

def main():
    out=admission.HERE/'cpu_check006';out.mkdir(exist_ok=False)
    approval=admission.admit_cpu();before=admission.sources()
    env=dict(os.environ,PYTHONPATH=str(admission.HERE),PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-m','unittest','-v','test_nvml_source'],cwd=admission.BASE,
                     env=env,text=True,capture_output=True,timeout=90)
    (out/'tests.log').write_text(p.stdout+p.stderr)
    after=admission.sources()
    result={'status':'passed' if p.returncode==0 and before==after else 'failed',
            'approval':approval,'source_sha256_before':before,'source_sha256_after':after,
            'gpu_authorized':False,'gpu_used':False,'model_loaded':False,'test_exit_code':p.returncode}
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
