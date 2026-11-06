"""Bounded synthetic host contract and source verification, no GPU work."""
import json,os,subprocess,sys
import host_admission
def main():
    out=host_admission.HERE/'cpu_check005';out.mkdir(exist_ok=False)
    approval=host_admission.admit_cpu();before=host_admission.sources()
    env=dict(os.environ,PYTHONPATH=str(host_admission.HERE),PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-m','unittest','-v','test_contract','test_runtime_gate'],cwd=host_admission.BASE,
                     env=env,capture_output=True,text=True,timeout=90)
    (out/'tests.log').write_text(p.stdout+p.stderr);after=host_admission.sources()
    result={'status':'passed' if p.returncode==0 and before==after else 'failed','gpu_authorized':False,
            'gpu_used':False,'model_loaded':False,'approval':approval,'source_sha256_before':before,
            'source_sha256_after':after,'test_exit_code':p.returncode}
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
