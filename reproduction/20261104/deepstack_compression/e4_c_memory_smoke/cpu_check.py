"""No-GPU synthetic worker wiring tests and reviewed-source freeze."""
import json,os,subprocess,sys
import cpu_gate
def main():
    out=cpu_gate.HERE/'cpu_check002';out.mkdir(exist_ok=False)
    approval=cpu_gate.admit_cpu();before=cpu_gate.sources()
    env=dict(os.environ,PYTHONPATH=str(cpu_gate.HERE),PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-m','unittest','-v','test_worker_contract','test_wiring'],
                     cwd=cpu_gate.BASE,env=env,capture_output=True,text=True,timeout=90)
    (out/'tests.log').write_text(p.stdout+p.stderr);after=cpu_gate.sources()
    result={'status':'passed' if p.returncode==0 and before==after else 'failed','gpu_authorized':False,
            'gpu_used':False,'model_loaded':False,'approval':approval,'source_sha256_before':before,
            'source_sha256_after':after,'test_exit_code':p.returncode}
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
