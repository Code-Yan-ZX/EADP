"""Supervised CPU-only controller/source check; never accesses GPU."""
import json,os,subprocess,sys
import gate

def main():
    out=gate.HERE/'cpu_check004';out.mkdir(exist_ok=False)
    approval=gate.admit_cpu();before=gate.sources()
    env=dict(os.environ,PYTHONPATH=str(gate.HERE),PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-m','unittest','-v','test_guard','test_worker'],cwd=gate.BASE,
                     env=env,text=True,capture_output=True,timeout=90)
    (out/'tests.log').write_text(p.stdout+p.stderr)
    after=gate.sources()
    result={'status':'passed' if p.returncode==0 and before==after else 'failed',
            'approval':approval,'source_sha256_before':before,'source_sha256_after':after,
            'gpu_authorized':False,'gpu_used':False,'model_loaded':False,'test_exit_code':p.returncode}
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
