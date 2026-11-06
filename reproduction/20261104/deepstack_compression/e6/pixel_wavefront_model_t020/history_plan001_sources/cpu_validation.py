"""Source-bound synthetic validation only; no GPU, labels or model loading."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from source_inventory import source_hashes
D=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();before=source_hashes();results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for script in ('test_identity_contract.py','test_identity_worker.py','test_reference_contract.py','test_runtime_support.py','test_admission.py','test_single_arm.py','test_offline_identity.py','test_p3_reconstruct.py','test_schedule_contract.py'):
  r=subprocess.run([sys.executable,str(D/script)],capture_output=True,text=True,timeout=120,env=env);results.append({'script':script,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 after=source_hashes();status='passed' if before==after and all(x['returncode']==0 for x in results) else 'failed'
 a.out.mkdir(exist_ok=False)
 with (a.out/'verification.json').open('x') as f:json.dump({'status':status,'source_sha256_before':before,'source_sha256_after':after,'tests':results,'GT_opened':False,'gpu_used':False,'model_loaded':False,'scope':'synthetic CPU only'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if status!='passed':raise SystemExit(1)
if __name__=='__main__':main()
