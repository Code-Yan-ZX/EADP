"""Freeze pure contract tests before opening chosen real identities."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
FILES=('SELECTION_PLAN.md','selection_plan_freeze001.json','SCHEDULE_CONTRACT.md','contract.py','test_contract.py','cpu_check.py','freeze_proposal.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'cpu_proof001.json'
 if out.exists():raise ValueError('proof exists')
 before={str((D/x).resolve()):sha(D/x) for x in FILES};freeze=json.loads((D/'selection_plan_freeze001.json').read_text())
 if freeze.get('status')!='frozen_before_selection' or freeze.get('selection_plan_sha256')!=sha(D/'SELECTION_PLAN.md') or freeze.get('GT_opened') is not False or freeze.get('sample_identities_opened') is not False or freeze.get('GPU_authorized') is not False:raise ValueError('selection plan not pre-frozen')
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_contract'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);after={str((D/x).resolve()):sha(D/x) for x in FILES}
 result={'status':'passed' if r.returncode==0 and before==after else 'failed','returncode':r.returncode,'test_output':r.stdout+r.stderr,'source_sha256_before':before,'source_sha256_after':after,'sample_identities_opened':False,'GT_opened':False,'predictions_opened':False,'GPU_authorized':False}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
