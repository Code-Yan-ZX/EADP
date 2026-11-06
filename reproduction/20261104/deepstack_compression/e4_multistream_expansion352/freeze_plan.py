"""Bind plan and metadata before reading question-level selection identities."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';S=B/'e5/selector_real32';E=B/'e1/input_preparation_002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('upstream supervisor')
def main():
 out=D/'plan_freeze001.json'
 if out.exists():raise ValueError('already frozen')
 terminal(S/'monitor_verify_run001',S/'verify.py');terminal(I/'monitor_selector_real32_run001',I/'review_selector_real32_run001.py',I/'selector_real32_run001_review.json')
 formal=read(S/'run_001_verification.json');review=read(I/'selector_real32_run001_review.json');manifest=read(S/'sample_manifest_001.json');prep=read(E/'prepare.json');ip=read(E/'independent_verification.json')
 if formal.get('status')!='passed' or formal.get('images')!=32 or formal.get('verifier_sha256')!=sha(S/'verify.py') or formal.get('source_sha256',{}).get(str((S/'sample_manifest_001.json').resolve()))!=sha(S/'sample_manifest_001.json') or review.get('status')!='passed' or review.get('independent_review') is not True or review.get('reviewer_sha256')!=sha(I/'review_selector_real32_run001.py') or review.get('formal_sha256')!=sha(S/'run_001_verification.json') or review.get('formal_terminal_sha256')!=sha(S/'monitor_verify_run001/terminal.json') or review.get('errors')!=[]:raise ValueError('unaccepted pilot provenance')
 inp=sha(E/'inputs.jsonl')
 if manifest.get('status')!='frozen' or manifest.get('inputs_sha256')!=inp or prep.get('n')!=384 or prep.get('gt_read') is not False or prep.get('inputs_sha256')!=inp or ip.get('status')!='passed' or ip.get('n_verified')!=384 or ip.get('gt_read') is not False or ip.get('inputs_sha256')!=inp:raise ValueError('E1 input provenance')
 paths=[D/'PLAN.md',D/'selection.py',D/'test_selection.py',D/'freeze_plan.py',S/'sample_manifest_001.json',S/'run_001_verification.json',I/'selector_real32_run001_review.json',E/'prepare.json',E/'independent_verification.json',E/'inputs.jsonl']
 before={str(p.resolve()):sha(p) for p in paths}
 test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_selection.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
 if test.returncode or before!={str(p.resolve()):sha(p) for p in paths}:raise ValueError('test or source drift')
 data={'status':'frozen','source_sha256':before,'tests':test.stdout+test.stderr,'selection_identity_values_opened':False,'feature_values_opened':False,'score_values_opened':False,'GT_opened':False,'predictions_opened':False,'GPU_used':False,'scope':'posthoc proposal and identity selection protocol only; no GPU or sample extraction authorization','errors':[]}
 with out.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
