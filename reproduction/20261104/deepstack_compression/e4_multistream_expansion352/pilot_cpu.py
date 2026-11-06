"""Freeze the pilot identity/timing contract only; does not run a model."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'pilot_cpu001.json'
 if out.exists():raise ValueError('already exists')
 plan=json.loads((D/'contract_freeze001.json').read_text());selection=json.loads((D/'selection001.json').read_text())
 if plan.get('status')!='frozen' or any(sha(k)!=v for k,v in plan['source_sha256'].items()) or selection.get('status')!='frozen' or len(selection['pilot'])!=9:raise ValueError('prior freeze')
 paths=[D/'pilot_contract.py',D/'test_pilot_contract.py',D/'pilot_cpu.py',D/'contract_freeze001.json',D/'selection001.json'];before={str(p):sha(p) for p in paths}
 t=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_pilot_contract.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
 if t.returncode or before!={str(p):sha(p) for p in paths}:raise ValueError('test or source drift')
 value={'status':'passed','source_sha256':before,'tests':t.stdout+t.stderr,'GPU_used':False,'GT_opened':False,'predictions_opened':False,'feature_values_opened':False,'score_values_opened':False,'scope':'synthetic identity/time/failure contract only, not a real runner or model proof','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
