"""Bounded fake-data-only tests for the draft verifier; no GPU authorization."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def hashes():return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
before=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v','test_contract','test_verify'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);after=hashes();(D/'cpu_test002.log').write_text(r.stdout+r.stderr)
v={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'limitations':['synthetic job only, actual nine-job process and real sampler not yet verified','formal verification plan and independent review required before GPU']}
with (D/'cpu_proof002.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if v['status']!='passed':raise SystemExit(1)
