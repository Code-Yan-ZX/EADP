"""CPU container import and synthetic regression probe, not a model preflight."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
source=ROOT/'e6/random_baselines'
proof=json.loads((source/'cpu_check_001/verification.json').read_text())
assert proof['status']=='passed' and proof['source_sha256_before']==proof['source_sha256_after']
for name,digest in proof['source_sha256_after'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
import torch
assert not torch.cuda.is_initialized()
result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(source),'-p','test_*.py','-v'],capture_output=True,text=True,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
assert result.returncode==0 and 'Ran 6 tests' in result.stderr and result.stderr.rstrip().endswith('OK'),result.stderr
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'passed','source_count':len(proof['source_sha256_after']),'tests':6,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'scope':'synthetic import probe; no actual worker or GPU authorization'}),flush=True)
