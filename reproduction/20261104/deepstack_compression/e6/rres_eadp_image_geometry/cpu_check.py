import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((D/'plan001.json').read_text());before={k:sha(k) for k in p['source_sha256']}
if before!=p['source_sha256']:raise ValueError('source changed')
a=subprocess.run([sys.executable,'-m','unittest','-v','test_core'],cwd=D,capture_output=True,text=True,timeout=120);(D/'cpu001.log').write_text(a.stdout+a.stderr)
if a.returncode:raise ValueError('tests failed')
if any(sha(k)!=v for k,v in before.items()):raise ValueError('source changed')
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','plan_sha256':sha(D/'plan001.json'),'source_sha256':before,'GT_opened':False,'individual_dimensions_read':False,'scores_parsed':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
