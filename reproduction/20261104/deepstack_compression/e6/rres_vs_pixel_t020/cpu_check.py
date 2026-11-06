import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((D/'plan001.json').read_text());before={k:sha(k) for k in plan['source_sha256']}
if before!=plan['source_sha256']:raise ValueError('source drift')
r=subprocess.run([sys.executable,'-m','unittest','-v','test_compare'],cwd=D,capture_output=True,text=True,timeout=180);(D/'cpu001.log').write_text(r.stdout+r.stderr)
if r.returncode:raise ValueError('tests failed')
after={k:sha(k) for k in before}
if after!=before:raise ValueError('source drift')
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','source_sha256':after,'plan_sha256':sha(D/'plan001.json'),'log_sha256':sha(D/'cpu001.log'),'GT_opened':False,'scores_parsed':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
