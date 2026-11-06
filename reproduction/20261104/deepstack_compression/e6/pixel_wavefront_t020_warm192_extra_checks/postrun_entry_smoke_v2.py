"""Import both real postrun entrypoints with --help, without reading run data."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent
result={}
for name in ('formal_v2.py','analysis_v2.py'):
 p=E/name;r=subprocess.run([sys.executable,str(p),'--help'],capture_output=True,text=True,timeout=60)
 if r.returncode!=0 or '--out' not in r.stdout:raise ValueError(f'{name} import failed: {r.stderr}')
 result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
out=E/'postrun_entry_smoke_v2_001.json'
with out.open('x') as f:json.dump({'status':'passed','entries_sha256':result,'GPU_used':False,'timing_results_opened':False,'GT_opened':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
