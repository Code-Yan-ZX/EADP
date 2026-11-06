"""Run unchanged author selector tests against original selector.py, isolating missing package logger."""
import hashlib,importlib.util,json,os,runpy,subprocess,sys,types
from pathlib import Path
B=Path(__file__).resolve().parent.parent;repo=B/'external_sources/stacktok_571bdc6';source=repo/'stacktok/core/selector.py';test=repo/'tests/test_selector.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
if commit!='571bdc6c4c6b3c92a94fe2a92e3aaf42b989cc96':raise ValueError('commit mismatch')
before={str(p):sha(p) for p in (source,test)}
spec=importlib.util.spec_from_file_location('_isolated_author_stacktok_selector',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
package=types.ModuleType('stacktok');package.__path__=[];core=types.ModuleType('stacktok.core');core.StackTokSelector=m.StackTokSelector;core.build_matrices=m.build_matrices;core.compute_beta=m.compute_beta;sys.modules['stacktok']=package;sys.modules['stacktok.core']=core
import torch
if torch.cuda.is_available():raise ValueError('CPU-only run requires CUDA hidden')
runpy.run_path(str(test),run_name='__main__')
if before!={str(p):sha(p) for p in (source,test)}:raise ValueError('source changed')
out={'status':'passed','author_commit':commit,'python':sys.version,'interpreter':sys.executable,'torch':torch.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tests':5,'source_sha256':before,'reviewer_sha256':sha(Path(__file__)),'limitations':['Original test file and selector.py executed unchanged; synthetic stacktok.core module bypassed package/runtime imports because loguru is missing.','Five toy CPU tests do not test Qwen3 integration, GPU behavior, numerical identity, performance, or actual text encoding.'],'GT_opened':False,'predictions_opened':False,'gpu_used':False}
with (Path(__file__).parent/'isolated_test001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
