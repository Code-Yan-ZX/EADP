import hashlib,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def load(name,p,expected):
 p=Path(p).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('source mismatch before import')
 old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=p:raise ValueError('foreign module')
  return old
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
if __name__=='__main__':
 gate=load('_pixel_t020_analysis_gate',D/'gate.py','4bb5a5a80356bcf40224987026ab6e2f2f1461ffcce1462b22444907c79e3769');a=gate.prelaunch();a.update(gate.result_admit());p=gate.read(D/'analysis_plan001.json');src=D/'analysis.py';mod=load('_pixel_t020_analysis_body',src,p['source_sha256'][str(src.resolve())]);mod.main(a)
