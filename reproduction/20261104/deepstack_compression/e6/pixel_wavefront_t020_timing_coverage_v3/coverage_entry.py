import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'coverage.py'
if hashlib.sha256(p.read_bytes()).hexdigest()!='64d4a4a8ed37e6cfc21df4cd91b05c8f66ebd78287445f37a725aadd3b1a39a3':raise ValueError('coverage source changed before import')
plan=json.loads((D/'coverage_plan001.json').read_text())
for name,expected in plan['source_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('coverage source changed')
sp=importlib.util.spec_from_file_location('_pixel_t020_observer_coverage',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m
try:sp.loader.exec_module(m)
except BaseException:del sys.modules[sp.name];raise
if __name__=='__main__':m.main()
