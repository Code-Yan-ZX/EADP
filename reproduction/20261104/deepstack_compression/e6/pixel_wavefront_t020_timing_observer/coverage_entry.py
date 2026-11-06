import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'coverage.py'
if hashlib.sha256(p.read_bytes()).hexdigest()!='dba7c1f691a636404d11358a5679fb8392c617be5439b7a01713c6cffe2d13fc':raise ValueError('coverage source changed before import')
plan=json.loads((D/'coverage_plan001.json').read_text())
for name,expected in plan['source_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('coverage source changed')
sp=importlib.util.spec_from_file_location('_pixel_t020_observer_coverage',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m
try:sp.loader.exec_module(m)
except BaseException:del sys.modules[sp.name];raise
if __name__=='__main__':m.main()
