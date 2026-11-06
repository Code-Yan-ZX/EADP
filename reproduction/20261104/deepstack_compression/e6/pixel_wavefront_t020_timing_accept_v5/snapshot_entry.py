import hashlib,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'snapshot.py'
EXPECTED='aec89fbbbf935fad673219dca2c84f163575189b5e7c98695315e920ae98d87c'
if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('formal source changed before import')
spec=importlib.util.spec_from_file_location('_pixel_t020_formal_snapshot',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
try:spec.loader.exec_module(module)
except BaseException:del sys.modules[spec.name];raise
if __name__=='__main__':module.main()
