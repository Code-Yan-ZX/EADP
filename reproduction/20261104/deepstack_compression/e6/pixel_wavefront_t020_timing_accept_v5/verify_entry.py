import hashlib,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'verify.py'
EXPECTED='c02886eca4f45d10a0f596766b772ecb7449a122aada237662aac23404e9973a'
if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('formal source changed before import')
spec=importlib.util.spec_from_file_location('_pixel_t020_formal_verify',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
try:spec.loader.exec_module(module)
except BaseException:del sys.modules[spec.name];raise
if __name__=='__main__':module.main()
