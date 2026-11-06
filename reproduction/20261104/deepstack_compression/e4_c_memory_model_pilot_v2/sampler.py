"""Approved host-only real NVML sampler; never launches a model."""
import argparse,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;sys.path[:0]=[str(D),str(B/'e4_c_memory_nvml')]
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=D/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,check_existing
gate=checked_import('gate',D/'gate.py')
check_existing({'contract':B/'e4_c_memory/contract.py','sampler_io':B/'e4_c_memory/sampler_io.py','nvml_source':B/'e4_c_memory_nvml/nvml_source.py'})
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('sampler gate collision')
spec=importlib.util.spec_from_file_location('eadp_modelmemory_nvml_sampler',B/'e4_c_memory_nvml/real_sampler.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
check_existing({'contract':B/'e4_c_memory/contract.py','sampler_io':B/'e4_c_memory/sampler_io.py','nvml_source':B/'e4_c_memory_nvml/nvml_source.py'})
if Path(m.__file__).resolve()!=B/'e4_c_memory_nvml/real_sampler.py':raise ValueError('sampler source mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
 if a.directory.resolve()!=(D/'run_001').resolve():raise ValueError('sampler directory mismatch')
 gate.review(preflight_only=False,host=True);m.launch(a.directory,500_000_000,1_500_000_000_000)
if __name__=='__main__':main()
