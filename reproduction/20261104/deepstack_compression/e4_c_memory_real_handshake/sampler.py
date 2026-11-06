"""Host-only real sanitized XML-backed sampler; no allocation or model."""
import argparse,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;sys.path.insert(0,str(B/'e4_c_memory_nvml'))
spec=importlib.util.spec_from_file_location('eadp_nvml_sampler_real',B/'e4_c_memory_nvml/real_sampler.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if Path(m.__file__).resolve()!=B/'e4_c_memory_nvml/real_sampler.py':raise ValueError('sampler module mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
 if a.directory.resolve()!=D/'run_001':raise ValueError('sampler path mismatch')
 sys.path.insert(0,str(D));import gate
 if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('gate import collision')
 gate.verify_review();m.launch(a.directory,500_000_000,150_000_000_000)
if __name__=='__main__':main()
