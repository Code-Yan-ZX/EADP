"""Future supervised host NVML sampler; hard locked pending admission."""
import argparse,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate'
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
 s=importlib.util.spec_from_file_location('_pixel_nvml_sampler_gate',D/'gate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);approval=g.admit_gpu()
 directory=a.directory.resolve();allowed={(D/'run_001'/('job_%03d'%i)).resolve() for i in range(9)}
 if directory not in allowed:raise ValueError('sampler output directory')
 import hashlib
 src=approval['source_sha256']
 for name in ('real_sampler.py','nvml_source.py','contract.py','sampler_io.py','sampler.py','barrier.py','host_evidence.py'):
  path=(R/name).resolve()
  if hashlib.sha256(path.read_bytes()).hexdigest()!=src[str(path)]:raise ValueError('sampler source changed')
 for name in ('contract','sampler','sampler_io','nvml_source','barrier','host_evidence','real_sampler'):
  if name in sys.modules:raise ValueError('preloaded sampler dependency')
 sys.path.insert(0,str(R));import real_sampler
 if Path(real_sampler.__file__).resolve()!=R/'real_sampler.py':raise ValueError('wrong sampler')
 real_sampler.launch(directory,500_000_000,1_500_000_000_000)
 if g.admit_gpu()!=approval:raise ValueError('sampler approval changed')
 return 0
if __name__=='__main__':main()
