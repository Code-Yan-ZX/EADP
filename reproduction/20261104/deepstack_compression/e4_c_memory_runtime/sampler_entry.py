"""Bounded host-only sampler entry for future reviewed controller.

It samples only a controller-bound host PID/GPU UUID, never initializes CUDA,
and does not authorize the memory worker or launch a container.
"""
import argparse,json,os,sys
from pathlib import Path
MEMORY=Path(__file__).resolve().parent.parent/'e4_c_memory'
sys.path.insert(0,str(MEMORY))
from contract import identity
from sampler_io import sample_only

def launch(directory,expected_path,interval_ns,timeout_ns,sample=sample_only):
 d=Path(directory).resolve();e=Path(expected_path).resolve()
 if e!=d/'expected.json' or not d.is_dir():raise ValueError('sampler directory/identity path')
 expected=json.loads(e.read_text());identity(expected)
 if type(interval_ns) is not int or not 100_000_000<=interval_ns<=10_000_000_000 or type(timeout_ns) is not int or not interval_ns<timeout_ns<=36_000_000_000_000:raise ValueError('sampler bounds')
 out=sample(expected,d/'samples.jsonl',d/'sampler_ready.json',d/'stopped.json',d/'stopped_inspect.json',interval_ns,timeout_ns)
 if out.get('status')!='completed':raise ValueError('sampler did not complete')
 with (d/'sampler_complete.json').open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--interval-ns',type=int,required=True);p.add_argument('--timeout-ns',type=int,required=True);a=p.parse_args()
 launch(a.directory,a.expected,a.interval_ns,a.timeout_ns)
if __name__=='__main__':main()
