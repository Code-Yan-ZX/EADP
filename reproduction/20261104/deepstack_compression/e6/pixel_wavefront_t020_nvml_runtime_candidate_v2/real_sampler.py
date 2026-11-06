"""Real XML sampler continuation; no CLI, allocation, or GPU authorization."""
import json
import os
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;MEMORY=HERE
sys.path.insert(0,str(MEMORY))
from contract import identity
from sampler_io import sample_only
from nvml_source import source

def launch(directory,interval_ns,timeout_ns,*,sample=sample_only,source_factory=source):
    d=Path(directory).resolve()
    if not d.is_dir() or type(interval_ns) is not int or not 100_000_000<=interval_ns<=10_000_000_000 or type(timeout_ns) is not int or not interval_ns<timeout_ns<=36_000_000_000_000:
        raise ValueError('invalid sampler paths or bounds')
    expected=json.loads((d/'expected.json').read_text());identity(expected)
    with (d/'device_metadata.jsonl').open('x') as metadata:
        result=sample(expected,d/'samples.jsonl',d/'sampler_ready.json',d/'stopped.json',
                      d/'stopped_inspect.json',interval_ns,timeout_ns,run=source_factory(expected,metadata))
    if result.get('status')!='completed':raise ValueError('real sampler incomplete')
    final={'status':'completed','scope':'NVML process/device samples, not timing',
           'sample_count':result.get('samples'),'interval_ns':interval_ns,
           'limitations':['MiB rounding and interval sampling can miss transient peaks',
                          'device memory is not process memory; graphics processes remain background']}
    with (d/'sampler_complete.json').open('x') as f:json.dump(final,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return final

if __name__=='__main__':raise SystemExit('locked: independently reviewed real GPU admission required')
