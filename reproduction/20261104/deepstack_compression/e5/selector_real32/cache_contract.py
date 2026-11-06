"""Check actual per-run Triton cache writability without using CUDA."""
import os,tempfile
from pathlib import Path

def probe_cache(path):
 path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
 if not path.is_dir():raise ValueError('Triton cache is not a directory')
 with tempfile.NamedTemporaryFile(prefix='.permission-probe-',dir=path) as f:
  f.write(b'E5 Triton cache writable\n');f.flush();os.fsync(f.fileno())
 return {'path':str(path),'writable':True,'uid':os.geteuid(),'gid':os.getegid()}
