"""Fail closed if the isolated installed runtime files or links change."""
import hashlib,json,os
from pathlib import Path
INSTALL=Path(__file__).resolve().parent.parent/'install001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify_env():
 path=INSTALL/'env_fingerprint.json';manifest=json.loads(path.read_text());env=INSTALL/'env';seen=set();links=set()
 if manifest.get('status')!='passed' or manifest.get('root')!=str(env):raise ValueError('wrong env fingerprint')
 for root,dirs,names in os.walk(env,followlinks=False):
  for name in dirs+names:
   p=Path(root)/name;key=str(p.relative_to(env))
   if p.is_symlink():
    links.add(key)
    if manifest['symlinks'].get(key)!=os.readlink(p):raise ValueError('runtime symlink changed: '+key)
   elif p.is_file():
    seen.add(key);m=manifest['files'].get(key)
    if not m or p.stat().st_size!=m['bytes'] or sha(p)!=m['sha256']:raise ValueError('runtime file changed: '+key)
 if seen!=set(manifest['files']) or links!=set(manifest['symlinks']):raise ValueError('runtime file inventory changed')
 return sha(path)
