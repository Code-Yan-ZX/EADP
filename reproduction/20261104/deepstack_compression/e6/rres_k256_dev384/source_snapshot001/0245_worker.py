"""Same-image no-device visibility probe for config/tokenizer aliases, no weights."""
import hashlib,json,os
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('device not disabled')
old=json.loads((B/'e6/rres_k256_pilot/preflight_plan001.json').read_text())['source_sha256'];src='/data/models/huggingface/hub/';dst='/home/dell/.cache/huggingface/hub/';paths={}
for key,digest in old.items():
 if key.startswith(src):
  visible=dst+key[len(src):];p=Path(visible)
  if 'safetensors' in p.name:raise ValueError('weight payload forbidden')
  h=hashlib.sha256(p.read_bytes()).hexdigest()
  if h!=digest:raise ValueError('alias hash mismatch')
  paths[key]={'container_path':visible,'sha256':h,'size':p.stat().st_size}
if len(paths)!=10:raise ValueError('model config inventory')
out=D/'run001/worker.json'
with out.open('x') as f:json.dump({'status':'passed','paths':paths,'CUDA_VISIBLE_DEVICES':'','weights_opened':False,'model_loaded':False,'GT_opened':False,'errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
