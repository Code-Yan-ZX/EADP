"""Fully specified random K256 selection; uses no model, dataset or labels."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=HERE/'plan.json'

def selection(image_sha256,seed):
 plan=json.loads(PLAN.read_text())
 if seed not in plan['seeds'] or type(seed)!=int or not isinstance(image_sha256,str) or len(image_sha256)!=64 or any(x not in '0123456789abcdef' for x in image_sha256):
  raise ValueError('invalid frozen image/seed identity')
 raw=('eadp-e6-random-v1|'+str(seed)+'|'+image_sha256).encode('utf8')
 seed64=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big');counter=0;buffer=[]
 def draw(bound):
  nonlocal counter,buffer
  limit=(1<<64)-((1<<64)%bound)
  while True:
   if not buffer:
    block=hashlib.sha256(seed64.to_bytes(8,'big')+counter.to_bytes(8,'big')).digest();counter+=1
    buffer=[int.from_bytes(block[i:i+8],'big') for i in range(0,32,8)]
   value=buffer.pop(0)
   if value<limit:return value%bound
 ids=list(range(1024))
 for i in range(1023,0,-1):
  j=draw(i+1);ids[i],ids[j]=ids[j],ids[i]
 chosen=sorted(ids[:256])
 return {'image_sha256':image_sha256,'seed':seed,'derived_seed64':seed64,'algorithm':'sha256-rejection-fisher-yates-v1','spatial_ids':chosen}
