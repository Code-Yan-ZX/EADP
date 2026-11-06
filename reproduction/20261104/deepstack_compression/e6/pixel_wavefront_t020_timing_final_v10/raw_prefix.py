"""Recompute stored prompt IDs hash against accepted actual processor identity."""
import hashlib,json,struct,sys
from pathlib import Path

def check(raw_path,reference):
 raw=json.loads(Path(raw_path).read_text());ids=raw.get('full_ids');length=raw.get('input_length');processor=reference.get('processor',{}).get('input_ids',{})
 if (sys.byteorder!='little' or type(length) is not int or length<=0 or not isinstance(ids,list) or len(ids)<length or processor.get('dtype')!='torch.int64' or processor.get('shape')!=[1,length] or any(type(x) is not int or x<-(2**63) or x>=2**63 for x in ids[:length])):raise ValueError('invalid prompt IDs or processor shape')
 actual=hashlib.sha256(struct.pack('<%dq'%length,*ids[:length])).hexdigest()
 if actual!=processor.get('sha256'):raise ValueError('prompt IDs differ from accepted actual processor')
 return actual
