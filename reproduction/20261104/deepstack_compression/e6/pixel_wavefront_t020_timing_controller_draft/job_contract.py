"""Validate one persisted timing job; no statement about future jobs."""
import hashlib,json
from pathlib import Path

def check(job,record,reference,raw_path,job_dir):
 if any(record.get(k)!=v for k,v in job.items()):raise ValueError('job identity')
 for k,expected in (('processor','processor'),('selected_indices','selected_indices'),('output_ids','tokens'),('eos_ids','eos_ids'),('generation_scalars','generation_kwargs')):
  if record.get(k)!=reference.get(expected):raise ValueError('reference mismatch '+k)
 p=Path(raw_path).resolve()
 if not p.is_relative_to(Path(job_dir).resolve()) or not p.is_file():raise ValueError('raw output absent')
 raw=json.loads(p.read_text());full=raw.get('full_ids');length=raw.get('input_length')
 if type(length) is not int or not isinstance(full,list) or length<1 or full[length:]!=record['output_ids'] or raw.get('job')!=job:raise ValueError('raw output mismatch')
 a,b,c=(record.get(x) for x in ('start_monotonic_ns','first_token_monotonic_ns','completion_monotonic_ns'))
 if any(type(x) is not int for x in (a,b,c)) or not 0<a<b<=c or record.get('first_token_synchronized') is not True or record.get('gpu_synchronized_before_start') is not True:raise ValueError('timing boundary')
 if type(record.get('start_unix_ns')) is not int or type(record.get('completion_unix_ns')) is not int or record['start_unix_ns']>record['completion_unix_ns']:raise ValueError('wall time')
 return {'ttft_ns':b-a,'completion_ns':c-a,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_count':len(record['selected_indices'])}
