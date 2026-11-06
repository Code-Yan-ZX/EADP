"""Offline no-GT proof checks for one-token warm-process request."""
import hashlib,struct
from pathlib import Path
import request
def check(record,raw,identity,job,phase,index):
 if record.get('status')!='completed' or any(str(record.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id')) or record.get('arm')!=job['arm'] or record.get('processor')!=identity['processor'] or record.get('recipe')!=request.RECIPE or record.get('rope_state_reset_before_request') is not True or record.get('hook_cleanup_verified') is not True:raise ValueError('record identity or request contract')
 cache=record.get('cache_check',{})
 if cache.get('initial_kv_lengths')!=[0]*36 or cache.get('initial_cache_position')!=0 or cache.get('cache_position_check_synchronizes') is not True:raise ValueError('fresh cache evidence')
 ids=record.get('selected_indices');k=record.get('selected_count')
 if not isinstance(ids,list) or ids!=sorted(set(ids)) or not 1<=len(ids)<=1024 or any(type(x) is not int or not 0<=x<1024 for x in ids) or len(ids)!=k:raise ValueError('selected indices')
 data=raw.get('full_sequence_ids');n=record.get('input_length');info=identity['processor']['input_ids']
 if not isinstance(data,list) or len(data)!=1 or not isinstance(data[0],list) or type(n) is not int or n<1 or len(data[0])!=n+1 or any(type(x) is not int for x in data[0]) or info.get('shape')!=[1,n] or info.get('dtype')!='torch.int64' or hashlib.sha256(struct.pack('<%dq'%n,*data[0][:n])).hexdigest()!=info.get('sha256') or data[0][-1]!=record.get('first_token_id'):raise ValueError('raw prompt or token')
 event=raw.get('request') or {}
 if event.get('phase')!=phase or event.get('index')!=index or event.get('block')!=job['block'] or event.get('arm')!=job['arm'] or any(str(event.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id')):raise ValueError('raw event identity')
 eos=record.get('eos_ids')
 if not isinstance(eos,list) or not eos or any(type(x) is not int for x in eos) or record.get('first_token_is_eos')!=(data[0][-1] in eos):raise ValueError('EOS identity')
 b=record.get('boundaries') or {};start=b.get('t0_before_image_read_ns');first=b.get('first_token_selected_ns');end=b.get('generation_complete_ns')
 if any(type(x) is not int for x in (start,first,end)) or not 0<start<first<=end or b.get('first_criteria_length')!=n+1 or not b.get('time_namespace') or b.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end']:raise ValueError('timing boundary')
 return {'start_ns':start,'first_ns':first,'completion_ns':end,'time_namespace':b['time_namespace']}
