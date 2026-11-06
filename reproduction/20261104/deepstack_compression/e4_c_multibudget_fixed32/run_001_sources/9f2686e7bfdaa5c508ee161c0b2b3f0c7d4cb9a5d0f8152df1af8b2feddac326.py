"""Pre-timing processor identity and image span validation.

This reads/decodes image data before timing and therefore establishes a warm
file/cache condition; it must not be presented as a cold-cache measurement.
The accepted reference must itself be independently verified and bound to the
same task/qid/group/message/image before this function is called.
"""
import hashlib

def tensor_meta(tensor):
 import torch
 x=tensor.detach().contiguous().cpu()
 return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def validate_processor(torch,actual,expected,image_token_id,expected_start):
 keys=('input_ids','attention_mask','pixel_values','image_grid_thw')
 if set(expected)!=set(keys):raise ValueError('missing processor reference')
 for key in keys:
  if key not in actual or tensor_meta(actual[key])!=expected[key]:raise ValueError('processor identity mismatch: '+key)
 ids=actual['input_ids'].detach().cpu()
 if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('batch must be one')
 positions=torch.where(ids[0]==image_token_id)[0]
 if positions.numel()!=1024:raise ValueError('expected 1024 visual tokens')
 if type(expected_start) is not int or not torch.equal(positions,torch.arange(expected_start,expected_start+1024)):
  raise ValueError('visual token span mismatch')
 return {'image_start':expected_start,'visual_tokens':1024,'cache_condition':'warm: input preflight read/decode',
         'processor':{key:expected[key] for key in keys}}

def bind_e1_references(rows,records):
 """Use identity-accepted E1 DEV P3-D0 records to bind processor and span.

 Caller must first validate the complete E1 verifier/reviewer and the hashes of
 both inputs and records. Only identity fields, not predictions, are returned.
 """
 if len(rows)!=384 or any(r.get('split')!='DEV' for r in rows):raise ValueError('full DEV inputs required')
 inputs={(r['task'],str(r['qid'])):r for r in rows}
 if len(inputs)!=384:raise ValueError('duplicate input identity')
 found={}
 for record in records:
  if record.get('arm')!='P3-D0' or record.get('forced') is not False:continue
  key=(record['task'],str(record['qid']));row=inputs.get(key)
  if row is None or key in found:raise ValueError('unknown or duplicate E1 reference')
  if any(row.get(k)!=record.get(k) for k in ('task','group_id','message_sha256')):raise ValueError('E1 input/reference mismatch')
  processor=record.get('processor');start=record.get('image_start')
  if (not isinstance(processor,dict) or set(processor)!={'input_ids','attention_mask','pixel_values','image_grid_thw'} or
      any(not isinstance(v,dict) or not isinstance(v.get('dtype'),str) or not isinstance(v.get('shape'),list) or
          not isinstance(v.get('sha256'),str) or len(v['sha256'])!=64 for v in processor.values()) or
      type(start) is not int or start<0):raise ValueError('missing processor identity')
  found[key]={'task':key[0],'qid':key[1],'group_id':row['group_id'],'message_sha256':row['message_sha256'],
              'image_sha256':row['image_sha256'],'processor':processor,'image_start':start}
 if len(found)!=384:raise ValueError('incomplete E1 references')
 return found

def load_accepted_e1_references(acceptance,started,terminal):
 """Run E1 verification in an isolated interpreter to avoid module collisions.

 It rechecks accepted files and verifier exit, reads only accepted E1 DEV
 artifacts, and returns identity fields without predictions or GT.
 """
 import json,subprocess,sys
 from pathlib import Path
 script=Path(__file__).resolve().parent/'e1_reference_gate.py'
 result=subprocess.run([sys.executable,str(script),str(acceptance),str(started),str(terminal)],capture_output=True,text=True,timeout=600)
 if result.returncode!=0:raise ValueError('E1 identity subprocess failed: '+result.stderr[-2000:])
 payload=json.loads(result.stdout);rows=payload['rows'];refs={(r['task'],str(r['qid'])):r for r in payload['references']}
 if len(rows)!=384 or len(refs)!=384:raise ValueError('E1 identity subprocess inventory mismatch')
 return rows,refs
