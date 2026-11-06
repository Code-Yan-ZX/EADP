"""Component-profile worker building blocks; GPU entry remains locked.

One DEV image at a time is captured outside profiler intervals. Callers must
first verify the independent real-model identity, input/model fingerprints,
exclusive GPU, supervisor and retained Docker lifecycle. This file grants no
admission and cannot run a GPU model as a CLI.
"""
import sys
from pathlib import Path
from capture import CaptureInputs
from runner import replay

HERE=Path(__file__).resolve().parent
LATENCY=HERE.parent/'e4_c_latency'
TIMING=HERE.parent/'e4_c_timing'
sys.path.insert(0,str(LATENCY))


def capture_one(torch,wrapper,native,row,author_generate,journal):
 """Collect an actual C-ON prefill, then verify no-hook component replays."""
 from light_adapter import LightAdapter
 original=wrapper.model.generate
 def once(*args,**kwargs):
  kwargs['max_new_tokens']=1
  result=original(*args,**kwargs)
  ids=result.detach().cpu().tolist()
  journal({'status':'capture_generated','task':row['task'],'qid':str(row['qid']),
           'group_id':row['group_id'],'output_ids':ids,'scope':'diagnostic capture; not timing'})
  return result
 wrapper.model.generate=once
 try:
  with torch.inference_mode():
   with LightAdapter(torch,wrapper,native,'C-ON',row['image_start']) as adapter:
    with CaptureInputs(torch,wrapper,adapter) as capture:
     author_generate(wrapper,row['message'],dataset=row['task'])
    if not adapter.state['prefill_seen']:raise ValueError('capture did not reach prefill')
   # Both CaptureInputs and LightAdapter must be removed before replay.
   operations=capture.prepare_operations(row['image_start'])
 finally:wrapper.model.generate=original
 return operations


def profile(torch,schedule,expected,rows,identities,preflight,capture,event_factory,synchronize,journal):
 """Run already-admitted operations; input and group bindings are fail-closed."""
 bykey={(r['task'],str(r['qid'])):r for r in rows}
 byidentity={(r['task'],r['group_id']):r for r in identities}
 if len(bykey)!=len(rows) or len(byidentity)!=32:raise ValueError('input group inventory')
 for task,group in schedule['images']:
  ref=byidentity.get((task,group));row=bykey.get((task,str(ref['qid']))) if ref else None
  if row is None or ref is None or row.get('split')!='DEV' or any(str(row.get(k))!=str(ref.get(k)) for k in ('task','qid','group_id','message_sha256','image_sha256','image_start')):
   raise ValueError('component image identity mismatch')
 current=None;operations=None
 def prepare(task,group,component):
  nonlocal current,operations
  key=(task,group)
  if key!=current:
   ref=byidentity[key];row=bykey[(task,str(ref['qid']))];preflight(row,ref)
   operations=capture(row);current=key
  if set(operations)!=set(schedule['components']):raise ValueError('component operation inventory')
  ref=byidentity[key]
  identity={k:ref[k] for k in ('task','qid','group_id','message_sha256','image_sha256','image_start')}
  identity['component']=component
  return operations[component],identity
 with torch.inference_mode():
  return replay(schedule,expected,prepare,event_factory,synchronize,journal)


def main():
 raise SystemExit('GPU profiler is locked: real-model identity, independent review and bounded launcher required')
if __name__=='__main__':main()
