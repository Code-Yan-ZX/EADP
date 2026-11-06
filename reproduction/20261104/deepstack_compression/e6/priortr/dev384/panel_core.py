"""PriorTR DEV384 execution core; no admission, model, labels or CLI.

An external caller must prove the frozen sample, reviewed source, true no-GPU
container preflight, bounded container lifetime and exclusive GPU first.
"""
import json,os
from pathlib import Path
ARMS=('stock_native','author_default','priortr_k256')
EVENT=('task','qid','group_id','arm','output_ids','generation_scalars','forced')

def append(path,value):
 with Path(path).open('a',encoding='utf8') as f:
  f.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def validate(record,row,arm,e1):
 for name in ('task','qid','group_id','image_sha256','message_sha256'):
  if str(record[name])!=str(row[name]):raise ValueError('sample identity: '+name)
 if record['arm']!=arm or record['forced'] is not False:raise ValueError('incorrect arm or generation mode')
 if record['processor']!=e1['processor'] or record['generation_scalars']!=e1['generation_scalars']:raise ValueError('processor or generation differs from E1')
 ids=record['output_ids'];end=bool(ids and ids[-1] in record['eos_ids']);maximum=record['generation_scalars']['max_new_tokens']
 if not ids or maximum!=2048 or len(ids)>maximum or record['eos']!=end or record['truncated']!=bool(len(ids)==maximum and not end) or not (end or record['truncated']):raise ValueError('incomplete natural generation')
 if record['backend']!='sdpa' or record['precision']!='bfloat16':raise ValueError('wrong author environment')
 if record['arrays']['kind'] not in ('full_arrays','summary_only'):raise ValueError('archive type')
 if not record['trace']['calls'] or not record['trace']['layers'] or len(record['trace']['injections'])!=3:raise ValueError('missing actual model trace')
 prunes=record['trace']['prunes']
 if arm=='stock_native' and prunes:raise ValueError('stock unexpectedly pruned')
 if arm!='stock_native' and (len(prunes)!=1 or prunes[0]['layer']!=2):raise ValueError('missing layer-3 pruning')
 if arm=='priortr_k256' and prunes[0]['range_after'][1]-prunes[0]['range_after'][0]!=256:raise ValueError('actual visual count differs from 256')

def run(*,rows,e1_references,out,invoke):
 """Arm-major to permit one stock→patched class swap in a single model.

 invoke(row, arm, on_generated, save_full_arrays) must journal generation
 immediately after model.generate and before any decoding/trace validation.
 """
 out=Path(out);out.mkdir(exist_ok=False);(out/'arrays').mkdir()
 if len(rows)!=384 or len({(r['task'],str(r['qid'])) for r in rows})!=384:raise ValueError('sample denominator')
 anchors={next((r['task'],str(r['qid'])) for r in rows if r['task']==t) for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')}
 for arm in ARMS:
  for row in rows:
   key=(row['task'],str(row['qid']));written=[]
   def on_generated(event):
    if written or set(event)!=set(EVENT):raise ValueError('duplicate or invalid generation event')
    append(out/'generated.jsonl',event);written.append(event)
   rec=invoke(row,arm,on_generated,key in anchors)
   if len(written)!=1 or written[0]!={k:rec[k] for k in EVENT}:raise ValueError('generation event mismatch')
   validate(rec,row,arm,e1_references[key])
   if rec['arrays']['kind']!=('full_arrays' if key in anchors else 'summary_only'):raise ValueError('archive policy')
   append(out/'records.jsonl',rec)
 result={'questions':384,'natural_generations':1152,'GT_opened':False,'scoring_performed':False}
 with (out/'worker_complete.json').open('x') as f:json.dump(result,f);f.write('\n');f.flush();os.fsync(f.fileno())
 return result
