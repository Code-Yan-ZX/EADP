"""One already-admitted model process, one warmup, 48 measured requests.

Unsafe continuation only; controller must prove source, inputs, process and GPU
admission. No filtering or replacement on candidate output disagreement.
"""
import hashlib,json,os,time
from pathlib import Path
def append(path,value):
 with Path(path).open('a',encoding='utf8') as f:f.write(json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def save(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(job,out,rows,references,execute):
 out=Path(out);measure=job['measured'];arm=job['arm']
 if arm not in ('author','wavefront') or len(measure)!=48 or len({(x['task'],str(x['qid'])) for x in measure})!=48:raise ValueError('block inventory')
 items=[('warmup',job['warmup'])]+[('measure',x) for x in measure];done=0
 for index,(phase,identity) in enumerate(items):
  key=(identity['task'],str(identity['qid']));row=rows[key];reference=references[key]
  rawpath=out/('raw_%03d.json'%index);event={'index':index,'phase':phase,'task':key[0],'qid':key[1],'group_id':identity['group_id'],'arm':arm,'block':job['block']}
  append(out/'journal.jsonl',dict(event,event='started',monotonic_ns=time.monotonic_ns()))
  try:
   if any(str(row.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')) or reference['processor']!=identity['processor'] or reference['image_start']!=identity['image_start'] or sha(row['image_path'])!=identity['image_sha256']:raise ValueError('request source identity')
   def raw(value):save(rawpath,{'request':event,'full_sequence_ids':value})
   record=execute(row,reference,arm,raw)
   if record.get('status')!='completed' or any(str(record.get(k))!=str(event[k]) for k in ('task','qid','group_id','arm')) or not rawpath.is_file():raise ValueError('request result incomplete')
   append(out/'journal.jsonl',dict(event,event='completed',raw_path=str(rawpath),raw_sha256=sha(rawpath),record=record,monotonic_ns=time.monotonic_ns()));done+=1
  except BaseException as exc:
   append(out/'journal.jsonl',dict(event,event='failed',error_type=type(exc).__name__,message=str(exc),raw_exists=rawpath.exists(),monotonic_ns=time.monotonic_ns()));raise
 if done!=49:raise ValueError('incomplete block')
 result={'status':'completed','order':job['order'],'block':job['block'],'arm':arm,'requests':49,'warmup':1,'measured':48,'GT_opened':False,'scoring_performed':False}
 return result
