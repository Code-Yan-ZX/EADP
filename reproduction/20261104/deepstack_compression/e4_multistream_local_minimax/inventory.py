"""Hash 32 previously accepted archives as raw bytes; never deserialize tensors."""
import datetime,hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;E=B/'e4_c_microbench_full';I=B/'e1_independent_review';END=1794065414
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def terminal(mon,script):
 a=read(mon/'started.json');b=read(mon/'terminal.json');cwd=Path(a['cwd'])
 if a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in a['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor')
def main():
 if datetime.datetime.now(datetime.timezone.utc).timestamp()>=END or read(B/'initial_plan.json')['budget_end_unix']!=END:raise ValueError('deadline')
 cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(E/'monitor_verify_full001',E/'full_verify.py');terminal(E/'monitor_full001',E/'full_launch.py');terminal(I/'monitor_component_full001',I/'review_component_full001.py')
 if cpu['status']!='passed' or cpu['GT_opened'] is not False or cpu['real_tensors_deserialized'] is not False or any(sha(k)!=v for k,v in cpu['source_sha256'].items()):raise ValueError('CPU proof/source')
 formal=read(E/'full_001_verification.json');review=read(I/'component_full001_review.json');schedule=B/'e4_c_microbench/schedule001.json';entries=read(schedule)['input_identities'];expected=formal['array_sha256']
 if formal.get('status')!='passed' or formal.get('images')!=32 or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False or review.get('status')!='passed' or review.get('independent_review') is not True or review.get('images')!=32 or review.get('GT_opened') is not False or review.get('errors')!=[] or review.get('formal_sha256')!=sha(E/'full_001_verification.json') or review.get('reviewer_sha256')!=sha(I/'review_component_full001.py') or formal['source_sha256'][str(schedule)]!=sha(schedule) or review['array_sha256']!=expected or len(expected)!=32 or len(entries)!=32:raise ValueError('accepted 32 source')
 if len({(x['task'],str(x['qid']),x['group_id']) for x in entries})!=32:raise ValueError('duplicate image identity')
 out=[]
 for i,x in enumerate(entries):
  p=(E/'full_001'/f'image_{i:03d}'/'reference.pt').resolve();stat=p.stat();digest=sha(p);post=p.stat()
  if str(p) not in expected or digest!=expected[str(p)] or (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=(post.st_dev,post.st_ino,post.st_size,post.st_mtime_ns,post.st_ctime_ns):raise ValueError('changed or mismatched archive')
  out.append({'index':i,**{k:x[k] for k in ('task','qid','group_id','image_sha256','message_sha256')},'path':str(p),'size_bytes':stat.st_size,'sha256':digest})
 if set(expected)!={x['path'] for x in out} or any(sha(k)!=v for k,v in cpu['source_sha256'].items()):raise ValueError('inventory/source drift')
 value={'status':'passed','scope':'32 accepted archives independently rehashed as bytes; real tensors not deserialized','images':32,'archives':out,'GT_opened':False,'real_tensors_deserialized':False,'GPU_used':False,'source_sha256':cpu['source_sha256'],'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'formal_sha256':sha(E/'full_001_verification.json'),'independent_review_sha256':sha(I/'component_full001_review.json'),'schedule_sha256':sha(schedule),'limitations':['32 posthoc DEV pilot images','Prior independent review established four real streams; this inventory hashes bytes only.','No tensor/quality analysis performed.']}
 with (D/'inventory001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
