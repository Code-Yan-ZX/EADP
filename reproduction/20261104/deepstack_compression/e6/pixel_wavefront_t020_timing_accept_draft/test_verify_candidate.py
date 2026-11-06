import json,tempfile,unittest
from pathlib import Path
import verify_candidate as v
class Tests(unittest.TestCase):
 def test_missing_inventory_fails_before_any_job(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'controller.jsonl';p.write_text(json.dumps({'event':'controller_started','jobs':24})+'\n')
   with self.assertRaises(ValueError):v.validate([],{},t,lambda *a:None)
 def test_incomplete_final_denominator_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'controller.jsonl';p.write_text((json.dumps({'event':'controller_started','jobs':24})+'\n')*49+json.dumps({'event':'all_completed','jobs':23,'warmups':6,'measured':17})+'\n')
   with self.assertRaises(ValueError):v.validate([{}]*24,{},t,lambda *a:None)
if __name__=='__main__':unittest.main()

class EndToEndTests(unittest.TestCase):
 def make(self,root):
  jobs=[];refs={};journal=[{'event':'controller_started','jobs':24}]
  for i in range(24):
   j={'order':i,'phase':'warmup' if i<6 else 'measurement','task':'t','qid':str(i),'arm':'author'};jobs.append(j);refs[('t',str(i),'author')]={}
   d=Path(root)/('job_%03d'%i);d.mkdir();cid=format(i+1,'064x');host={'pid':i+1,'start_ticks':3,'boot_id':'boot'};controller={'pid':88}
   launch={'job':j,'controller':controller};(d/'launch.json').write_text(json.dumps(launch))
   (d/'release.json').write_text(json.dumps({'job':j,'controller':controller,'container_id':cid,'host_init':host,'launch_sha256':v.digest(d/'launch.json')}))
   (d/'exit.json').write_text(json.dumps({'container_identity':{'container_id':cid,'host_init':host},'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'gpu_pids_after':[],'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'failure_type':None}))
   (d/'raw.json').write_text('{}');(d/'record.json').write_text(json.dumps({'job':j,'record':{},'metrics':{'ttft_ns':i+1}}))
   journal.extend([{'event':'job_started','job':j},{'event':'job_complete','job':j,'container_id':cid,'host_init':host}])
  journal.append({'event':'all_completed','jobs':24,'warmups':6,'measured':18});(Path(root)/'controller.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in journal));return jobs,refs
 def test_complete_and_tampered_release(self):
  with tempfile.TemporaryDirectory() as t:
   jobs,refs=self.make(t);check=lambda j,*args:{'ttft_ns':j['order']+1}
   r=v.validate(jobs,refs,t,check);self.assertEqual(r['jobs'],24);self.assertEqual(len(r['files_sha256']),120)
   p=Path(t)/'job_003/release.json';x=json.loads(p.read_text());x['launch_sha256']='wrong';p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):v.validate(jobs,refs,t,check)
