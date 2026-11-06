import json,tempfile,unittest
from pathlib import Path
from panel_gate import terminal,files_match,sha,execution_gate,check_preflight,validate_probe,HERE,ROOT
class GateTests(unittest.TestCase):
 def test_terminal_binds_pid_exit_command_and_output(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);s=d/'started';t=d/'terminal';script=d/'audit.py';out=d/'proof.json'
   started={'worker_pid':42,'cwd':str(d),'command':['python',str(script),'--out',str(out)]};end={'worker_pid':42,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
   s.write_text(json.dumps(started));t.write_text(json.dumps(end));terminal(s,t,script,out)
   for key,value in [('worker_pid',43),('worker_returncode',1),('reason','timeout'),('remaining_process_group',[42])]:
    bad=dict(end);bad[key]=value;t.write_text(json.dumps(bad))
    with self.assertRaises(ValueError):terminal(s,t,script,out)
 def test_preflight_rejects_log_and_command_tampering(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);prior=ROOT/'e4_panel_v2/panel_003/launch.json';cmd=json.loads(prior.read_text())['command'][:];cmd[cmd.index('--name')+1]='eadp-e2e3full-preflight-001'
   while '--device' in cmd:
    i=cmd.index('--device');del cmd[i:i+2]
   cmd[cmd.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
   image='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf';cmd=cmd[:cmd.index(image)+1]+[str(HERE/'panel_worker.py'),'--execute','--preflight-only','--e1-acceptance',str(ROOT/'e1/e1full001_completion.json'),'--e1-started',str(ROOT/'e1/monitorverifye1full001/started.json'),'--e1-terminal',str(ROOT/'e1/monitorverifye1full001/terminal.json'),'--out',str(out)]
   result={'status':'preflight_passed','cuda_initialized':False,'model_loaded':False,'predictions':False};(out/'launch.json').write_text(json.dumps({'command':cmd,'upstream_launch_sha256':sha(prior)}));(out/'stdout.log').write_text(json.dumps(result));(out/'stderr.log').write_text('')
   f={'status':'passed','returncode':0,'timed_out':False,'container_running':False,'container_exit_code':0,'container_pid':0,'interrupted':None,'errors':[],'container_stopped_verified':True,'client_exited_while_container_running':False,'container_id':'x','result':result,'worker_sha256':sha(HERE/'panel_worker.py'),'preflight_source_sha256':sha(HERE/'preflight.py'),'launch_sha256':sha(out/'launch.json'),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log')}
   check_preflight(f,out);(out/'stdout.log').write_text('changed')
   with self.assertRaises(ValueError):check_preflight(f,out)
   (out/'stdout.log').write_text(json.dumps(result));cmd[-1]='/wrong';(out/'launch.json').write_text(json.dumps({'command':cmd,'upstream_launch_sha256':sha(prior)}));f['launch_sha256']=sha(out/'launch.json')
   with self.assertRaises(ValueError):check_preflight(f,out)
 def test_probe_requires_passing_same_formal_proof_and_no_gt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);files=[root/x for x in ('proof','checker','reviewer')]
   for p in files:p.write_text(p.name)
   good={'status':'passed','GT_opened':False,'scoring_performed':False,'formal_smoke_sha256':sha(files[0]),'checker_sha256':sha(files[1]),'reviewer_sha256':sha(files[2]),'intervention_records_verified':42};validate_probe(good,*files)
   for key,value in (('status','failed'),('GT_opened',True),('scoring_performed',True),('formal_smoke_sha256','a'*64),('intervention_records_verified',41)):
    bad=dict(good);bad[key]=value
    with self.assertRaises(ValueError):validate_probe(bad,*files)
 def test_missing_mutated_evidence_and_execution_lock(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'e';f.write_text('a');saved=sha(f);files_match({str(f):saved});f.write_text('b')
   with self.assertRaises(ValueError):files_match({str(f):saved})
   with self.assertRaises(ValueError):files_match({})
  with self.assertRaisesRegex(ValueError,'locked'):execution_gate()
if __name__=='__main__':unittest.main()
