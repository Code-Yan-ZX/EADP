import copy,unittest
from panel_verify import validate_trace

def sample():
 seq=1030;k=256;start=3;end=1027;ret=list(range(start))+[start+i for i in range(k)]+list(range(end,seq));pos=[[[i for i in range(seq)]] for _ in range(3)]
 return {'arm':'priortr_k256','output_ids':[2],'trace':{'calls':[{'tokens':seq,'cache_after':[seq-(1024-k)]*36}],'layers':[{'step':0,'tokens':x} for x in [seq]*3+[seq-(1024-k)]*33],'injections':[{'rows':n,'mask_count':n} for n in (1024,1024,k)],'prunes':[{'layer':2,'range_before':[start,end],'range_after':[start,start+k]}]},'small_diagnostics':{'selected_ids_0':list(range(k)),'selection_scores_0':[1.]*k+[0.]*(1024-k),'prune_before_position_0':pos,'prune_after_position_0':[[[plane[0][i] for i in ret]] for plane in pos],'prune_after_mask_0':[[False]*start+[True]*k+[False]*(seq-end)]}}
class VerifyTests(unittest.TestCase):
 def test_pruning_and_tamper(self):
  good=sample();validate_trace(good)
  for mode in ('topk','position','injection','layer','cache'):
   bad=copy.deepcopy(good)
   if mode=='topk':bad['small_diagnostics']['selection_scores_0'][0]=-1.
   if mode=='position':bad['small_diagnostics']['prune_after_position_0'][0][0][0]=99
   if mode=='injection':bad['trace']['injections'][2]['rows']=255
   if mode=='layer':bad['trace']['layers'][5]['tokens']=1030
   if mode=='cache':bad['trace']['calls'][0]['cache_after'][0]=2
   with self.assertRaises(ValueError):validate_trace(bad)
if __name__=='__main__':unittest.main()

class E4GateTests(unittest.TestCase):
 def test_real_exit_failure_blocks_comparison(self):
  from unittest.mock import patch
  import panel_verify
  with patch.object(panel_verify.gate,'terminal',side_effect=ValueError('verifier still live')),patch.object(panel_verify.gate,'read') as reader:
   with self.assertRaisesRegex(ValueError,'still live'):panel_verify.e4_comparison_gate()
   reader.assert_not_called()
 def test_review_source_or_hash_tampering_fails(self):
  from unittest.mock import patch
  import panel_verify as v
  rec=v.BASE/'e4_panel_v2/panel_003/records.jsonl'
  f={'status':'passed','stage':'full','n_questions':384,'all_arms_validated':3072,'GT_opened':False,'scoring_performed':False,'verifier_sha256':'ok','files_sha256':{str(rec.resolve()):'ok'}}
  r={'status':'passed','n_verified':384,'arms_verified':3072,'errors':[],'GT_opened':False,'scoring_performed':False,'formal_proof_sha256':'ok','formal_verifier_started_sha256':'ok','formal_verifier_terminal_sha256':'ok','reviewer_sha256':'ok'}
  with patch.object(v.gate,'terminal'),patch.object(v.gate,'sha',return_value='ok'),patch.object(v.gate,'read',side_effect=[f,r]):
   v.e4_comparison_gate()
  r['reviewer_sha256']='tampered'
  with patch.object(v.gate,'terminal'),patch.object(v.gate,'sha',return_value='ok'),patch.object(v.gate,'read',side_effect=[f,r]):
   with self.assertRaisesRegex(ValueError,'proof or exit'):v.e4_comparison_gate()

class ContainerIdentityTests(unittest.TestCase):
 def test_expected_image_devices_and_command(self):
  import json,tempfile
  from pathlib import Path
  from unittest.mock import patch
  import panel_verify as v
  from launch import configure,PYTHON
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run_001';run.mkdir();budget=v.gate.read(v.HERE/'resource_plan.json');prior=v.gate.read(v.PREP/'smoke_001/launch.json')['command'][:];name='eadp-priortr32-001-999999';prior[prior.index('--name')+1]=name;i=prior.index(str(v.PREP/'smoke_worker.py'));cmd=configure(prior[:i]+[str(v.HERE/'panel_worker.py'),'--execute','--out',str(run)],run/'runtime_cache')
   plan=v.gate.read(v.BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');image=plan['container_image'];ctrl={'pid':999999,'start_ticks':1,'boot_id':'none'};host={'pid':999998,'start_ticks':1,'boot_id':'none'}
   launch={'controller':ctrl,'command':cmd,'worker_sha256':v.gate.sha(v.HERE/'panel_worker.py'),'launcher_sha256':v.gate.sha(v.HERE/'launch.py'),'budget_sha256':v.gate.sha(v.HERE/'resource_plan.json'),'started_unix':1}
   started={'worker_pid':999999,'command':['python',str(v.HERE/'launch.py'),'--run-id','001'],'cwd':str(run),'timeout_seconds':budget['outer_timeout_seconds'],'stale_seconds':budget['journal_stale_seconds'],'snapshot_seconds':budget['snapshot_seconds'],'poll_seconds':budget['poll_seconds'],'grace_seconds':budget['grace_seconds'],'progress_paths':[str(run/'data/generated.jsonl')]};terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':999999}
   ex={'controller':ctrl,'name':name,'container_identity':{'container_id':'abc','image':image,'host_init':host},'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'failure':None,'container_stopped_verified':True,'client_exited_while_container_running':False,'gpu_pids_after':[]}
   devices=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'];envs=[cmd[j+1] for j,x in enumerate(cmd[:-1]) if x=='--env']
   obj={'Id':'abc','Image':image,'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'Config':{'Image':image,'Cmd':cmd[cmd.index(image)+1:],'Entrypoint':[str(PYTHON)],'User':'1000:1000','WorkingDir':plan['author_root'],'Env':envs},'HostConfig':{'NetworkMode':'none','Devices':[{'PathOnHost':x,'PathInContainer':x,'CgroupPermissions':'rwm'} for x in devices]}}
   with patch.object(v.subprocess,'check_output',return_value=json.dumps([obj]).encode()):v.actual_exit(run,started,terminal,launch,ex,budget)
   for mode in ('image','device','entrypoint'):
    bad=copy.deepcopy(obj)
    if mode=='image':bad['Image']='sha256:other'
    if mode=='device':bad['HostConfig']['Devices'].pop()
    if mode=='entrypoint':bad['Config']['Entrypoint']=['/wrong/python']
    with patch.object(v.subprocess,'check_output',return_value=json.dumps([bad]).encode()):
     with self.assertRaises(ValueError):v.actual_exit(run,started,terminal,launch,ex,budget)
