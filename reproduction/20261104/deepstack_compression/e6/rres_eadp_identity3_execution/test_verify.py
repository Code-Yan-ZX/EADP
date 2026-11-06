"""Run the actual verifier against synthetic files; docker is mocked."""
import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'e6/rres_eadp_identity3_v2'));sys.path.insert(0,str(ROOT/'e6/rres_eadp_timing_pilot_v4'))
import verify,formal_contract,identity_core,docker_contract
from test_identity_core import fixture,Decode
ANCHORS=(('DocVQA_VAL','64588'),('TextVQA_VAL','36511'),('AI2D_TEST','207406'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def done(p):save(p/'started.json',{'worker_pid':123});save(p/'terminal.json',{'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]})
def populate(root):
 done(root/'monitor_run001');done(root/'monitor_snapshot001');save(root/'run001/controller_complete.json',{'status':'passed'});source=root/'source.py';copy=root/'run001_sources/copy';copy.parent.mkdir(parents=True);source.write_text('fixture');copy.write_text('fixture');digest=verify.sha(source)
 save(root/'gpu_plan001.json',{'source_sha256':{str(source):digest}});save(root/'run001_sources/manifest.json',{'status':'passed','post_run_snapshot':True,'source_files':{str(source):{'sha256':digest,'copy':str(copy)}}});(root/'snapshot.py').write_text('fixture');save(root/'verification_plan001.json',{'gpu_plan_sha256':verify.sha(root/'gpu_plan001.json'),'verifier_sha256':verify.sha(verify.__file__),'snapshot_sha256':verify.sha(root/'snapshot.py')})
 image=root/'image';image.write_bytes(b'image');rows={};refs={}
 for arm in ('rres','eadp'):
  armdir=root/'run001'/arm;cid=arm*16;save(armdir/'exit.json',{'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'gpu_pids_after':[]});save(armdir/'container_identity.json',{'container_id':cid});save(armdir/'launch.json',{'arm':arm,'gpu_pids_before':[]});ob=armdir/'observations';save(ob/'worker_complete.json',{'status':'passed','anchors':3,'timing_claims':False});save(ob/'identity_endpoints.json',{'status':'passed','weights_before':[1],'weights_after':[1]})
  (armdir/'docker.log').write_text('fake diagnostic log');(ob/'progress.jsonl').write_text('{}\n')
  for task,qid in ANCHORS:
   a=fixture(arm,image);a['row'].update(task=task,qid=qid);ref=a['ref'];ref.update(task=task,qid=qid,group_id='g',image_sha256=a['row']['image_sha256'],image_start=2,eos_ids=[151645,151643]);r=identity_core.finish(**a);r['checks']['hooks_and_methods_restored']=True;r['generation_observation']={'explicit_kwargs':formal_contract.RECIPE.copy(),'effective_generation_config':{**formal_contract.RECIPE,'eos_token_id':[151645,151643]},'prepare_calls':1};r['first_criteria_length']=len(a['prompt'])+1 if arm=='rres' else 1;r['criteria_calls']=1
   raw={'arm':arm,'task':task,'qid':qid,'full_sequence':[(a['prompt'] if arm=='rres' else [])+a['output']]};save(ob/(task+'_'+qid+'.json'),r);save(ob/(task+'_'+qid+'_raw.json'),raw);rows[(task,qid)]=a['row'];refs[(arm,task,qid)]=ref
 return rows,refs
class Tests(unittest.TestCase):
 def test_real_main_recomputes_six_and_rejects_raw_tampering(self):
  for tamper in ('none','prefix','raw_arm'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);rows,refs=populate(root)
    if tamper!='none':
     p=root/'run001/rres/observations/DocVQA_VAL_64588_raw.json';x=json.loads(p.read_text())
     if tamper=='prefix':x['full_sequence'][0][0]=999
     else:x['arm']='eadp'
     save(p,x)
    def docker(command,**kwargs):
     if command[:2]!=['docker','inspect']:raise AssertionError('unexpected subprocess')
     return types.SimpleNamespace(stdout=json.dumps([{'Id':command[2],'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'HostConfig':{'Devices':[]}}]))
    refs_module=types.ModuleType('reference_inputs');refs_module.load=lambda:{'rows':rows,'references':refs};decode_module=types.ModuleType('decode_observer');decode_module.validate_trace=Decode.validate_trace
    with patch.object(verify,'HERE',root),patch.object(verify.subprocess,'run',docker),patch.object(docker_contract,'check',return_value=True),patch.dict(sys.modules,{'reference_inputs':refs_module,'decode_observer':decode_module}),patch.object(sys,'argv',['verify.py','--out',str(root/'verification001.json')]):
     if tamper!='none':
      with self.assertRaisesRegex(ValueError,'prompt prefix' if tamper=='prefix' else 'raw arm/task/qid'):verify.main()
      self.assertFalse((root/'verification001.json').exists())
     else:verify.main();self.assertEqual(json.loads((root/'verification001.json').read_text())['generations'],6)
