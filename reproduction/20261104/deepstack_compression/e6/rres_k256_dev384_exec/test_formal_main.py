"""Drive the real formal main on 384 synthetic files, without labels or GPU."""
import hashlib,json,struct,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import verify
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class TestFormal(unittest.TestCase):
 def test_full_main_and_missing_raw_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);e=base/'exec';d=base/'panel';e.mkdir();d.mkdir();obs=d/'run001/observations';obs.mkdir(parents=True)
   gate=e/'gpu_gate.py';gate.write_text('def admit_gpu(): return {}\n');plan={'source_sha256':{str(gate):h(gate)}};write(d/'gpu_plan001.json',plan)
   # Verify the actual frozen script path while using an isolated synthetic gate.
   write(d/'verification_plan001.json',{'gpu_plan_sha256':h(d/'gpu_plan001.json'),'verifier_sha256':h(verify.__file__),'GT_opened':False})
   copy=base/'gate_copy';copy.write_text(gate.read_text());write(d/'source_snapshot001/manifest.json',{'status':'passed','plan_sha256':h(d/'gpu_plan001.json'),'files':{str(gate):{'sha256':h(gate),'path':str(copy)}}})
   def monitor(name,script):
    write(d/name/'started.json',{'worker_pid':42,'cwd':str(base),'command':['python',str(script)]});write(d/name/'terminal.json',{'worker_pid':42,'reason':'completed','worker_returncode':0,'remaining_process_group':[]})
   monitor('monitor_run001',e/'launch.py');monitor('monitor_snapshot001',e/'snapshot.py')
   cid='a'*64;identity={'container_id':cid,'image':'img'};write(d/'run001/launch.json',{'gpu_pids_before':[],'controller':{'pid':42},'command':[]});write(d/'run001/exit.json',{'controller':{'pid':42},'returncode':0,'container_exit_code':0,'timed_out':False,'container_stopped_verified':True,'gpu_pids_after':[],'container_identity':identity,'last_inspect':{'id':cid}})
   write(obs/'worker_complete.json',{'status':'completed','expected':384,'started':list(range(384)),'recorded':list(range(384)),'resolution_restored':True,'source_and_weight_endpoints_verified':True,'GT_opened':False,'scoring_performed':False})
   anchors=[];events=[];pilot=[];prefix=[1,2]+[99]*256+[3,4];length=len(prefix);digest=hashlib.sha256(struct.pack('<'+'q'*length,*prefix)).hexdigest();mask=bytearray(length);mask[2:258]=b'\1'*256
   proc={'input_ids':{'dtype':'torch.int64','shape':[1,length],'sha256':digest},'attention_mask':{'synthetic':True},'pixel_values':{},'image_grid_thw':{}}
   visual={'shape':[256,8],'sha256':'f'*64};position={'shape':[3,1,length],'sha256':'d'*64};recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    for i in range(128):
     q=str(i);group=task+q;name=task+'_'+q;identity_fields={'task':task,'qid':q,'group_id':group,'image_sha256':'a'*64,'message_sha256':'b'*64};anchor={**identity_fields,'low_processor':proc,'image_start':2,'rope_delta':[[0]],'position_ids':position};anchors.append(anchor)
     record={**identity_fields,'status':'completed','rope_reset_before_generation':True,'processor':proc,'resolution':512,'visual_tokens':256,'hooks_and_methods_restored':True,'timing_claims':False,'output_ids':[151645],'generation_scalars':recipe,'eos_ids':[151645,151643],'eos':True,'truncated':False,'answer':'x','main_visual':visual,'deepstack':[visual]*3,'injected_ds':[visual]*3,'delta':[[0]],'prefill_position_ids':position,'prefill_attention_mask':proc['attention_mask'],'block_calls':[1,1],'merger_calls':1,'ds_calls':[1]*3,'visual_mask':{'dtype':'torch.bool','shape':[1,length],'sha256':hashlib.sha256(mask).hexdigest()},'decode_trace':[{'rope_delta':[[0]],'layer_cache_lengths_after':[length]*36,'cache_position':list(range(length)),'layer_cache_lengths':[]}]}
     write(obs/(name+'.json'),record);write(obs/(name+'_generated.json'),{**{k:identity_fields[k] for k in ('task','qid','group_id')},'full_sequence':[prefix+[151645]],'input_length':length});events.extend([{'event':'started','task':task,'qid':q},{'event':'recorded','task':task,'qid':q}])
     if i==0:
      pilot.append({'task':task,'qid':q});write(base/'e6/rres_k256_execution_v2/run001/observations'/(name+'.json'),record);write(base/'e6/rres_k256_execution_v2/run001/observations'/(name+'_generated.json'),{'full_sequence':[prefix+[151645]]})
   write(base/'e6/rres_k256_execution_v2/verification001.json',{'results':pilot})
   write(d/'run_preflight001/probe/processor_probe.json',{'status':'passed','anchors':anchors});(obs/'progress.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
   devices=[{'PathOnHost':x,'PathInContainer':x,'CgroupPermissions':'rwm'} for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']]
   obj={'Id':cid,'Image':'img','State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False},'HostConfig':{'NetworkMode':'none','Devices':devices},'Config':{'Cmd':[str(e/'worker.py'),'--out',str(obs)],'Env':[]},'Mounts':[]}
   fake_tokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:SimpleNamespace(decode=lambda *a,**kw:'x'))
   with patch.object(verify,'D',d),patch.object(verify,'E',e),patch.object(verify,'P',d),patch.object(verify,'O',obs),patch.object(verify,'B',base),patch.object(verify.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([obj]))),patch('transformers.AutoTokenizer',fake_tokenizer),patch('sys.argv',['verify.py','--out',str(d/'verification001.json')]):
    verify.main();v=json.loads((d/'verification001.json').read_text());self.assertEqual(v['images'],384);self.assertGreaterEqual(len(v['files_sha256']),768)
    (d/'verification001.json').unlink();(obs/'DocVQA_VAL_0_generated.json').unlink()
    with self.assertRaises(FileNotFoundError):verify.main()
if __name__=='__main__':unittest.main()
