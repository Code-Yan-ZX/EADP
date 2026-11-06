import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import verify
class VerifyTest(unittest.TestCase):
 def test_actual_verifier_job_core_and_failures(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);(out/'sampler_monitor').mkdir();job={'order':0,'arm':'t0_author'};seq=[7,8,9];prompt=torch.tensor([[7,8]],dtype=torch.int64).view(torch.uint8).numpy().tobytes();ref={'processor':{'input_ids':{'dtype':'torch.int64','shape':[1,2],'sha256':hashlib.sha256(prompt).hexdigest()}},'selected_indices':[1,5],'tokens':[9],'eos_ids':[9],'generation_kwargs':{'x':1}}
   expected={'container_id':'a'*64,'nonce':'n','mode':'t0_author','host_pid':7,'start_ticks':8,'boot_id':'b','gpu_uuid':'GPU-x','max_sample_gap_ns':100};ready=dict(expected,cuda_initialized=False,namespace_pid=1);release=dict(expected,status='release',first_sample_ns=2);stopped=dict(expected,monotonic_ns=20,inspect_sha256='');sample={'type':'sample','query_end_ns':2};start={'status':'sampling'};finish={'type':'summary','samples':1};events=[start,sample,finish];marks=[{'event':x,'monotonic_ns':i+3} for i,x in enumerate(('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'))]
   for m in (marks[2],marks[4]):m.update(allocated_peak_bytes=1,reserved_peak_bytes=2)
   ident={'status':'passed','job':job,'processor':ref['processor'],'selected_indices':ref['selected_indices'],'output_ids':[9],'eos_ids':[9],'generation_kwargs':{'x':1},'GT_opened':False,'scoring_performed':False};audit={'status':'passed'};summary={'status':'passed','job':job,'stage_summary':{'fake':'ok'},'allocator_model_load':{'allocated_peak_bytes':1,'reserved_peak_bytes':2},'allocator_request':{'allocated_peak_bytes':1,'reserved_peak_bytes':2}}
   def write(name,value):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
   write('running_inspect.json',[]);write('stopped_inspect.json',[]);stopped['inspect_sha256']=verify.sha(out/'stopped_inspect.json')
   for name,val in {'launch.json':{'job':job,'plan_sha256':'plan','nonce':'n','gpu_uuid':'GPU-x','sampler_worker':['worker']},'expected.json':expected,'worker_ready.json':ready,'release.json':release,'stopped.json':stopped,'controller_result.json':{'status':'passed','phase':'completed','error':None,'container_id':expected['container_id'],'audit':audit},'memory_summary.json':summary,'identity.json':ident,'raw.json':{'job':job,'full_sequence_ids':seq,'accepted_input_length':2},'worker_complete.json':{'status':'completed','job':job,'post_source_and_weight_check':True},'sampler_ready.json':{'started':start,'first_sample':sample},'sampler_monitor/started.json':{'worker_pid':12,'process_group':12,'command':['worker']},'sampler_monitor/terminal.json':{'worker_pid':12,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}}.items():write(name,val)
   for name,arr in [('samples.jsonl',events),('device_metadata.jsonl',[{'fake':1}]),('worker_events.jsonl',marks)]: (out/name).write_text(''.join(json.dumps(x)+'\n' for x in arr))
   source={str((verify.D/'contract.py').resolve()):verify.sha(verify.D/'contract.py')};proto=SimpleNamespace(admit_release=lambda *x:release,audit=lambda *x:audit);meta=SimpleNamespace(verify=lambda *x:None);stage=SimpleNamespace(summarize=lambda *x:{'fake':'ok'});docker=SimpleNamespace(stdout=json.dumps([{'Id':expected['container_id'],'State':{'Running':False,'Pid':0,'ExitCode':0,'OOMKilled':False}}]));agreement={'plan_sha256':'plan'}
   def run():
    sys.modules.pop('_pixel_nvml_accept_contract_0',None)
    with patch.object(verify.subprocess,'run',return_value=docker):return verify.verify_job(job,out,ref,source,agreement,stage,meta,proto,torch)
   self.assertEqual(run()['sample_count'],1)
   write('raw.json',{'job':job,'full_sequence_ids':[8,7,9],'accepted_input_length':2})
   with self.assertRaisesRegex(ValueError,'raw prompt mismatch'):run()
if __name__=='__main__':unittest.main()
