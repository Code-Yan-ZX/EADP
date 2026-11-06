"""Post-run no-GT verifier for nine isolated Pixel memory jobs.

This draft cannot authorize a GPU run. It expects a separately frozen plan,
valid prelaunch approval and all nine clean lifecycles before emitting a proof.
"""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v6';R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate';H=B/'e6/pixel_wavefront_t020_nvml_host_candidate';REF=B/'e6/pixel_wavefront_t020_nvml_reference'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(v) for v in Path(p).read_text().splitlines() if v.strip()]
def module(name,path,source):
 path=Path(path).resolve()
 if name in sys.modules or source.get(str(path))!=sha(path):raise ValueError('source/module changed')
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
 try:spec.loader.exec_module(obj)
 except BaseException:del sys.modules[name];raise
 return obj
def terminal(path,expected_command=None):
 s=read(path/'started.json');t=read(path/'terminal.json')
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or s.get('process_group')!=s['worker_pid'] or (expected_command is not None and s.get('command')!=expected_command):raise ValueError('supervisor did not cleanly exit')
 return s,t

def verify_job(job,out,reference,source,agreement,stages,metadata,protocol,torch,guard=None,image=None):
 launch=read(out/'launch.json');expected=read(out/'expected.json');ready=read(out/'worker_ready.json');release=read(out/'release.json');stopped=read(out/'stopped.json');result=read(out/'controller_result.json');summary=read(out/'memory_summary.json');identity=read(out/'identity.json');raw=read(out/'raw.json');complete=read(out/'worker_complete.json')
 if launch.get('job')!=job or launch.get('plan_sha256')!=agreement['plan_sha256'] or launch.get('nonce')!=expected.get('nonce') or launch.get('gpu_uuid')!=expected.get('gpu_uuid') or result.get('status')!='passed' or result.get('phase')!='completed' or result.get('error') is not None or result.get('container_id')!=expected.get('container_id') or summary.get('status')!='passed' or summary.get('job')!=job or complete.get('status')!='completed' or complete.get('job')!=job or complete.get('post_source_and_weight_check') is not True:raise ValueError('job lifecycle')
 events=lines(out/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample']; started=[x for x in events if x.get('status')=='sampling'];last=[x for x in events if x.get('type')=='summary'];dev=lines(out/'device_metadata.jsonl')
 if len(started)!=1 or len(last)!=1 or events[0]!=started[0] or events[-1]!=last[0] or not samples or last[0].get('samples')!=len(samples) or any(x.get('type')=='failure' for x in events):raise ValueError('sampler journal')
 sm=out/'sampler_monitor';s,t=terminal(sm,launch['sampler_worker'])
 if read(out/'sampler_ready.json')!={'started':started[0],'first_sample':samples[0]}:raise ValueError('sampler readiness')
 contract=module('_pixel_nvml_accept_contract_'+str(job['order']),D/'contract.py',source)
 contract.check_handshake(job,expected,ready,release,stopped,samples[0])
 if protocol.admit_release(expected,ready,started[0],samples[0])!=release or protocol.audit(expected,release,samples,t,stopped['monotonic_ns'])!=result['audit']:raise ValueError('barrier or process sample audit')
 metadata.verify(expected,dev,samples)
 marks,bounds=contract.stages(lines(out/'worker_events.jsonl'))
 if samples[0]['query_end_ns']>bounds['release'] or stages.summarize(dev,expected['host_pid'],expected['gpu_uuid'],bounds,expected['max_sample_gap_ns'])!=summary['stage_summary']:raise ValueError('stage sample summary')
 for event,key in (('model_load_end','allocator_model_load'),('request_end','allocator_request')):
  if {k:marks[event][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')}!=summary[key]:raise ValueError('allocator stage')
 def prompt_bytes(ids):return torch.tensor([ids],dtype=torch.int64).contiguous().view(torch.uint8).numpy().tobytes()
 check=contract.verify_identity(job,identity,raw,reference,prompt_bytes)
 inspect=json.loads(subprocess.run(['docker','inspect',expected['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=inspect['State']
 if inspect.get('Id')!=expected['container_id'] or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False or stopped.get('inspect_sha256')!=sha(out/'stopped_inspect.json'):raise ValueError('retained Docker state')
 if guard is not None:
  saved=read(out/'running_inspect.json');worker=[str(E/'worker.py'),'--job-index',str(job['order']),'--out',str(out),'--nonce',expected['nonce']];running=guard.validate_running(saved[0],launch['docker_create'],image,'eadp-memory-pixel-'+expected['nonce'],worker)
  if running['container_id']!=expected['container_id'] or running['host_pid']!=expected['host_pid'] or (inspect.get('Config',{}).get('Labels') or {}).get('eadp.memory.nonce')!=expected['nonce']:raise ValueError('container profile/identity')
 files=[out/x for x in ('running_inspect.json','launch.json','expected.json','worker_ready.json','release.json','stopped.json','stopped_inspect.json','controller_result.json','memory_summary.json','identity.json','raw.json','worker_complete.json','worker_events.jsonl','samples.jsonl','device_metadata.jsonl','sampler_ready.json')]+[sm/'started.json',sm/'terminal.json']
 return {'job':job,'container_id':expected['container_id'],'host_pid':expected['host_pid'],'gpu_uuid':expected['gpu_uuid'],'identity':check,'sample_count':len(samples),'stage_summary':summary['stage_summary'],'allocator_model_load':summary['allocator_model_load'],'allocator_request':summary['allocator_request'],'files_sha256':{str(x.resolve()):sha(x) for x in files}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=(D/'verification001.json').resolve():raise ValueError('unplanned proof path')
 plan=read(D/'verification_plan001.json');gpu=E/'gpu_plan001.json'
 if plan.get('gpu_plan_sha256')!=sha(gpu) or plan.get('GPU_used') is not False or plan.get('GT_opened') is not False:raise ValueError('formal pre-run plan')
 source=plan['source_sha256']
 if not source or any(sha(k)!=v for k,v in source.items()):raise ValueError('formal source changed')
 gate=module('_pixel_nvml_accept_gate',E/'gate.py',source);agreement=gate.admit_gpu();outer=E/'monitor_run001';terminal(outer,[PY,str(E/'launch.py'),'--run-id','001'])
 run=E/'run_001';journal=lines(run/'journal.jsonl');jobs=agreement['jobs']
 inventory=module('_pixel_nvml_accept_inventory',D/'contract.py',source);inventory.check_journal(journal,jobs,read(run/'final.json'))
 refs=module('_pixel_nvml_accept_refs',REF/'reference.py',source);references,_=refs.load_refs();stages=module('_pixel_nvml_accept_stages',H/'stages.py',source);metadata=module('_pixel_nvml_accept_metadata',R/'verify_metadata.py',source);protocol=module('_pixel_nvml_accept_protocol',R/'contract.py',source);guard=module('_pixel_nvml_accept_guard',R/'guard.py',source);author=read(B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 import torch
 results=[]
 for i,job in enumerate(jobs):
  terminal(run/('monitor_job_%03d'%i),[PY,str(E/'controller_entry.py'),'--job-index',str(i)]);item=verify_job(job,run/('job_%03d'%i),references[(job['task'],str(job['qid']),job['arm'])],source,agreement,stages,metadata,protocol,torch,guard,author['container_image'])
  if journal[2*i+1].get('result')!=read(run/('job_%03d'%i)/'memory_summary.json'):raise ValueError('job journal summary')
  results.append(item)
 if len({x['container_id'] for x in results})!=9 or len({(x['host_pid'],read(run/('job_%03d'%i)/'expected.json')['start_ticks']) for i,x in enumerate(results)})!=9:raise ValueError('jobs are not isolated')
 value={'status':'passed','jobs':9,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'gpu_plan_sha256':sha(gpu),'formal_plan_sha256':sha(D/'verification_plan001.json'),'run_started_sha256':sha(outer/'started.json'),'run_terminal_sha256':sha(outer/'terminal.json'),'journal_sha256':sha(run/'journal.jsonl'),'verifier_sha256':sha(__file__),'results':results,'errors':[],'limitations':['sampled MiB peaks may miss transient peaks; stages use overlapping query windows','target process, device and allocator memory differ; graphics background and unknown values remain explicit','processor and selected-index association with live input relies on online checks; high-dimensional intermediates are not all archived','capture and synchronization may perturb memory; no latency claim','three post-hoc DEV anchors and dynamic selected counts are not equal compute','endpoint file stats cannot exclude transient replacement']}
 with a.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
