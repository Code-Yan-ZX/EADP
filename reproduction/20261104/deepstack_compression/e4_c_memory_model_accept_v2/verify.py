"""Independent no-GT acceptance of one complete sampled model-memory lifecycle.

The archive permits independent processor and ordered-index checks. The link
between the archive and live inputs remains a runtime byte comparison.
"""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_c_memory_model_pilot_v2';sys.path[:0]=[str(P),str(B/'e4_c_latency')]
import gate
if Path(gate.__file__).resolve()!=P/'gate.py':raise ValueError('wrong gate')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path.resolve():raise ValueError('wrong module')
 return m

def verify_identity(run,identity,returned,reference):
 import torch
 archive=run/'processor_inputs.pt'
 if identity.get('processor_archive')!={'path':str(archive.resolve()),'sha256':sha(archive)}:raise ValueError('processor archive mismatch')
 inputs=torch.load(archive,map_location='cpu',weights_only=True)
 keys=('input_ids','attention_mask','pixel_values','image_grid_thw')
 if set(inputs)!=set(keys):raise ValueError('processor input inventory')
 meta={}
 for key in keys:
  t=inputs[key]
  if not isinstance(t,torch.Tensor):raise ValueError('processor input is not tensor')
  v=t.detach().contiguous().cpu()
  meta[key]={'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()}
 for key in ('task','qid','group_id','image_sha256','message_sha256','ordered_ids'):
  if identity.get(key)!=reference.get(key):raise ValueError('reference identity mismatch: '+key)
 if meta!=identity.get('processor') or meta!=reference.get('processor'):raise ValueError('processor values mismatch')
 for key in ('output_ids','generation_scalars','eos_ids'):
  if returned.get(key)!=reference.get(key):raise ValueError('generation identity mismatch: '+key)
 return meta

def independent_windows(samples,metadata,bounds,max_gap):
 fields=('release_ns','model_load_start_ns','model_load_end_ns','request_start_ns','request_end_ns','teardown_end_ns');values=[bounds.get(k) for k in fields]
 if any(type(x) is not int for x in values) or any(a>=b for a,b in zip(values,values[1:])) or len(samples)!=len(metadata) or not samples:raise ValueError('invalid time windows')
 previous=None
 for s,m in zip(samples,metadata):
  lo=s['query_start_ns'];hi=s['query_end_ns']
  if not lo<=m['query_start_ns']<=m['query_end_ns']<=hi or (previous is not None and (lo<previous or lo-previous>max_gap)) or s['gpu_uuid']!=m['gpu_uuid']:raise ValueError('sample time identity or gaps')
  previous=hi
 out={}
 for name,(lo,hi) in {'pre_cuda':(None,values[0]),'model_load':(values[1],values[2]),'request':(values[3],values[4]),'teardown':(values[4],values[5])}.items():
  ps=[];ds=[];present=absent=present_unknown=0
  for s,m in zip(samples,metadata):
   if (lo is not None and s['query_start_ns']<lo) or s['query_end_ns']>hi:continue
   val=s.get('used_bytes');matches=[x for x in m['processes'] if x['pid']==s['host_pid'] and x['type'] in ('C','C+G')]
   if len(matches)>1 or (matches and matches[0]['used_bytes']!=val) or (not matches and val is not None):raise ValueError('process identity mismatch')
   present+=bool(matches);absent+=not matches;present_unknown+=bool(matches and val is None);ps.append(val);ds.append(m['device_memory_bytes']['used'])
  def counts(v):
   known=[x for x in v if x is not None]
   return {'known':len(known),'unknown':len(v)-len(known),'max_known_bytes':max(known) if known else None}
  out[name]={'fully_contained_samples':len(ps),'process':dict(counts(ps),present=present,absent=absent,present_unknown=present_unknown),'device_used':counts(ds)}
 return out

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a=a.parse_args();run=P/'run_001';ss=P/'monitor_run001/started.json';tt=P/'monitor_run001/terminal.json';s=read(ss);t=read(tt);cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name=='controller.py']!=[(P/'controller.py').resolve()] or cmd[-2:]!=['--run-id','001']:raise ValueError('controller supervision incomplete')
 approval=gate.review(preflight_only=False,host=True);launch=read(run/'launch.json');ex=read(run/'exit.json');expected=read(run/'expected.json');ready=read(run/'worker_ready.json');release=read(run/'release.json');stopped=read(run/'stopped.json');summary=read(run/'memory_summary.json');complete=read(run/'worker_complete.json');identity=read(run/'identity.json')
 if launch.get('approval')!=approval or ex.get('status')!='passed' or ex.get('failure') is not None or ex.get('memory_summary_sha256')!=sha(run/'memory_summary.json') or complete.get('status')!='completed' or complete.get('GT_opened') is not False or identity.get('status')!='passed':raise ValueError('run incomplete')
 if ready.get('cuda_initialized') is not False or ready.get('namespace_pid')!=1 or any(ready.get(k)!=expected[k] or release.get(k)!=expected[k] for k in ('nonce','container_id','mode')) or release.get('status')!='release' or type(release.get('first_sample_ns')) is not int:raise ValueError('pre-CUDA handshake mismatch')
 for k in ('container_id','nonce','host_pid','start_ticks','boot_id'):
  if stopped.get(k)!=expected[k]:raise ValueError('stopped identity mismatch')
 if stopped.get('inspect_sha256')!=sha(run/'stopped_inspect.json'):raise ValueError('stopped inspect hash mismatch')
 docker=json.loads(subprocess.run(['docker','inspect',expected['container_id']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=docker['State']
 if docker['Id']!=expected['container_id'] or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False:raise ValueError('retained container not stopped cleanly')
 sm=run/'sampler_monitor';sx=read(sm/'started.json');st=read(sm/'terminal.json')
 if st.get('reason')!='completed' or st.get('worker_returncode')!=0 or st.get('remaining_process_group')!=[] or st.get('worker_pid')!=sx.get('worker_pid') or sx.get('command')!=launch['sampler_worker']:raise ValueError('sampler terminal mismatch')
 events=lines(run/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample'];meta=lines(run/'device_metadata.jsonl');verified=module('e4c_accept_verify_metadata',B/'e4_c_memory_nvml/verify_metadata.py').verify(expected,meta,samples)
 if verified!=summary['nvml_metadata'] or not samples or samples[0]['query_end_ns']>release['first_sample_ns']:raise ValueError('NVML metadata or first sample mismatch')
 w=lines(run/'worker_events.jsonl');wanted=('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end');stages={}
 for e in w:
  if e.get('event') in wanted:
   if e['event'] in stages:raise ValueError('duplicate stage')
   stages[e['event']]=e
 if set(stages)!=set(wanted):raise ValueError('stage missing')
 returned=[e for e in w if e.get('event')=='generate_returned']
 if len(returned)!=1 or returned[0].get('task')!='DocVQA_VAL' or returned[0].get('qid')!='63709' or not returned[0].get('output_ids'):raise ValueError('actual returned generation missing')
 reference_path=B/'e4_c_timing/light_smoke_003/records.jsonl';reference_proof=read(B/'e4_c_timing/light_smoke_003_verification.json')
 if reference_proof['evidence_sha256'].get(str(reference_path.resolve()))!=sha(reference_path):raise ValueError('reference evidence changed')
 refs=[x['record'] for x in lines(reference_path) if (x['job'].get('task'),str(x['job'].get('qid')),x['job'].get('path'),x['job'].get('forced'))==('DocVQA_VAL','63709','C-ON',False)]
 if len(refs)!=1:raise ValueError('reference is not unique')
 processor=verify_identity(run,identity,returned[0],refs[0])
 bound={'release_ns':stages['barrier_released']['monotonic_ns'],**{k+'_ns':stages[k]['monotonic_ns'] for k in wanted[1:]}}
 if bound!=summary['stage_bounds_monotonic_ns'] or samples[0]['query_end_ns']>bound['release_ns']:raise ValueError('stage boundary mismatch')
 windows=independent_windows(samples,meta,bound,5_000_000_000)
 if windows!=summary['nvml_stages']['stages'] or not windows['model_load']['fully_contained_samples']:raise ValueError('window summary mismatch')
 for event,key in (('model_load_end','allocator_model_load'),('request_end','allocator_request')):
  if {x:stages[event][x] for x in ('allocated_peak_bytes','reserved_peak_bytes')}!=summary[key]:raise ValueError('allocator evidence mismatch')
 files=[ss,tt,reference_path,B/'e4_c_timing/light_smoke_003_verification.json']+[run/x for x in ('launch.json','exit.json','expected.json','worker_ready.json','release.json','stopped.json','stopped_inspect.json','samples.jsonl','device_metadata.jsonl','worker_events.jsonl','worker_complete.json','identity.json','processor_inputs.pt','memory_summary.json')]+[sm/'started.json',sm/'terminal.json'];proof={'status':'passed','stage':'one model process memory','GT_opened':False,'scoring_performed':False,'files_sha256':{str(x.resolve()):sha(x) for x in files},'sample_count':len(samples),'stage_bounds_monotonic_ns':bound,'nvml_stages':windows,'allocator_model_load':summary['allocator_model_load'],'allocator_request':summary['allocator_request'],'processor':processor,'ordered_ids':identity['ordered_ids'],'verifier_sha256':sha(__file__),'limitations':['archive checked against accepted reference; association with actual model inputs relies on online byte comparison','processor CPU snapshots and synchronization during request may perturb memory','full process and device sampled peaks differ from per-request memory and can miss transients','one image only; no latency claim','mixed E1/E4 DEV prediction journals used for input identity']}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
