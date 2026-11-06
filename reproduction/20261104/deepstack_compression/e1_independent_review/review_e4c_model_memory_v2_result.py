"""Independently reconstruct sampled one-model memory windows; no GT."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_model_pilot_v2';R=D/'run_001';A=B/'e4_c_memory_model_accept_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def supervised(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script:raise ValueError('supervision failed '+str(mon))
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out):raise ValueError('supervisor output mismatch')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=I/'e4c_model_memory_v2_001_review.json'
 if a.out.resolve()!=out:raise ValueError('output path')
 fs=A/'monitor_verify001';formal_path=A/'verification001.json';supervised(fs,A/'verify.py',formal_path);formal=read(formal_path)
 if formal.get('status')!='passed' or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False or formal.get('verifier_sha256')!=sha(A/'verify.py'):raise ValueError('formal proof')
 for name,h in formal['files_sha256'].items():
  if sha(name)!=h:raise ValueError('formal evidence changed '+name)
 run=supervised(D/'monitor_run001',D/'controller.py');
 if run['command'][-2:]!=['--run-id','001'] or run['timeout_seconds']!=1680 or run['grace_seconds']!=90:raise ValueError('run supervision')
 plan=read(D/'plan001.json');pre=read(D/'independent_final_review.json')
 for name,h in plan['source_sha256'].items():
  if sha(name)!=h:raise ValueError('run source changed '+name)
 if pre.get('status')!='passed' or pre.get('independent_review') is not True or pre.get('source_sha256')!=plan['source_sha256']:raise ValueError('GPU prelaunch source')
 supervised(I/'monitor_e4c_model_memory_v2_final',I/'review_e4c_model_memory_v2_final.py',D/'independent_final_review.json')
 if pre.get('reviewer_sha256')!=sha(I/'review_e4c_model_memory_v2_final.py'):raise ValueError('prelaunch reviewer')
 launch=read(R/'launch.json');exit=read(R/'exit.json');expected=read(R/'expected.json');ready=read(R/'worker_ready.json');release=read(R/'release.json');stopped=read(R/'stopped.json');summary=read(R/'memory_summary.json')
 if launch['approval']['review_sha256']!=sha(D/'independent_final_review.json') or exit.get('status')!='passed' or exit.get('failure') is not None or exit.get('memory_summary_sha256')!=sha(R/'memory_summary.json'):raise ValueError('run binding')
 keys=('container_id','nonce','mode')
 if ready.get('namespace_pid')!=1 or ready.get('cuda_initialized') is not False or release.get('status')!='release' or any(ready.get(k)!=expected[k] or release.get(k)!=expected[k] for k in keys):raise ValueError('pre-CUDA handshake')
 if any(stopped.get(k)!=expected[k] for k in ('container_id','nonce','host_pid','start_ticks','boot_id')) or stopped.get('inspect_sha256')!=sha(R/'stopped_inspect.json'):raise ValueError('stopped identity')
 container=json.loads(subprocess.check_output(['docker','inspect',expected['container_id']],text=True))[0];s=container['State'];cmd=launch['docker_create'];image=cmd.index(container['Image']);cfg=container['Config'];host=container['HostConfig']
 if container['Id']!=expected['container_id'] or s['Status']!='exited' or s['Pid']!=0 or s['ExitCode']!=0 or s['OOMKilled'] or cfg['Entrypoint']!=[cmd[image-1]] or cfg['Cmd']!=cmd[image+1:] or host['NetworkMode']!='none':raise ValueError('actual container identity')
 sample_events=lines(R/'samples.jsonl');samples=[x for x in sample_events if x.get('type')=='sample'];meta=lines(R/'device_metadata.jsonl');sampler=supervised(R/'sampler_monitor',D/'sampler.py')
 if sampler['command']!=launch['sampler_worker'] or len(samples)!=46 or len(meta)!=len(samples) or read(R/'sampler_complete.json')['sample_count']!=len(samples):raise ValueError('sampler evidence')
 prev=None;pid=expected['host_pid'];uuid=expected['gpu_uuid'];presence=[]
 for v,m in zip(samples,meta):
  lo,hi=v['query_start_ns'],v['query_end_ns'];ml,mh=m['query_start_ns'],m['query_end_ns']
  if not lo<=ml<=mh<=hi or hi-lo>5_000_000_000 or (prev is not None and (lo<prev[1] or hi-prev[0]>5_000_000_000)):raise ValueError('query time/gap')
  if any(v.get(k)!=expected[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')) or m['gpu_uuid']!=uuid or m['status']!='completed':raise ValueError('sample PID/UUID identity')
  matches=[z for z in m['processes'] if z['pid']==pid and z['type'] in ('C','C+G')]
  if len(matches)>1 or (matches and matches[0]['used_bytes']!=v['used_bytes']) or (not matches and v['used_bytes'] is not None) or any(z['type'] in ('C','C+G') and z['pid']!=pid for z in m['processes']):raise ValueError('target/concurrent compute mismatch')
  if any(x is not None and (type(x) is not int or x<0 or x%1048576) for x in [v['used_bytes'],*m['device_memory_bytes'].values()]):raise ValueError('invalid NVML memory values')
  presence.append(bool(matches));prev=(lo,hi)
 if samples[0]['used_bytes'] is not None or samples[0]['query_end_ns']!=release['first_sample_ns'] or samples[-1]['query_start_ns']<stopped['monotonic_ns']:raise ValueError('sample coverage before release / after exit')
 if presence.count(True)!=44 or presence[0] or presence[-1] or not all(presence[1:-1]):raise ValueError('target presence sequence')
 events=lines(R/'worker_events.jsonl');names=['barrier_released','model_load_start','model_load_end','request_start','generate_returned','request_end','teardown_end']
 if [e.get('event') for e in events]!=names:raise ValueError('stage sequence')
 by={e['event']:e for e in events};bounds={'release_ns':by['barrier_released']['monotonic_ns'],**{name+'_ns':by[name]['monotonic_ns'] for name in ('model_load_start','model_load_end','request_start','request_end','teardown_end')}}
 if list(bounds.values())!=sorted(set(bounds.values())) or bounds!=summary['stage_bounds_monotonic_ns'] or bounds!=formal['stage_bounds_monotonic_ns']:raise ValueError('stage boundaries')
 returned=by['generate_returned'];complete=read(R/'worker_complete.json');ident=read(R/'identity.json')
 if returned['task']!='DocVQA_VAL' or returned['qid']!='63709' or not returned['output_ids'] or complete.get('status')!='completed' or ident.get('status')!='passed':raise ValueError('online identity/returned output')
 import torch
 archive=R/'processor_inputs.pt'
 if ident.get('processor_archive')!={'path':str(archive.resolve()),'sha256':sha(archive)}:raise ValueError('processor archive identity')
 tensors=torch.load(archive,map_location='cpu',weights_only=True);keys=('input_ids','attention_mask','pixel_values','image_grid_thw')
 if set(tensors)!=set(keys):raise ValueError('archived tensor inventory')
 metas={}
 for name in keys:
  tensor=tensors[name].detach().contiguous().cpu();metas[name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
 refpath=B/'e4_c_timing/light_smoke_003/records.jsonl';reference_proof=read(B/'e4_c_timing/light_smoke_003_verification.json')
 if reference_proof['evidence_sha256'].get(str(refpath.resolve()))!=sha(refpath):raise ValueError('reference journal identity')
 refs=[x['record'] for x in lines(refpath) if (x['job'].get('task'),str(x['job'].get('qid')),x['job'].get('path'),x['job'].get('forced'))==('DocVQA_VAL','63709','C-ON',False)]
 if len(refs)!=1 or metas!=ident['processor'] or metas!=formal['processor'] or metas!=refs[0]['processor'] or ident['ordered_ids']!=refs[0]['ordered_ids'] or ident['ordered_ids']!=formal['ordered_ids'] or any(returned[k]!=refs[0][k] for k in ('output_ids','generation_scalars','eos_ids')):raise ValueError('archived input/ordered selection/reference mismatch')
 windows={}
 for name,(lo,hi) in {'pre_cuda':(None,bounds['release_ns']),'model_load':(bounds['model_load_start_ns'],bounds['model_load_end_ns']),'request':(bounds['request_start_ns'],bounds['request_end_ns']),'teardown':(bounds['request_end_ns'],bounds['teardown_end_ns'])}.items():
  chosen=[(v,m) for v,m in zip(samples,meta) if (lo is None or v['query_start_ns']>=lo) and v['query_end_ns']<=hi]
  vals=[v['used_bytes'] for v,m in chosen];dev=[m['device_memory_bytes']['used'] for v,m in chosen];known=lambda x:[v for v in x if v is not None]
  n_present=sum(any(z['pid']==pid and z['type'] in ('C','C+G') for z in m['processes']) for _,m in chosen)
  windows[name]={'fully_contained_samples':len(chosen),'process':{'known':len(known(vals)),'unknown':len(vals)-len(known(vals)),'max_known_bytes':max(known(vals),default=None),'present':n_present,'absent':len(vals)-n_present,'present_unknown':sum(v['used_bytes'] is None and any(z['pid']==pid and z['type'] in ('C','C+G') for z in m['processes']) for v,m in chosen)},'device_used':{'known':len(known(dev)),'unknown':len(dev)-len(known(dev)),'max_known_bytes':max(known(dev),default=None)}}
 if windows!=formal['nvml_stages'] or windows!=summary['nvml_stages']['stages']:raise ValueError('independent stage recomputation mismatch')
 for event,key in (('model_load_end','allocator_model_load'),('request_end','allocator_request')):
  pair={name:by[event][name] for name in ('allocated_peak_bytes','reserved_peak_bytes')}
  if pair!=formal[key] or pair!=summary[key] or pair['allocated_peak_bytes']>pair['reserved_peak_bytes']:raise ValueError('allocator evidence')
 result={'status':'passed','independent_review':True,'scope':'one EADP K256 C-ON model process sampled memory; post-hoc v2 repeat','formal_sha256':sha(formal_path),'formal_started_sha256':sha(fs/'started.json'),'formal_terminal_sha256':sha(fs/'terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'sample_count':len(samples),'processor_archive_sha256':sha(archive),'processor':metas,'ordered_ids':ident['ordered_ids'],'independently_recomputed_windows':windows,'allocator_model_load':formal['allocator_model_load'],'allocator_request':formal['allocator_request'],'GT_opened':False,'scoring_performed':False,'predictions_opened_for_identity':True,'limitations':['post-hoc repeat on one image after observing v1; no latency or generalization claim','NVML process and whole-device usage differ; graphics background present; sampled peaks may miss transient maxima, rounded to MiB','process absent in pre-CUDA sample is unknown, not measured zero','archived processor bytes and ordered indices independently match reference; their link to actual model input relies on runtime byte comparison','CPU snapshots and synchronization during request may perturb memory; P3 and three-DS behavior relies on accepted same-source identity diagnostic'],'errors':[]}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
