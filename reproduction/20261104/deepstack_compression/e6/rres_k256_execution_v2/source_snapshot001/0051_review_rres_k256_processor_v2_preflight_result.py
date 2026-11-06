"""Independent review of actual three-input CPU-only R-res preflight."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/rres_k256_pilot_v2';I=B/'e1_independent_review';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if (len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor')
 return s
def position_digest(n,start):
 suffix=n-start-256
 if start<0 or suffix<0:raise ValueError('sequence length')
 prefix=list(range(start));after=list(range(start+16,start+16+suffix))
 axes=[prefix+[start]*256+after,prefix+[start+r for r in range(16) for _ in range(16)]+after,prefix+[start+c for _ in range(16) for c in range(16)]+after]
 values=sum(axes,[]);return hashlib.sha256(struct.pack('<%dq'%len(values),*values)).hexdigest(),max(values)+1-n
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'rres_k256_processor_v2_preflight_result_review.json').resolve():raise ValueError('output')
 plan=read(D/'preflight_plan001.json');source=plan['source_sha256'];qpath=I/'rres_k256_processor_v2_preflight_review.json';reviewer=I/'review_rres_k256_processor_v2_preflight.py';terminal(I/'monitor_rres_k256_processor_v2_preflight',reviewer,qpath);q=read(qpath)
 if any(sha(k)!=v for k,v in source.items()) or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('plan_sha256')!=sha(D/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('frozen source or CPU review')
 s=terminal(D/'monitor_preflight001',D/'preflight_launch.py',D/'run_preflight001');launch=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json');ident=read(D/'run_preflight001/container_identity.json');worker=read(D/'run_preflight001/worker_preflight.json');probe=read(D/'run_preflight001/processor_probe.json')
 if (launch.get('approval')!={'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(qpath)} or launch.get('controller',{}).get('pid')!=s['worker_pid'] or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('container_stopped_verified') is not True or ex.get('failure_type') is not None):raise ValueError('exit')
 cid=ex['container_identity']['container_id']
 if ident!=ex['container_identity'] or ex['last_inspect']['id']!=cid or cid not in ident['host_cgroup'] or worker.get('container_id')!=cid:raise ValueError('host container identity')
 author_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author=read(author_path);path=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';spec=importlib.util.spec_from_file_location('_independent_rres_author',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);runtime=m.load_author(source)
 cmd=runtime.container_command(author,[str(D/'preflight_worker.py'),'--out',str(D/'run_preflight001/worker_preflight.json')],False,launch['name'])
 if cmd.count('--rm')!=1:raise ValueError('fixed command')
 cmd.remove('--rm')
 if cmd!=launch['command']:raise ValueError('launcher command')
 obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=author['container_image']
 if (obj['Id']!=cid or obj['Name']!='/'+launch['name'] or obj['Image']!=image or cfg['Image']!=image or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or cfg['Cmd']!=cmd[cmd.index(image)+1:] or host['NetworkMode']!=cmd[cmd.index('--network')+1] or host.get('PidMode') not in ('',None) or host.get('Devices') not in (None,[]) or host.get('DeviceRequests') not in (None,[])):raise ValueError('Docker actual identity')
 for value in (cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'):
  if [x for x in cfg['Env'] if x.startswith(value.split('=',1)[0]+'=')]!=[value]:raise ValueError('env')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   parts=cmd[i+1].split(',');v=dict(z.split('=',1) for z in parts if '=' in z)
   if v.get('type')!='bind':raise ValueError('mount type')
   mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in parts))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('mounts')
 if (worker.get('status')!='passed' or worker.get('approval')!=launch['approval'] or worker.get('processor_probe_sha256')!=sha(D/'run_preflight001/processor_probe.json') or worker.get('images')!=3 or worker.get('visual_tokens')!=256 or worker.get('GT_opened') is not False or worker.get('model_loaded') is not False or worker.get('cuda_initialized') is not False or worker.get('GPU_authorized') is not False or worker.get('errors')!=[]):raise ValueError('worker')
 if (probe.get('status')!='passed' or probe.get('images')!=3 or probe.get('resolution')!=512 or probe.get('original_resolution')!=1024 or probe.get('GT_opened') is not False or probe.get('model_loaded') is not False or probe.get('cuda_initialized') is not False or probe.get('errors')!=[]):raise ValueError('processor proof')
 for field,path in {'e1_formal_sha256':B/'e1/e1full001_completion.json','e1_independent_review_sha256':I/'e1full001_review.json','e1_independent_started_sha256':I/'monitor_e1full001/started.json','e1_independent_terminal_sha256':I/'monitor_e1full001/terminal.json','design_sha256':B/'e6/pixel_wavefront_t020_nvml_nine/design001.json'}.items():
  if probe.get(field)!=sha(path):raise ValueError('processor reference')
 if any(sha(k)!=v for k,v in probe['source_sha256'].items()):raise ValueError('processor source')
 jobs=read(B/'e6/pixel_wavefront_t020_nvml_nine/design001.json')['jobs'];expected={(j['task'],str(j['qid'])):j for j in jobs};seen=set()
 if len(expected)!=3:raise ValueError('anchor inventory')
 for item in probe['anchors']:
  key=(item['task'],str(item['qid']));ref=expected[key];seen.add(key)
  if any(item.get(k)!=ref.get(k) for k in ('group_id','image_sha256','message_sha256')) or item.get('original_visual_tokens')!=1024 or item.get('low_visual_tokens')!=256 or item.get('grid')!=[[1,32,32]] or item.get('text_prefix_suffix_identical') is not True or item.get('position_formula_verified') is not True:raise ValueError('anchor identity or online processor')
  inp=item['low_processor']['input_ids'];n=inp['shape'][1];position=item['position_ids'];expected_hash,delta=position_digest(n,item['image_start'])
  if inp['dtype']!='torch.int64' or position['dtype']!='torch.int64' or position['shape']!=[3,1,n] or position['sha256']!=expected_hash or item['rope_delta']!=[[delta]] or item['low_processor']['pixel_values']['shape']!=[1024,1536]:raise ValueError('position reconstruction or grid')
 if seen!=set(expected) or len(probe['anchors'])!=3:raise ValueError('anchor denominator')
 manifest=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];stats=worker['weights_stats']
 if len(manifest)!=4 or set(stats)!={x['path'] for x in manifest}:raise ValueError('weight count')
 for item in manifest:
  st=Path(item['path']).stat()
  if stats[item['path']]!=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] or st.st_size!=item['bytes']:raise ValueError('weight stat')
 fields={'worker_sha256':D/'run_preflight001/worker_preflight.json','reviewer_sha256':Path(__file__),'plan_sha256':D/'preflight_plan001.json','launch_sha256':D/'run_preflight001/launch.json','identity_sha256':D/'run_preflight001/container_identity.json','exit_sha256':D/'run_preflight001/exit.json','started_sha256':D/'monitor_preflight001/started.json','terminal_sha256':D/'monitor_preflight001/terminal.json','processor_probe_sha256':D/'run_preflight001/processor_probe.json'}
 result={'status':'passed','independent_review':True,'images':3,'GT_opened':False,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'errors':[],**{k:sha(v) for k,v in fields.items()},'container_id':cid,'bind_mounts':len(mounts),'limitations':['raw processor tensors not archived; original/low prompt equality is online evidence','independently recomputed 3-axis position fingerprints from recorded shape and start','weight hashes checked by worker online; reviewer checked four endpoint stats, not payload','existing DEV mixed journal opened for input identity; no GPU model, DS or generation tested']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
