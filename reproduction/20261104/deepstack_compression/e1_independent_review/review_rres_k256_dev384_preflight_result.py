"""Independent complete no-device 384-input preflight audit. No model, GT or new predictions."""
import argparse,collections,hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_dev384';P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2];AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b');R=D/'run_preflight001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 assert type(s['worker_pid']) is int and s['worker_pid']>0 and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(d)
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s

def meta(torch,t):
 x=t.detach().contiguous().cpu();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();assert not args.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 planfile=D/'preflight_plan001.json';plan=read(planfile);pre=I/'rres_k256_dev384_cpu_preflight_review.json';precode=I/'review_rres_k256_dev384_cpu_preflight.py';terminal(I/'monitor_rres_k256_dev384_cpu_preflight',precode,pre);q=read(pre);s=terminal(D/'monitor_preflight001',D/'preflight_launch.py',R)
 assert q['status']=='passed' and q['independent_review'] and q['preflight_only'] and q['GPU_authorized'] is False and q['GT_opened'] is False and q['errors']==[] and q['plan_sha256']==sha(planfile) and q['reviewer_sha256']==sha(precode) and q['source_sha256']==plan['source_sha256']
 assert plan['scope']=='rres384_noGPU_processor_preflight' and plan['images']==384 and plan['GPU_authorized'] is False and all(sha(k)==v for k,v in plan['source_sha256'].items())
 launch=read(R/'launch.json');identity=read(R/'container_identity.json');ex=read(R/'exit.json');worker=read(R/'worker_preflight.json');probe_path=R/'probe/processor_probe.json'
 assert launch['approval']=={'plan_sha256':sha(planfile),'review_sha256':sha(pre)} and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['container_identity']==identity and ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
 assert worker['status']=='passed' and worker['approval']==launch['approval'] and worker['container_id']==identity['container_id'] and worker['images']==384 and worker['visual_tokens']==256 and worker['GPU_authorized'] is False and worker['model_loaded'] is False and worker['cuda_initialized'] is False and worker['GT_opened'] is False and worker['existing_DEV_predictions_opened'] is True and worker['errors']==[] and worker['processor_probe_sha256']==sha(probe_path)
 weights=worker['weights'];prior=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];assert len(weights)==len(prior)==4
 for item,expected in zip(weights,prior):
  stat=Path(item['path']).stat();assert item['path']==expected['path'] and item['sha256']==expected['expected_lfs_sha256'] and item['stat']==[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns] and stat.st_size==expected['bytes']
 obj=json.loads(subprocess.run(['docker','inspect',identity['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];assert obj['Id']==identity['container_id']==ex['last_inspect']['id'] and obj['Image']==identity['image'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['HostConfig']['NetworkMode']=='none' and not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests']
 cmd=launch['command'];assert '--gpus' not in cmd and '--device' not in cmd and obj['Config']['Cmd']==[str(D/'preflight_worker.py'),'--out',str(R/'worker_preflight.json')] and obj['Config']['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and obj['Config']['User']==cmd[cmd.index('--user')+1] and obj['Config']['WorkingDir']==cmd[cmd.index('--workdir')+1]
 mounts=[]
 for i,c in enumerate(cmd[:-1]):
  if c=='--env':assert cmd[i+1] in obj['Config']['Env']
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 assert sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
 saved=read(probe_path);anchors=read(D/'design001.json')['anchors'];records=saved['anchors'];keys=[(x['task'],str(x['qid'])) for x in anchors]
 assert saved['status']=='passed' and saved['images']==len(records)==384 and saved['GT_opened'] is False and saved['model_loaded'] is False and saved['cuda_initialized'] is False and saved['errors']==[] and len(set(keys))==384 and collections.Counter(x[0] for x in keys)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 events=[json.loads(x) for x in (R/'probe/processor_progress.jsonl').read_text().splitlines()];assert [(x['task'],str(x['qid'])) for x in events if x['event']=='started']==keys==[(x['task'],str(x['qid'])) for x in events if x['event']=='verified']
 sys.path[:0]=[str(P),str(B/'e4_c_latency'),str(AUTHOR),str(AUTHOR/'VLMEvalKit')]
 import torch,vlmeval.vlm.qwen3_vl.model_fixed_res as mod
 from resolution import fixed_resolution
 from input_preflight import load_accepted_e1_references
 from transformers import AutoProcessor,AutoConfig
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 assert Path(inspect.getfile(mod.Qwen3VLChatFixedRes)).resolve()==AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py' and not torch.cuda.is_initialized()
 rows,refs=load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json');rows={(r['task'],str(r['qid'])):r for r in rows};assert len(rows)==len(refs)==384 and set(rows)==set(keys)==set(refs)
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);wrapper=object.__new__(mod.Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 for anchor,record in zip(anchors,records):
  key=(anchor['task'],str(anchor['qid']));row=rows[key];ref=refs[key];assert all(str(row.get(k))==str(anchor[k])==str(record[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and sha(row['image_path'])==anchor['image_sha256'] and ref['processor']==anchor['processor'] and ref['image_start']==anchor['image_start']
  messages=wrapper._build_messages(row['message'],dataset=row['task']);original=wrapper._processor_inputs(messages)
  with fixed_resolution(mod,512):low=wrapper._processor_inputs(messages)
  names=('input_ids','attention_mask','pixel_values','image_grid_thw');assert {k:meta(torch,original[k]) for k in names}==ref['processor'] and {k:meta(torch,low[k]) for k in names}==record['low_processor'] and low['image_grid_thw'].tolist()==record['grid']==[[1,32,32]] and config.vision_config.spatial_merge_size==2 and torch.all(low['attention_mask']==1)
  old=original['input_ids'][0];new=low['input_ids'][0];oi=torch.where(old==config.image_token_id)[0].tolist();ni=torch.where(new==config.image_token_id)[0].tolist();start=anchor['image_start'];assert oi==list(range(start,start+1024)) and ni==list(range(start,start+256)) and old[:start].tolist()==new[:start].tolist() and old[start+1024:].tolist()==new[start+256:].tolist() and record['image_start']==start and record['original_visual_tokens']==1024 and record['low_visual_tokens']==256 and record['text_prefix_suffix_identical'] is True
  pos,delta=Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),input_ids=low['input_ids'],image_grid_thw=low['image_grid_thw'],attention_mask=low['attention_mask']);L=int(new.numel());xyz=[[i for i in range(start)] for _ in range(3)]
  for y in range(16):
   for x in range(16):xyz[0].append(start);xyz[1].append(start+y);xyz[2].append(start+x)
  for j in range(L-start-256):
   for axis in range(3):xyz[axis].append(start+16+j)
  assert pos.detach().cpu().tolist()==[[axis] for axis in xyz] and delta.tolist()==record['rope_delta'] and meta(torch,pos)==record['position_ids'] and record['position_formula_verified'] is True and int(delta.item())==max(max(axis) for axis in xyz)+1-L
  assert mod.QWEN3_FIXED_RESOLUTION==1024
 assert not torch.cuda.is_initialized() and all(sha(k)==v for k,v in plan['source_sha256'].items())
 out={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'images':384,'GT_opened':False,'existing_DEV_predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'scoring_performed':False,'plan_sha256':sha(planfile),'worker_preflight_sha256':sha(R/'worker_preflight.json'),'processor_probe_sha256':sha(probe_path),'processor_progress_sha256':sha(R/'probe/processor_progress.jsonl'),'launch_sha256':sha(R/'launch.json'),'identity_sha256':sha(R/'container_identity.json'),'exit_sha256':sha(R/'exit.json'),'started_sha256':sha(D/'monitor_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'reviewer_sha256':sha(__file__),'prior_review_sha256':sha(pre),'recomputed_processor_images':384,'recomputed_mrope_images':384,'weight_payload_rehashed_by_review':False,'scope':'independent host CPU rerun of all 384 processor tensors and mRoPE; no GPU authority','limitations':['independently re-ran processors and mRoPE on CPU, not a model/GPU generation','four weight payload hashes were produced by the preflight; independent review checked reports and current stat, not payload again','accepted E1 DEV mixed predictions were parsed only for input identity'],'errors':[]}
 with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('independent 384 processor/mRoPE preflight passed')
if __name__=='__main__':main()
