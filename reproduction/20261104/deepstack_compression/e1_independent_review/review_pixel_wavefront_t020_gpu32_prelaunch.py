"""Independent GPU-prelaunch review of post-hoc 32-image positive threshold."""
import argparse,hashlib,json,struct,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_wavefront_t020_gpu32';I=B/'e1_independent_review';PIXEL=B/'e6/external_sources/pixelprune_8d347eb'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script:raise ValueError('supervisor failure')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out):raise ValueError('output path')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=D/'independent_prelaunch_review.json'
 if a.out.resolve()!=out:raise ValueError('wrong output')
 plan=read(D/'plan001.json');sources=plan['source_sha256']
 if plan['threshold']!=.2 or len(plan['anchors'])!=32 or len(sources)!=55 or plan['GT_opened'] is not False or plan['new_predictions_opened'] is not False or any(sha(k)!=v for k,v in sources.items()):raise ValueError('plan/source mismatch')
 low=I/'pixel_low_threshold_cpu_review.json';ls=I/'monitor_pixel_low_threshold_cpu/started.json';lt=I/'monitor_pixel_low_threshold_cpu/terminal.json';reviewer=I/'review_pixel_low_threshold_cpu.py';supervised(ls.parent,reviewer,low);r=read(low)
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('reviewer_sha256')!=sha(reviewer) or r.get('ordered_matches')!=96 or r.get('case_count')!=96 or r.get('GT_opened') is not False or r.get('predictions_opened') is not False or r.get('errors')!=[] or r.get('result_sha256')!=sha(B/'e6/pixel_low_threshold_cpu/result001.json') or r.get('plan_sha256')!=sha(B/'e6/pixel_low_threshold_cpu/plan001.json'):raise ValueError('low-threshold CPU review')
 rows=[x for x in r['cases'] if x['threshold']==.2];pre=read(D/'preflight001.json')
 if len(rows)!=32 or len(pre['cases'])!=32 or pre.get('status')!='passed' or pre.get('plan_sha256')!=sha(D/'plan001.json') or pre.get('cuda_initialized') is not False or pre.get('model_loaded') is not False or pre.get('gpu_used') is not False or pre.get('GT_opened') is not False:raise ValueError('preflight incomplete')
 for i,(anchor,row,case) in enumerate(zip(plan['anchors'],rows,pre['cases'])):
  ids=anchor['expected_indices'];digest=hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()
  if row['index']!=i or row['identity']['task']!=anchor['task'] or str(row['identity']['qid'])!=str(anchor['qid']) or row['identity']['group_id']!=anchor['group_id'] or row['ordered_indices_sha256']!=digest or row['selected_count']!=len(ids) or row['actual_author_anchored_dispatch'] is not True or case['task']!=anchor['task'] or str(case['qid'])!=str(anchor['qid']) or case['k']!=len(ids) or case['capture_sha256']!=sha(anchor['capture_path']):raise ValueError('anchor/ordered selection')
 sys.path.insert(0,str(PIXEL));import torch
 from pixelprune.methods import pred_2d
 if torch.cuda.is_initialized() or Path(pred_2d.__file__).resolve()!=PIXEL/'pixelprune/methods/pred_2d.py' or time.get_clock_info('perf_counter').implementation!='clock_gettime(CLOCK_MONOTONIC)':raise ValueError('CPU source or monotonic clock')
 selector=pred_2d.Pred2DSelector(method='max',threshold=.2,anchored=True);called=[];original=selector._select_2d_loco_anchored
 def anchored(*x,**kw):called.append(True);return original(*x,**kw)
 def fast(*_x,**_kw):raise ValueError('unexpected fast dispatch')
 selector._select_2d_loco_anchored=anchored;selector._select_2d_loco_fast=fast;v=torch.zeros((9,4));dispatch=selector._select_2d_loco(v,3,3,v.device)
 if called!=[True] or dispatch.tolist()!=[0]:raise ValueError('author dispatch is not anchored')
 test=D/'monitor_cpu_test001';supervised(test,D/'test_worker.py')
 if 'Ran 3 tests' not in (test/'stderr.log').read_text():raise ValueError('CPU tests')
 mon=D/'monitor_preflight001';s=supervised(mon,D/'launch.py');
 if s['command'][-2:]!=['--stage','preflight'] or s['timeout_seconds']!=720 or s['grace_seconds']!=90:raise ValueError('preflight supervisor')
 launch=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json');ident=read(D/'run_preflight001/container_identity.json')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ident!=ex['container_identity'] or ident['container_id'] not in ident['host_cgroup']:raise ValueError('preflight container evidence')
 obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];cmd=launch['command'];idx=cmd.index(ident['image'])
 if obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Status']!='exited' or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or cfg['Entrypoint']!=[cmd[idx-1]] or cfg['Cmd']!=cmd[idx+1:] or cfg['User']!='1000:1000' or host['NetworkMode']!='none' or host['Devices'] or host['DeviceRequests']:raise ValueError('actual container identity')
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 if env.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('preflight GPU visible')
 mounts=[]
 for i,x in enumerate(cmd):
  if x=='--mount':
   parts=cmd[i+1].split(',');f=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((f['src'],f['dst'],'readonly' not in parts))
 if sorted(mounts)!=sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts']):raise ValueError('container mounts')
 proof={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(__file__),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(mon/'started.json'),'preflight_terminal_sha256':sha(mon/'terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'cpu_started_sha256':sha(test/'started.json'),'cpu_terminal_sha256':sha(test/'terminal.json'),'cpu_stderr_sha256':sha(test/'stderr.log'),'GT_opened':False,'errors':[],'limitations':['same 32 post-hoc DEV processor captures, not independent new sample','one synchronized host-wall measurement per arm and image; records both raw monotonic endpoints','component-only, no GPU/model identity or end-to-end performance claim','CPU author selector dispatch at positive threshold is anchored; full model and other inputs not established']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
