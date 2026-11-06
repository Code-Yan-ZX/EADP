"""Independent prelaunch check of post-hoc PixelPrune 384-image DEV panel."""
import argparse,hashlib,json,subprocess
from pathlib import Path

B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_t045_dev384';V=B/'e6/pixelprune_t045_accept';M=B/'e1_independent_review/monitor_pixel_t045_dev384_prelaunch'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(directory):
 s,t=read(directory/'started.json'),read(directory/'terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']
 return s
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);a=parser.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json'
 plan=read(D/'plan001.json');assert sha(D/'plan001.json')=='c68742ed65f69d9598e3943f04872763fce0ce62ae7a5db178891e260debb1b0'
 assert len(plan['sha256'])==132
 for path,h in plan['sha256'].items():assert sha(path)==h,path
 assert plan['method']=='pred_2d' and plan['metric']=='max' and plan['threshold']==0.45 and plan['anchored'] is True and plan['vit_layer']==0
 assert plan['generation']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 anchors=plan['anchors'];assert len(anchors)==384 and len({(x['task'],str(x['qid'])) for x in anchors})==384 and len({x['group_id'] for x in anchors})==384
 assert all(sum(x['task']==task for x in anchors)==128 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))
 rows=[json.loads(x) for x in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];idx={(x['task'],str(x['qid'])):x for x in rows};assert len(idx)==384
 for x in anchors:
  y=idx[(x['task'],str(x['qid']))]
  assert x['group_id']==y['group_id'] and x['message_sha256']==y['message_sha256'] and x['image_sha256']==y['image_sha256']
 smoke=read(B/'e1_independent_review/pixel_t045_result_review.json');assert sha(B/'e1_independent_review/pixel_t045_result_review.json')==plan['smoke_review_sha256'] and smoke['status']=='passed' and smoke['independent_review'] is True and smoke['images']==4 and smoke['GT_opened'] is False
 assert all(x['cpu_gpu_indices_equal'] is True and x['eos'] is True for x in smoke['observed']) and sorted(x['visual_tokens'] for x in smoke['observed'])==[1,499,500,568]
 assert len(smoke['files_sha256'])==16
 for path,h in smoke['files_sha256'].items():assert sha(path)==h
 for mon in (D/'monitor_test002',B/'e1_independent_review/monitor_pixel_t045_dev384_cpu_review'):
  s=terminal(mon);assert 'Ran 13 tests' in (mon/'stderr.log').read_text() and '\nOK' in (mon/'stderr.log').read_text() and s['command'][-8:]==['-m','unittest','discover','-s','e6/pixelprune_t045_dev384','-p','test_*.py','-v']
 vp=read(V/'verification_plan001.json');assert sha(V/'verification_plan001.json')=='2bef125db712e9070ec69b5b7e0c0f46ab9f473e6e9b89cc2e36dd34b0527ad7' and vp['panel_plan_sha256']==sha(D/'plan001.json') and vp['expected_questions']==384 and vp['GT_opened'] is False
 for path,h in vp['source_sha256'].items():assert sha(path)==h
 terminal(V/'monitor_test001');assert 'Ran 6 tests' in (V/'monitor_test001/stderr.log').read_text() and '\nOK' in (V/'monitor_test001/stderr.log').read_text()
 s=terminal(D/'monitor_preflight001');pre=read(D/'preflight001.json');launch=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json')
 assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['weight_sha256_verified']==4 and pre['patch_before_model'] is True
 for x in pre['processor_verified']:
  assert x['processor']==next(y['processor'] for y in anchors if (y['task'],str(y['qid']))==(x['task'],str(x['qid'])))
 assert len(pre['processor_verified'])==3 and len({x['task'] for x in pre['processor_verified']})==3
 for x in pre['weight_stats']:
  st=Path(x['path']).stat();assert (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_dev)==(x['size'],x['mtime_ns'],x['ctime_ns'],x['inode'],x['device'])
 assert (Path(s['cwd'])/s['command'][-3]).resolve()==D/'launch.py' and s['command'][-2:]==['--stage','preflight']
 assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['failure_type'] is None and ex['timed_out'] is False
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];st=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];cmd=launch['command']
 assert obj['Id']==ex['last_inspect']['id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'] and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['User']==cmd[cmd.index('--user')+1]
 mounts=[]
 for i,v in enumerate(cmd[:-1]):
  if v=='--mount':
   parts=cmd[i+1].split(',');d=dict(p.split('=',1) for p in parts if '=' in p);mounts.append((d['src'],d['dst'],'readonly' not in parts))
 assert sorted(mounts)==sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind') and len(mounts)==len(obj['Mounts'])
 result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'verification_plan_sha256':sha(V/'verification_plan001.json'),'GT_opened':False,'scoring_performed':False,'cpu_tests':13,'gpu_authorized':'subject to root scheduling','scope':'384 post-hoc DEV natural generations; no scoring or timing','limitations':['prior DEV mixed prediction journal read for input identity; four smoke output files hashed','dynamic K is not equal-K or equal-compute','threshold chosen after DEV results','three real processors checked in no-GPU preflight, all 384 rechecked at runtime','full high-dimensional visual/DS/KV contents not archived','GPU inference and quality not yet verified'],'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
