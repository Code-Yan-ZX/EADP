"""Independent prelaunch review of post-hoc PixelPrune threshold 0.1 smoke."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_t010_smoke';R=B/'e1_independent_review';C=B/'e6/pixel_low_threshold_cpu';P=B/'e6/pixelprune_dev384';S=B/'e6/pixelprune_threshold32';M=R/'monitor_pixel_t010_prelaunch'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def supervised(p):
 s=read(p/'started.json');t=read(p/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('unclean supervision '+str(p))
 return s,t
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();plan=read(D/'plan001.json')
 if (plan['threshold'],plan['method'],plan['metric'],plan['anchored'],plan['vit_layer'])!=(0.1,'pred_2d','max',True,0) or plan['GT_opened'] is not False or plan['predictions_opened_for_identity'] is not True:raise ValueError('plan controls')
 def check_sources():
  for path,expected in plan['sha256'].items():
   if sha(path)!=expected:raise ValueError('source drift '+path)
 check_sources();supervised(D/'monitor_test001');supervised(D/'monitor_freeze001');ps,pt=supervised(D/'monitor_preflight001');supervised(R/'monitor_pixel_low_threshold_cpu')
 low=read(R/'pixel_low_threshold_cpu_review.json');rows=read(C/'result001.json')['rows'];samples=read(S/'plan001.json')['samples'];original=read(P/'plan001.json');original_index={(x['task'],str(x['qid'])):x for x in original['anchors']}
 if low['status']!='passed' or low['independent_review'] is not True or low['case_count']!=96 or low['ordered_matches']!=96 or low['GT_opened'] is not False or low['predictions_opened'] is not False or low['result_sha256']!=sha(C/'result001.json') or low['plan_sha256']!=sha(C/'plan001.json') or low['reviewer_sha256']!=sha(R/'review_pixel_low_threshold_cpu.py') or sha(R/'pixel_low_threshold_cpu_review.json')!=plan['calibration_review_sha256']:raise ValueError('calibration evidence')
 subset=[x for x in rows if x['threshold']==0.1]
 if len(subset)!=32 or not all(x['ordered_equal'] for x in subset):raise ValueError('calibration denominator')
 chosen=[next(x for x in subset if x['identity']['task']==task) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')];used={x['index'] for x in chosen};chosen.append(min((x for x in subset if x['index'] not in used),key=lambda x:(x['count'],x['index'])))
 if len(plan['anchors'])!=4 or len({a['group_id'] for a in plan['anchors']})!=4:raise ValueError('anchor denominator')
 for a,x in zip(plan['anchors'],chosen):
  sample=samples[x['index']];key=(a['task'],str(a['qid']));ref=original_index[key]
  if x['index']!=a['cpu_index'] or x['count']!=a['cpu_K'] or x['selected_indices']!=a['cpu_selected_indices'] or x['identity']['task']!=a['task'] or str(x['identity']['qid'])!=str(a['qid']) or sample['group_id']!=a['group_id'] or sha(sample['capture_path'])!=sample['capture_sha256'] or any(ref[k]!=a[k] for k in ('group_id','message_sha256','image_sha256','processor','image_start')):raise ValueError('anchor binding')
 pre=read(D/'preflight001.json');launch=read(D/'run_preflight001/launch.json');end=read(D/'run_preflight001/exit.json');ph=sha(D/'plan001.json');cid=end['container_identity']['container_id']
 if pre['status']!='passed' or pre['source_plan_sha256']!=ph or pre['cuda_initialized'] is not False or pre['model_loaded'] is not False or pre['GT_opened'] is not False or pre['weight_sha256_verified']!=4 or pre['patch_before_model'] is not True or len(pre['processor_verified'])!=4 or end['returncode']!=0 or not end['container_stopped_verified'] or end['container_exit_code']!=0 or end['failure_type'] is not None or end['timed_out'] or end['interrupted'] is not None or ps['worker_pid']!=launch['controller']['pid'] or ps['worker_pid']!=end['controller']['pid'] or launch['plan_sha256']!=ph:raise ValueError('preflight exit')
 for a,v in zip(plan['anchors'],pre['processor_verified']):
  if (v['task'],str(v['qid']),v['processor'])!=(a['task'],str(a['qid']),a['processor']):raise ValueError('processor identity')
 obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=20).stdout)[0];state=obj['State'];host=obj['HostConfig'];config=obj['Config']
 if state['Running'] or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or host['DeviceRequests'] is not None or host['Devices'] or host['NetworkMode']!='none' or config['User']!='1000:1000' or obj['Image']!=end['container_identity']['image'] or config['Cmd']!=[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]:raise ValueError('actual container mismatch')
 env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';test=subprocess.run(['/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python','-m','unittest','discover','-s',str(D),'-p','test_worker.py','-v'],cwd=B,env=env,text=True,capture_output=True,timeout=120)
 if test.returncode or 'Ran 11 tests' not in test.stderr:raise ValueError('independent synthetic tests failed: '+test.stderr[-2000:])
 check_sources();proof={'status':'passed','independent_review':True,'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_terminal_path':str(M/'terminal.json'),'review_started_sha256':sha(M/'started.json'),'plan_sha256':ph,'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'source_count':len(plan['sha256']),'anchors':[{k:a[k] for k in ('task','qid','cpu_K')} for a in plan['anchors']],'independent_cpu_tests':11,'GT_opened':False,'predictions_opened_for_identity':True,'gpu_launched':False,'limitations':plan['limitations']}
 with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
