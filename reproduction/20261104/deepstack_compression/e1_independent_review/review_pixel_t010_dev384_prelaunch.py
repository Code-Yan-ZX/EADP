"""Independent no-GPU review of PixelPrune threshold .2 full-DEV readiness."""
import argparse,collections,hashlib,json,os,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixelprune_t010_dev384';OLD=B/'e6/pixelprune_dev384';A=B/'e6/pixelprune_t010_accept';S=B/'e6/pixelprune_t010_dev384_analysis';M=R/'monitor_pixel_t010_dev384_prelaunch'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def req(ok,why):
 if not ok:raise ValueError(why)
def supervised(m):
 s=read(m/'started.json');t=read(m/'terminal.json');req(t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'],'supervisor '+str(m));return s,t
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();plan=read(D/'plan001.json');ph=sha(D/'plan001.json')
 req((plan['threshold'],plan['method'],plan['metric'],plan['anchored'],plan['vit_layer'])==(.1,'pred_2d','max',True,0) and plan['GT_opened'] is False and plan['predictions_opened_for_identity'] is True,'pixel config')
 def sources():
  for path,h in plan['sha256'].items():req(sha(path)==h,'source changed '+path)
 sources();supervised(D/'monitor_test002');supervised(D/'monitor_freeze001');ps,pt=supervised(D/'monitor_preflight001');supervised(R/'monitor_pixel_t010_result')
 old=read(OLD/'plan001.json');anchors=plan['anchors'];req(anchors==old['anchors'] and len(anchors)==384 and len({(a['task'],str(a['qid'])) for a in anchors})==384 and len({a['group_id'] for a in anchors})==384 and collections.Counter(a['task'] for a in anchors)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128},'DEV identity/order')
 sm=read(R/'pixel_t010_result_review.json');sp=read(B/'e6/pixelprune_t010_smoke/plan001.json');req(sha(R/'pixel_t010_result_review.json')==plan['smoke_review_sha256'] and sha(B/'e6/pixelprune_t010_smoke/plan001.json')==plan['smoke_plan_sha256'] and sm['status']=='passed' and sm['independent_review'] is True and sm['reviewer_sha256']==sha(R/'review_pixel_t010_result.py') and sm['images']==4 and sm['GT_opened'] is False and sm['scoring_performed'] is False and len(sm['files_sha256'])==16 and all(sha(p)==h for p,h in sm['files_sha256'].items()),'four-image proof')
 req({(x['task'],str(x['qid']),x['visual_tokens']) for x in sm['observed']}=={(x['task'],str(x['qid']),x['cpu_K']) for x in sp['anchors']} and all(x['cpu_gpu_indices_equal'] is True and x['eos'] is True for x in sm['observed']),'four-image ordered identity')
 pre=read(D/'preflight001.json');launch=read(D/'run_preflight001/launch.json');end=read(D/'run_preflight001/exit.json');cid=end['container_identity']['container_id']
 req(pre['status']=='passed' and pre['source_plan_sha256']==ph and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['weight_sha256_verified']==4 and pre['patch_before_model'] is True and len(pre['processor_verified'])==3 and end['returncode']==end['container_exit_code']==0 and end['container_stopped_verified'] is True and end['failure_type'] is None and end['timed_out'] is False and end['interrupted'] is None and ps['worker_pid']==launch['controller']['pid']==end['controller']['pid'] and launch['plan_sha256']==ph,'preflight exit')
 ai={(a['task'],str(a['qid'])):a for a in anchors};req({x['task'] for x in pre['processor_verified']}=={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'} and all(x['processor']==ai[(x['task'],str(x['qid']))]['processor'] for x in pre['processor_verified']),'actual processor checks')
 obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=20).stdout)[0];state=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];req(not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'] and host['DeviceRequests'] is None and not host['Devices'] and host['NetworkMode']=='none' and cfg['User']=='1000:1000' and obj['Image']==end['container_identity']['image'] and cfg['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')],'actual preflight container')
 env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';test=subprocess.run(['/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python','-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=B,env=env,text=True,capture_output=True,timeout=120);req(test.returncode==0 and 'Ran 13 tests' in test.stderr,'independent tests '+test.stderr[-1500:])
 av=read(A/'verification_plan001.json');req(av['panel_plan_sha256']==ph and av['expected_questions']==384 and av['GT_opened'] is False and av['scoring_performed'] is False and all(sha(p)==h for p,h in av['source_sha256'].items()),'pre-frozen acceptance plan');supervised(A/'monitor_test001')
 sv=read(S/'score_plan001.json');req(sv['status']=='frozen_before_panel_results' and sv['threshold']==.1 and sv['arms']==['pixel_t010','pixel_t0','e4_k256_d1','e4_k1024_d1'] and sv['bootstrap']['unit']=='image group' and sv['bootstrap']['paired'] is True and sv['bootstrap']['replicates']==10000 and sv['bootstrap']['seed']==20261104 and len(sv['contrasts'])==3,'pre-frozen scoring plan')
 sources();proof={'status':'passed','independent_review':True,'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_terminal_path':str(M/'terminal.json'),'review_started_sha256':sha(M/'started.json'),'plan_sha256':ph,'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'acceptance_plan_sha256':sha(A/'verification_plan001.json'),'score_plan_sha256':sha(S/'score_plan001.json'),'source_count':len(plan['sha256']),'questions':384,'independent_cpu_tests':13,'GT_opened':False,'new_panel_predictions_opened':False,'predictions_opened_for_identity':True,'limitations':plan['limitations'],'scoring_source_prelaunch_still_required':True}
 with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
