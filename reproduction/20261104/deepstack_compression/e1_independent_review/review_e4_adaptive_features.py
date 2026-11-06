"""Independent posthoc DEV feature/allocation recheck; reads no new answers or predictions."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e4_adaptive_features';F=D/'features001';O=D/'result001'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def req(x,m):
 if not bool(x):raise ValueError(m)
def terminal(path,script,out):
 s=read(path/'started.json');t=read(path/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);req(t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve(),'supervisor/source');return s,t
def key(r):return (r['task'],str(r['qid']),r['group_id'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan_path=D/'plan001.json';plan=read(plan_path);es,et=terminal(D/'monitor_extract001',D/'extract.py',F);rs,rt=terminal(D/'monitor_analysis001',D/'analyze.py',O);ex=read(F/'extraction.json');report=read(O/'analysis.json')
 req(plan_path.stat().st_mtime<=datetime.fromisoformat(es['created_utc']).timestamp() and ex['status']=='completed' and ex['n']==384 and ex['plan_sha256']==sha(plan_path)==report['plan_sha256'] and ex['extractor_sha256']==sha(D/'extract.py') and ex['GT_opened'] is False and ex['score_or_prediction_opened'] is False,'plan/extraction provenance')
 src=B/plan['inputs'];score=B/plan['scores'];fp=F/'features.jsonl';ap=O/'assignments.jsonl';req(sha(src)==plan['inputs_sha256']==ex['input_sha256'] and sha(score)==plan['scores_sha256']==report['scores_sha256'] and sha(fp)==ex['feature_sha256']==report['feature_sha256'] and sha(ap)==report['assignments_sha256'] and report['status']=='completed' and report['GT_opened'] is False and report['new_predictions_opened'] is False,'source/outputs')
 for path,h in plan['score_proofs'].items():req(sha(B/path)==h and read(B/path)['status']=='passed','score proof')
 raw=lines(src);features=lines(fp);scores=lines(score);assignments=lines(ap);req(len(raw)==len(features)==len(scores)==384 and len(assignments)==2304,'inventory');fi={key(x):x for x in features};si={key(x):x for x in scores};req(len(fi)==len(si)==384 and set(fi)==set(si)=={key(x) for x in raw},'pair identities')
 ai={(x['task'],str(x['qid']),x['group_id'],x['rule']):x for x in assignments};req(len(ai)==2304,'assignment uniqueness')
 panel_path=B/'data_manifest/panel_manifest.jsonl';ep=read(B/'e1/e1full001_completion.json');req(ep['status']=='passed' and Path(ep['files']['panel']['path']).resolve()==panel_path.resolve() and sha(panel_path)==ep['files']['panel']['sha256'],'accepted panel manifest');dev={key(x) for x in lines(panel_path) if x['split']=='DEV'};req(set(fi)==dev and len(dev)==384,'frozen DEV panel')
 for row in raw:
  k=key(row);v=fi[k];path=Path(row['image_path']);req(sha(path)==row['image_sha256']==v['image_sha256'],'image bytes');req(k in dev,'non DEV')
  with Image.open(path) as image:im=image.convert('L').resize((256,256),Image.Resampling.BILINEAR);hist=im.histogram();arr=np.asarray(im,dtype=np.int16)
  bins=[sum(hist[i*16:(i+1)*16]) for i in range(16)];entropy=-sum((x/65536)*math.log2(x/65536) for x in bins if x);gradient=(np.abs(arr[:,1:]-arr[:,:-1]).sum()/float(256*255)+np.abs(arr[1:,:]-arr[:-1,:]).sum()/float(255*256))/255.;chars=sum(len(z['value']) for z in row['message'] if z.get('type')=='text')
  req(abs(entropy-v['gray_entropy16'])<1e-12 and abs(gradient-v['gray_gradient'])<1e-12 and chars==v['user_text_chars'],'feature mismatch')
 digest=lambda v:hashlib.sha256(v.encode('utf8')).hexdigest();summaries={}
 for task in plan['tasks']:
  part=[x for x in features if x['task']==task];req(len(part)==128 and len({x['group_id'] for x in part})==128,'task')
  rankings={name:sorted(part,key=lambda x:(-x[name],digest('20261105|'+x['group_id']))) for name in ('gray_entropy16','gray_gradient','user_text_chars')}
  rankings.update({f'random_{seed}':sorted(part,key=lambda x:digest(f"20261105|random|{seed}|"+x['group_id'])) for seed in range(3)})
  base=[si[key(x)]['arms']['K256-D1']['score'] for x in part];baseline=sum(base)/128;rep=report['tasks'][task];req(rep['n']==128 and math.isclose(rep['fixed_k256_mean_pct'],100*baseline,abs_tol=1e-10),'baseline');summaries[task]={}
  for rule,rank in rankings.items():
   high={x['group_id'] for x in rank[:42]};req(len(high)==42,'high inventory');values=[];deltas=[]
   for x,b in zip(part,base):
    k=512 if x['group_id'] in high else 128;row=ai[(task,str(x['qid']),x['group_id'],rule)];v=si[key(x)]['arms'][f'K{k}-D1']['score'];req(row['selected_k']==k and abs(row['score']-v)<1e-12 and abs(row['baseline_score']-b)<1e-12,'assignment or score');values.append(v);deltas.append(v-b)
   y=rep['rules'][rule];mean=sum(values)/128;req(y['high_k_count']==42 and y['mean_k']==254 and math.isclose(y['mean_pct'],100*mean,abs_tol=1e-10) and math.isclose(y['minus_fixed256_pp'],100*(mean-baseline),abs_tol=1e-10) and y['improved']==sum(x>0 for x in deltas) and y['declined']==sum(x<0 for x in deltas) and y['unchanged']==sum(x==0 for x in deltas),'rule summary');summaries[task][rule]={'high_count':42,'low_count':86,'total_K':32512,'mean_K':254,'mean_pct':100*mean,'minus_fixed256_pp':100*(mean-baseline)}
  req(set(rep['rules'])==set(rankings),'missing rule')
 out={'status':'passed','independent_review':True,'questions':384,'assignments':2304,'rules_per_task':6,'tasks':summaries,'plan_sha256':sha(plan_path),'features_sha256':sha(fp),'scores_sha256':sha(score),'assignments_sha256':sha(ap),'analysis_sha256':sha(O/'analysis.json'),'extract_started_sha256':sha(D/'monitor_extract001/started.json'),'extract_terminal_sha256':sha(D/'monitor_extract001/terminal.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'new_predictions_opened':False,'existing_scored_prediction_records_opened':True,'prediction_scope':'E4 existing scored records parsed for score and identity; predictions not used','limitations':['Posthoc exploration on DEV after earlier results, including answer-informed oracle; all three rules and three random seeds retained.','Per-task rank uses all 128 DEV examples and is not an online deployment policy.','Average visual K254 vs fixed K256 is not measured equal compute, memory, or latency; no inference about significance.'],'errors':[]}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 print('independent image feature and rank review passed')
if __name__=='__main__':main()
