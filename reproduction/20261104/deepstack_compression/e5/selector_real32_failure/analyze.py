"""Posthoc CPU arithmetic audit of independently accepted divergent selections.

Does not reproduce unarchived GPU reduction order, kernel intermediates or scores.
"""
import argparse,json,math,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal

def divergence(torch,similarity,importance,author,candidate):
 sim=similarity.detach().cpu();weight=importance.detach().cpu()
 if sim.ndim==3 and sim.shape[0]==1:sim=sim[0]
 if weight.ndim==2 and weight.shape[0]==1:weight=weight[0]
 n=sim.shape[0]
 if sim.shape!=(n,n) or weight.shape!=(n,) or len(author)!=len(candidate) or len(set(author))!=len(author) or len(set(candidate))!=len(candidate) or any(type(x) is not int or not 0<=x<n for x in author+candidate):raise ValueError('invalid selection or numeric inputs')
 if not bool(torch.isfinite(sim).all()) or not bool(torch.isfinite(weight).all()):raise ValueError('nonfinite selector input')
 first=next((i for i,(a,b) in enumerate(zip(author,candidate)) if a!=b),None);out={'first_divergent_step_zero_based':first,'first_divergent_pick_one_based':None if first is None else first+1,'ordered_positions_differing':sum(a!=b for a,b in zip(author,candidate)),'set_intersection':len(set(author)&set(candidate)),'set_union':len(set(author)|set(candidate)),'author_only':sorted(set(author)-set(candidate)),'candidate_only':sorted(set(candidate)-set(author)),'author_order':author,'candidate_order':candidate}
 if first is None:return out
 ai,bi=author[first],candidate[first];prefix=author[:first]
 if prefix!=candidate[:first] or ai in prefix or bi in prefix:raise ValueError('invalid common prefix')
 def score(dtype,index):
  s=sim.to(dtype);w=weight.to(dtype);coverage=torch.zeros(n,dtype=dtype) if not prefix else s[prefix].amax(dim=0);return float(((torch.maximum(s[index],coverage)-coverage)*w).sum().item())
 a64,b64=score(torch.float64,ai),score(torch.float64,bi);a32,b32=score(torch.float32,ai),score(torch.float32,bi)
 if any(not math.isfinite(x) for x in (a64,b64,a32,b32)):raise ValueError('nonfinite reconstructed score')
 out.update(author_pick=ai,candidate_pick=bi,common_prefix_length=first,cpu_float64={'author_gain':a64,'candidate_gain':b64,'author_minus_candidate':a64-b64},cpu_float32={'author_gain':a32,'candidate_gain':b32,'author_minus_candidate':a32-b32},score_scope='CPU reconstruction from stored float32 inputs and common prefix; not the original GPU reduction or recorded per-round score')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=BASE/'e5/selector_real32';rdir=BASE/'e1_independent_review';formal=d/'run_001_verification.json';fs=d/'monitor_verify_run001/started.json';ft=d/'monitor_verify_run001/terminal.json';review=rdir/'selector_real32_run001_review.json';rs=rdir/'monitor_selector_real32_run001/started.json';rt=rdir/'monitor_selector_real32_run001/terminal.json';reviewer=rdir/'review_selector_real32_run001.py';runs=d/'monitor_run001/started.json';runt=d/'monitor_run001/terminal.json'
 supervised(fs,ft,d/'verify.py',formal);supervised(rs,rt,reviewer,review);terminal(runs,runt);f=json.loads(formal.read_text());r=json.loads(review.read_text())
 if f.get('status')!='passed' or r.get('status')!='passed' or r.get('independent_review') is not True or f.get('images')!=32 or f.get('cases')!=96 or r.get('images')!=32 or r.get('cases')!=96 or f.get('GT_opened') is not False or r.get('GT_opened') is not False or r.get('reviewer_sha256')!=sha(reviewer) or r.get('formal_sha256')!=sha(formal) or r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('run_started_sha256')!=sha(runs) or r.get('run_terminal_sha256')!=sha(runt) or r.get('array_sha256')!=f.get('array_sha256'):raise ValueError('independent evidence not complete')
 journal=d/'run_001/journal.jsonl'
 if f.get('files_sha256',{}).get(str(journal.resolve()))!=sha(journal):raise ValueError('journal hash mismatch')
 manifest=d/'sample_manifest_001.json'
 if f.get('source_sha256',{}).get(str(manifest.resolve()))!=sha(manifest):raise ValueError('sample manifest changed')
 pilot=json.loads(manifest.read_text())['pilot'];events=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()];cases=[e for e in events if e.get('status')=='case_completed'];failures=[];not_available=[]
 import torch
 torch.set_num_threads(1)
 for e in cases:
  i=e.get('image_index');k=e.get('budget');v=e.get('result')
  if not isinstance(i,int) or not 0<=i<32 or k not in (128,256,512):raise ValueError('invalid case key')
  if not isinstance(v,dict) or v.get('status')!='measured':not_available.append({'image_index':i,'budget':k,'scope':'all methods','reason':v if isinstance(v,str) else v.get('reason')});continue
  path=d/'run_001'/('image_%03d'%i)/('case_%d.pt'%k)
  if f.get('array_sha256',{}).get(str(path.resolve()))!=sha(path):raise ValueError('numeric input archive changed')
  data=torch.load(path,map_location='cpu',weights_only=True);author=data['ordered'][0].tolist()
  for name in ('fused','graph'):
   if v.get('candidate_matches_author',{}).get(name) is True:continue
   initial=v.get('identity',{}).get(name+'_ordered_ids');observations=[('initial',initial)]+[('repeat_'+str(x.get('repeat')),x.get('ordered_ids')) for x in v['measurements'] if x.get('implementation')==name and x.get('status')=='measured'];differing=next(((label,ids) for label,ids in observations if ids is not None and ids!=author),None)
   if differing is None:not_available.append({'image_index':i,'budget':k,'method':name,'reason':'no divergent recorded order; unavailable or gate rejected'});continue
   label,ids=differing;info=divergence(torch,data['similarity'],data['importance'],author,ids);info.update(image_index=i,task=pilot[i]['task'],qid=str(pilot[i]['qid']),group_id=pilot[i]['group_id'],budget=k,method=name,observation=label,numeric_archive_sha256=sha(path));failures.append(info)
 out={'status':'completed','scope':'posthoc CPU reconstruction of all recorded real32 candidate failures; no GT, no GPU and no new model predictions','planned_images':32,'planned_cases':96,'formal_counts':f.get('counts'),'divergences':failures,'unavailable':not_available,'formal_sha256':sha(formal),'independent_review_sha256':sha(review),'journal_sha256':sha(journal),'source_sha256':sha(__file__),'limitations':['CPU float64/float32 gains are recomputed from stored float32 similarity and importance; the original CUDA/Triton per-step gains were not archived','floating-point reduction order is a plausible difference, not proven by this analysis','posthoc one-case failure analysis is not a universal equivalence result']}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
if __name__=='__main__':main()
