"""CPU-only extraction of three already-reviewed DEV t=.2 identity references.

No labels, model execution or GPU admission. Callers must additionally pin this
file, the returned evidence and a frozen execution plan before using it.
"""
import hashlib,json,struct
from pathlib import Path
ANCHORS=(('DocVQA_VAL','16539'),('TextVQA_VAL','39421'),('AI2D_TEST','368100'))
RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def reference(proof_path,reviewer_path,started_path,terminal_path,observation_dir):
 proof_path=Path(proof_path).resolve();reviewer_path=Path(reviewer_path).resolve();started_path=Path(started_path).resolve();terminal_path=Path(terminal_path).resolve();observation_dir=Path(observation_dir).resolve()
 proof=read(proof_path);started=read(started_path);terminal=read(terminal_path);cmd=started['command'];cwd=Path(started['cwd'])
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid'):raise ValueError('review supervisor incomplete')
 if len(cmd)<4 or (cwd/cmd[1]).resolve()!=reviewer_path or (cmd[-2]!='--out' or (cwd/cmd[-1]).resolve()!=proof_path) or proof.get('reviewer_sha256')!=sha(reviewer_path):raise ValueError('review source/argv mismatch')
 if proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('images')!=4 or proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or proof.get('errors')!=[]:raise ValueError('prior four-image identity review missing')
 files=proof.get('files_sha256');
 if not isinstance(files,dict) or len(files)!=16:raise ValueError('prior evidence inventory mismatch')
 for path,digest in files.items():
  if sha(path)!=digest:raise ValueError('prior evidence changed: '+path)
 results=[]
 for task,qid in ANCHORS:
  op=observation_dir/(task+'_'+qid+'.json');gp=observation_dir/(task+'_'+qid+'_generated.json')
  if str(op) not in files or str(gp) not in files:raise ValueError('unbound observation')
  obs=read(op);gen=read(gp);ids=obs.get('selected_indices');tokens=gen.get('generated_token_ids');full=gen.get('full_sequence_ids');length=gen.get('prompt_length')
  if (obs.get('task'),str(obs.get('qid')))!=(task,qid) or obs.get('status')!='completed' or obs.get('threshold')!=.2 or obs.get('anchored') is not True or obs.get('cpu_gpu_selection_equal') is not True:raise ValueError('anchor provenance mismatch')
  if obs.get('generation_kwargs')!=RECIPE or type(length) is not int or length!=obs.get('sequence_length') or not isinstance(full,list) or full[length:]!=tokens or not isinstance(tokens,list) or not tokens or len(tokens)>2048 or (tokens[-1] not in obs.get('eos_token_ids',[]) and len(tokens)!=2048):raise ValueError('generation contract mismatch')
  if not isinstance(ids,list) or not ids or ids!=sorted(set(ids)) or ids!=obs.get('cpu_expected_indices') or len(ids)!=obs.get('selected_count') or any(type(x) is not int or not 0<=x<1024 for x in ids):raise ValueError('selection mismatch')
  actual=obs['processor_identity']['processor'];inp=actual['input_ids']
  if inp.get('shape')!=[1,length] or inp.get('dtype')!='torch.int64' or any(type(t) is not int or t<0 for t in full) or hashlib.sha256(struct.pack('<'+'q'*length,*full[:length])).hexdigest()!=inp.get('sha256') or any(len(obs.get(k,''))!=64 for k in ('group_id','image_sha256','message_sha256')):raise ValueError('input identity mismatch')
  anchor={k:obs[k] for k in ('task','qid','group_id','image_sha256','message_sha256')}
  results.append({'anchor':anchor,'expected':{'processor':actual,'selected_indices':ids,'tokens':tokens,'eos_ids':obs['eos_token_ids'],'generation_kwargs':RECIPE},'observation_sha256':sha(op),'generation_sha256':sha(gp)})
 if len({x['anchor']['group_id'] for x in results})!=3:raise ValueError('duplicate groups')
 return {'images':results,'review_sha256':sha(proof_path),'reviewer_sha256':sha(reviewer_path),'started_sha256':sha(started_path),'terminal_sha256':sha(terminal_path),'existing_DEV_predictions_opened':True,'GT_opened':False,'gpu_authorized':False}
