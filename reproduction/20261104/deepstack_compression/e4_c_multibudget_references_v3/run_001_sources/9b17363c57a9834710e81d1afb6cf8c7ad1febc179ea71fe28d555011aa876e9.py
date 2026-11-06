"""Prespecified 32-generation budget endpoint diagnostic; no GT or timing."""
BUDGETS=(128,256,512,1024)
PANEL_FIELDS=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','forced','forced_token','decode_trace','block_calls','main_merger_calls','ds_calls')

def schedule(rows):
 if len(rows)!=3 or len({r['task'] for r in rows})!=3 or len({r['group_id'] for r in rows})!=3 or any(r.get('split')!='DEV' for r in rows):raise ValueError('three distinct DEV task/image anchors required')
 jobs=[]
 for row in rows:
  for budget in BUDGETS:
   for forced in ([False,True] if row is rows[0] else [False]):
    jobs.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'forced':forced})
 return jobs

def compare_panel(actual,panel):
 if panel.get('actual_injection_verified') is not True:raise ValueError('panel DS identity absent')
 for key in PANEL_FIELDS:
  if key not in actual or key not in panel or actual[key]!=panel[key]:raise ValueError('accepted E4 panel mismatch: '+key)
 return True

def run(rows,refs,forced_token,run_fresh,run_light,archive_factory,emit,compare_pair,compare_arrays):
 if type(forced_token) is not int:raise ValueError('fixed token unavailable')
 jobs=schedule(rows);bykey={(r['task'],str(r['qid'])):r for r in rows};completed=0
 for j in jobs:
  row=bykey[(j['task'],j['qid'])];out=[]
  for label,fn in [('fresh',run_fresh),('light',run_light)]:
   emit({'event':'arm_started','side':label,'job':j})
   try:
    rec=fn(row,j['budget'],forced_token if j['forced'] else None,
           lambda raw:emit({'event':'raw_generation','side':label,'job':j,'raw':raw}),archive_factory(j,label))
    emit({'event':'arm_completed','side':label,'job':j,'record':rec});out.append(rec);completed+=1
   except BaseException as exc:
    emit({'event':'arm_failed','side':label,'job':j,'error_type':type(exc).__name__,'error':str(exc)});raise
  if not j['forced']:compare_panel(out[0],refs[(j['task'],j['qid'],j['budget'])])
  result=compare_pair(out[1],out[0]);arrays=compare_arrays(out[1],out[0])
  emit({'event':'pair_verified','job':j,'comparison':result,'arrays':arrays})
 if len(jobs)!=16 or completed!=32:raise ValueError('diagnostic denominator mismatch')
 return {'pairs':len(jobs),'generations':completed,'natural_pairs':12,'forced_pairs':4,'forced_decode_steps':32}
