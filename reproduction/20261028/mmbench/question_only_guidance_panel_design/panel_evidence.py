"""Read-only panel/input/history checks; no model, GT columns or scoring."""
import hashlib,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;MMB=HERE.parent;AUDIT=MMB/'question_only_guidance_cpu_audit_20261030';PRIOR=MMB/'streamwait_control'
EXPECTED_REPORT='a1ce95c88c11abba589eb38835761cd1e6408a995e1519833503228342160759'
EXPECTED_PREDICTIONS_SHA='6d9a38f9117b354a937eb0288cb287de4e573fa31bcda24dcb612ec5fc423754'
EXPECTED_RUNTIME_SHA='710acc77a7c027a52ebcaed7c8413aaac444e81f2116ad7d9033f04e39113ee4'
EXPECTED_GROUPS=[72,217,370,498,632,767,909,1030,1143,1255,1364,1480,1591,1706,1821,1936]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def rows(path):
 with path.open() as f:return [json.loads(line) for line in f]
def check():
 report=json.loads((AUDIT/'report.json').read_text());fingerprints=AUDIT/'token_fingerprints.jsonl'
 assert sha(AUDIT/'report.json')==EXPECTED_REPORT and report['success'] and report['n_rows']==4377 and report['n_circular_groups']==1176
 assert sha(fingerprints)==report['fingerprints']['sha256']
 samples=rows(fingerprints);groups=sorted({r['group'] for r in samples});chosen=[groups[(2*i+1)*len(groups)//32] for i in range(16)]
 assert chosen==EXPECTED_GROUPS and len(samples)==4377 and len(groups)==1176
 panel=[r for r in samples if r['group'] in set(chosen)];assert len(panel)==60 and len({r['index'] for r in panel})==60
 per_group={group:[r for r in panel if r['group']==group] for group in chosen}
 assert sorted(len(v) for v in per_group.values())==[2,2]+[4]*14
 assert all(len({r['image_payload_sha256'] for r in v})==1 and len({r['question_only_guidance']['text_sha256'] for r in v})==1 for v in per_group.values())
 assert all(r['full_guidance']['input_ids_sha256']!=r['question_only_guidance']['input_ids_sha256'] for r in panel)
 assert all(r['question_only_guidance']['n_segments']==1 for r in panel)
 state=json.loads((PRIOR/'state.json').read_text());supervisor=json.loads((PRIOR/'supervisor_state.json').read_text())
 raw=PRIOR/'v15_EADP_K128_oldEN_beta1_alpha05_streamwait.jsonl';runtime=PRIOR/'observed/runtime.jsonl'
 assert state['status']=='complete' and state['exit_code']==0 and supervisor['status']=='complete' and supervisor['exit_code']==0
 assert state['n_answers']==4377 and sha(raw)==state['prediction_sha256']==EXPECTED_PREDICTIONS_SHA and sha(runtime)==EXPECTED_RUNTIME_SHA
 answers=rows(raw);observed=rows(runtime);assert len(answers)==len(observed)==4377
 known={r['index']:r for r in panel};selected=[(a,o) for a,o in zip(answers,observed) if int(a['question_id']) in known]
 assert len(selected)==60 and len({int(a['question_id']) for a,o in selected})==60
 reference=json.loads((PRIOR/'cpu_inputs_current.json').read_text());llm=json.loads((PRIOR/'cpu_llm_tokens_current.json').read_text());llm_byid={x['index']:x for x in llm['records']}
 durations=[]
 for a,o in selected:
  idx=int(a['question_id']);expected=known[idx];assert a['round_id']==0 and isinstance(a['text'],str)
  assert hashlib.sha256(a['prompt'].encode()).hexdigest()==expected['full_guidance']['text_sha256']==o['guidance_text_sha256']
  assert o['input_shape']==[1,llm_byid[idx]['n_tokens']] and o['image_shape']==[1,3,336,336] and o['image_dtype']=='torch.float16'
  assert o['image_sizes'][0]==reference['images'][expected['image_payload_sha256']]['size']
  assert o['max_new_tokens']==1024 and o['do_sample'] is False and o['returned_visual_token_num']==128
  durations.append(o['seconds'])
 return dict(success=True,n_groups=16,n_rows=60,round_id_zero=60,expected_groups=chosen,source_text_hash_matches=60,source_llm_length_matches=60,source_image_geometry_matches=60,old_literal_answers_available=60,
  full_guidance_over77=sum(r['full_guidance']['n_tokens']>77 for r in panel),question_only_max_tokens=max(r['question_only_guidance']['n_tokens'] for r in panel),
  historical_source_call_seconds_sum=round(sum(durations),3),historical_source_call_seconds_median=round(statistics.median(durations),3),
  historical_output_ids_or_masks_available=False,model_loaded=False,GPU_started=False,GT_columns_loaded=False,scoring_started=False)
if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False))
