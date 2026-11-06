"""Complete dual-acceptance gates, before reading new capture contents."""
import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';T=B/'e4_multistream_budget_diagnostic_v2';P=B/'e4_multistream_expansion352';V=B/'e4_multistream_expansion352_launch_v4';F=B/'e4_multistream_expansion343_launch';R=B/'e4_multistream_expansion343_gpu/run001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1041048),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve():raise ValueError('supervisor '+str(m))
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
def prelaunch():
 plan=read(D/'plan_freeze003.json');cpu=read(D/'cpu_proof003.json');code=read(D/'code_plan003.json');terminal(D/'monitor_freeze_plan003',D/'freeze_plan003.py');terminal(D/'monitor_cpu003',D/'freeze_code003.py')
 if plan.get('status')!='frozen' or plan.get('plan_sha256')!=sha(D/'PLAN.md') or plan.get('new_343_values_opened') is not False or plan.get('GT_opened') is not False or any(sha(k)!=v for k,v in plan['identity_and_prior_sha256'].items()) or cpu.get('status')!='passed' or cpu['source_before']!=cpu['source_after'] or code.get('status')!='frozen' or code.get('source_sha256')!=cpu['source_after'] or code.get('cpu_sha256')!=sha(D/'cpu_proof003.json') or code.get('plan_sha256')!=sha(D/'plan_freeze003.json') or any(sha(k)!=v for k,v in code['source_sha256'].items()):raise ValueError('analysis freeze')
 proof=I/'e4_multistream_expansion343_analysis_v3_prelaunch_review.json';reviewer=I/'review_e4_multistream_expansion343_analysis_v3_prelaunch.py';terminal(I/'monitor_e4_multistream_expansion343_analysis_v3_prelaunch',reviewer,proof);q=read(proof)
 fields={'reviewer_sha256':reviewer,'plan_sha256':D/'plan_freeze003.json','code_sha256':D/'code_plan003.json','cpu_sha256':D/'cpu_proof003.json','cpu_started_sha256':D/'monitor_cpu003/started.json','cpu_terminal_sha256':D/'monitor_cpu003/terminal.json'}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GT_opened') is not False or q.get('new_343_values_opened') is not False or q.get('source_sha256')!=code['source_sha256'] or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('analysis independent review')
 return code
def validate_full_proofs(f,q,field_digests,expected):
 if f.get('status')!='passed' or f.get('images')!=343 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=343 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('predictions_opened') is not True or q.get('errors')!=[] or q.get('files_sha256')!=f.get('files_sha256') or any(q.get(k)!=v for k,v in field_digests.items()):raise ValueError('343 dual acceptance')
 files=f.get('files_sha256')
 if not isinstance(files,dict) or not expected.issubset(files):raise ValueError('343 file inventory')
 return files
def accepted():
 prelaunch()
 # Bind the already accepted original extraction and independent recomputation.
 terminal(T/'monitor_extract001',T/'extract.py',T/'result001.json');terminal(I/'monitor_e4_multistream_budget_v2_result',I/'review_e4_multistream_budget_v2_result.py',I/'e4_multistream_budget_v2_result_review.json')
 old=read(I/'e4_multistream_budget_v2_result_review.json')
 if old.get('status')!='passed' or old.get('independent_review') is not True or old.get('questions')!=32 or old.get('result_sha256')!=sha(T/'result001.json') or old.get('reviewer_sha256')!=sha(I/'review_e4_multistream_budget_v2_result.py') or old.get('errors')!=[]:raise ValueError('32 review')
 terminal(V/'monitor_verify001',V/'verify.py',V/'verification001.json');terminal(I/'monitor_e4_multistream_expansion352_pilot_v4_result',I/'review_e4_multistream_expansion352_pilot_v4_result.py',I/'e4_multistream_expansion352_pilot_v4_result_review.json')
 pilot=read(I/'e4_multistream_expansion352_pilot_v4_result_review.json')
 if pilot.get('status')!='passed' or pilot.get('independent_review') is not True or pilot.get('images')!=9 or pilot.get('formal_sha256')!=sha(V/'verification001.json') or pilot.get('reviewer_sha256')!=sha(I/'review_e4_multistream_expansion352_pilot_v4_result.py') or pilot.get('errors')!=[]:raise ValueError('9 review')
 # Existing E4 formal / independent full-panel and score chain already verified by prior gate.
 path=T/'gate.py';expected='90e27ed0d6714dd2bc100b1d5210b4ff9a7f2bbcd0c76c04767a994917ffc60c'
 if sha(path)!=expected or '_full_multistream_prior_gate' in sys.modules:raise ValueError('old gate source')
 spec=importlib.util.spec_from_file_location('_full_multistream_prior_gate',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
 try:spec.loader.exec_module(m);prior=m.prior(read(T/'plan_freeze001.json'))
 finally:del sys.modules[spec.name]
 # Missing future evidence refuses before opening even a single new feature file.
 formal=F/'verification001.json';review=I/'e4_multistream_expansion343_result_review.json';reviewer=I/'review_e4_multistream_expansion343_result.py'
 terminal(F/'monitor_run001',F/'launch.py');terminal(F/'monitor_snapshot001',F/'snapshot.py',F/'run001_sources');terminal(F/'monitor_verify001',F/'verify.py',formal);terminal(I/'monitor_e4_multistream_expansion343_result',reviewer,review)
 f=read(formal);q=read(review)
 fields={'formal_sha256':formal,'formal_started_sha256':F/'monitor_verify001/started.json','formal_terminal_sha256':F/'monitor_verify001/terminal.json','run_started_sha256':F/'monitor_run001/started.json','run_terminal_sha256':F/'monitor_run001/terminal.json','source_manifest_sha256':F/'run001_sources/manifest.json','reviewer_sha256':reviewer}
 if f.get('verifier_sha256')!=sha(F/'verify.py') or f.get('source_manifest_sha256')!=sha(F/'run001_sources/manifest.json'):raise ValueError('343 formal source')
 expected={str(R/'captures'/('image_%03d%s'% (i,s))) for i in range(343) for s in ('.json','_norms.npy')};files=validate_full_proofs(f,q,{k:sha(v) for k,v in fields.items()},expected)
 if any(sha(k)!=v for k,v in files.items()):raise ValueError('343 file hash')
 for i,row in enumerate(pilot['rows']):
  for k,p in (('record_sha256',B/'e4_multistream_expansion352_gpu_v4/run001/captures'/('image_%03d.json'%i)),('norms_sha256',B/'e4_multistream_expansion352_gpu_v4/run001/captures'/('image_%03d_norms.npy'%i))):
   if sha(p)!=row[k]:raise ValueError('pilot file hash')
 return {'old_result_sha256':sha(T/'result001.json'),'old_review_sha256':sha(I/'e4_multistream_budget_v2_result_review.json'),'pilot_review_sha256':sha(I/'e4_multistream_expansion352_pilot_v4_result_review.json'),'full_formal_sha256':sha(formal),'full_review_sha256':sha(review),'full_source_manifest_sha256':sha(F/'run001_sources/manifest.json'),'prior':prior}
