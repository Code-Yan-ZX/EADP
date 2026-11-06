"""Require independent source review before second-batch scorer opens data."""
import importlib.metadata,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;B=ROOT/'m3_oracle_batch2';sys.path.insert(0,str(B));import batch_gate
if Path(getattr(batch_gate,'__file__','/')).resolve()!=(B/'batch_gate.py').resolve():raise ValueError('wrong batch gate import')
CPU=HERE/'cpu_check_007/verification.json';MON=HERE/'monitor_cpu007';REVIEW=HERE/'independent_prelaunch_review.json';REVIEWER=ROOT/'e1_independent_review/review_m3batch2_score.py';RM=ROOT/'e1_independent_review/monitor_m3batch2_score_prelaunch'
def sources():
 first=json.loads((ROOT/'m3_oracle_analysis/analysis_approval.json').read_text());old=json.loads(Path(first['cpu_proof']).read_text());extra=[Path(x) for x in json.loads((ROOT/'e1/e1full001_score/frozen.json').read_text())['code_sha256']]+[Path(x) for x in old['source_sha256_after']]+[B/'batch_gate.py']
 return extra+[HERE/x for x in ('plan.json','plan_001_before_timing_clarification.json','analysis_gate.py','score_gate.py','score_oracle.py','paired.py','cpu_validation.py','test_gate.py','test_score_pipeline.py','test_native_predictions.py','test_paired.py','test_source_pinning.py')]+[ROOT/x for x in ('m3_oracle_batch2_verify/verify.py','m3_oracle_batch2_verify/container_identity.py','e1_analysis/score.py','e1_analysis/completion.py','e1_analysis/metrics.py','e1_analysis/paired_stats.py','m3_oracle_exec/e1_admission.py','m3_oracle_exec/e1_baseline.py')]
def validate():
 for p in (CPU,MON/'started.json',MON/'terminal.json',REVIEW,REVIEWER,RM/'started.json',RM/'terminal.json'):
  if not p.is_file():raise ValueError('second-batch scorer locked: independent CPU review required')
 batch_gate.supervised(MON/'started.json',MON/'terminal.json',HERE/'cpu_validation.py',CPU.parent);batch_gate.supervised(RM/'started.json',RM/'terminal.json',REVIEWER,REVIEW)
 old=ROOT/'m3_oracle_analysis/analysis_gate.py';sp=importlib.util.spec_from_file_location('m3_first_accepted_analysis_gate',old);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);module.validate()
 frozen=json.loads((ROOT/'e1/e1full001_score/frozen.json').read_text());expected=frozen['code_sha256'];proof=json.loads((ROOT/'e1_independent_review/e1full001_score_review.json').read_text())
 if len(expected)!=6 or any(batch_gate.sha(p)!=h for p,h in expected.items()) or proof.get('status')!='passed' or proof.get('score_frozen_sha256')!=batch_gate.sha(ROOT/'e1/e1full001_score/frozen.json'):raise ValueError('E1 scoring provenance changed')
 for mon,script in ((ROOT/'e1/monitorscoree1full001',ROOT/'e1_analysis/score.py'),(ROOT/'e1_independent_review/monitor_e1score001',ROOT/'e1_independent_review/review_score.py')):
  st=json.loads((mon/'started.json').read_text());et=json.loads((mon/'terminal.json').read_text());cwd=Path(st['cwd']);cmd=st['command']
  if et.get('reason')!='completed' or et.get('worker_returncode')!=0 or et.get('remaining_process_group')!=[] or st['worker_pid']!=st['process_group'] or st['worker_pid']!=et['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script]:raise ValueError('E1 scorer supervisor mismatch')
 if proof.get('score_terminal_sha256')!=batch_gate.sha(ROOT/'e1/monitorscoree1full001/terminal.json') or proof.get('reviewer_sha256')!=batch_gate.sha(ROOT/'e1_independent_review/review_score.py'):raise ValueError('E1 scorer review mismatch')
 if any(__import__('importlib').metadata.version(k)!=v for k,v in frozen['runtime_versions'].items()):raise ValueError('E1 scorer dependency version changed')
 c=json.loads(CPU.read_text());r=json.loads(REVIEW.read_text());actual={str(p.resolve()):batch_gate.sha(p) for p in sources()}
 if (c.get('status')!='passed' or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual or c.get('GT_opened') is not False or c.get('predictions_opened') is not False or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('source_sha256')!=actual or r.get('cpu_sha256')!=batch_gate.sha(CPU) or r.get('cpu_started_sha256')!=batch_gate.sha(MON/'started.json') or r.get('cpu_terminal_sha256')!=batch_gate.sha(MON/'terminal.json') or r.get('reviewer_sha256')!=batch_gate.sha(REVIEWER) or r.get('GT_opened') is not False or r.get('predictions_opened') is not False):raise ValueError('second-batch scorer proof changed')
 return {'review_sha256':batch_gate.sha(REVIEW),'cpu_sha256':batch_gate.sha(CPU),'source_sha256':actual}
