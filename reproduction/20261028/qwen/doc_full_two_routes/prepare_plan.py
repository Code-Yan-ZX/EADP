"""Freeze one native-model, full-Doc two-path calibration; never launch GPU."""
from pathlib import Path
import common as c

def main():
 assert not c.PLAN.exists() and not (c.HERE/'execution').exists() and not (c.HERE/'launch.claim.json').exists()
 panel=c.read(c.PANEL/'execution/independent_verification.json');root=c.read(c.PANEL/'root_terminal_process_proof.json')
 assert panel['success'] and panel['n_questions']==16 and panel['n_native_generations']==32
 assert root['root_owned_exec_actual_exit_code']==0 and root['all_original_processes_exited']
 assert c.read(c.HERE/'score_cpu_smoke_result.json')['actual_exit']==0
 prep=c.read(c.HERE/'input_preparation.json');assert prep['rows']==5349 and prep['images']==1286 and not prep['answers_copied']
 assert prep['input_manifest_sha256']==c.sha(c.HERE/'input_manifest.jsonl')
 files=['common.py','contracts.py','prepare_inputs.py','prepare_plan.py','worker.py','controller.py','cpu_container_check.py',
        'cpu_preflight.py','verify_generation.py','score_terminal.py','score_cpu_smoke.py','run_score_cpu_smoke.py','score_cpu_smoke_result.json','cpu_tests.py','README.md','input_manifest.jsonl','input_preparation.json']
 refs=[c.HERE/name for name in files]+[c.PANEL/'plan.json',c.PANEL/'root_terminal_process_proof.json',c.PANEL/'execution/independent_verification.json',
   c.PANEL/'execution/release.json',c.OLD/'run_plan.json']
 old=c.read(c.OLD/'run_plan.json');src=Path(old['author_root'])/'VLMEvalKit/vlmeval/dataset'
 score_paths=[c.Q/'independent_audit/DocVQA_VAL.canonical.jsonl',c.Q/'independent_audit/audit_qwen.py',
  src/'image_vqa.py',src/'utils/vqa_eval.py']
 p=dict(run_id=c.RUN_ID,status='prepared_not_started',rows=5349,model_instances=1,
   file_guards={str(x):c.sha(x) for x in refs},score_only_sha256={str(x):c.sha(x) for x in score_paths},
   execution_policy='One model; fixed order plain then unregistered EADP K1024 per canonical item; actual response saved before observation gates. No automatic retry or queue.',
   scoring_policy='Independent processor/output verification after native container exit. Official DocVQA evaluator and independent ANLS only after actual controller/root PTY exit proof. Preserve every answer and all 5349 denominator.',
   limits=['Two full paths are a joint intervention, not isolated DeepStack or mRoPE causality.',
     'K1024 is an explicit unregistered EADP diagnostic; paper main-table argv remains unverified.',
     'Input tensor snapshots cost memory and time; timing is not an efficiency benchmark.',
     'Root-owned PTY actual-exit proof and separate scoring step are required.'])
 c.save(c.PLAN,p);c.check(p)
 c.save(c.HERE/'READY.json',dict(status='CPU_prepared_not_launched',plan_sha256=c.sha(c.PLAN),model_loaded=False,GPU_started=False,queue_started=False))
 print(c.sha(c.PLAN))
if __name__=='__main__':main()
