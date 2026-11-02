"""Freeze fixed16 references, implementation and original source guards, CPU only."""
from pathlib import Path
import common as c

def main():
    assert not c.PLAN.exists() and not (c.HERE/'execution').exists()
    old=c.read(c.PREV/'run_plan.json')
    import sys
    sys.path.insert(0,str(c.PREV))
    import diagnose
    diagnose.validate(old)
    root=c.read(c.PREV/'root_terminal_process_proof.json')
    assert root['root_owned_exec_actual_exit_code']==0 and root['all_original_proc_paths_absent']
    records=[__import__('json').loads(x) for x in (c.PREV/'execution/per_question.jsonl').read_text().splitlines()]
    assert len(records)==16 and [r['position'] for r in records]==[r['position'] for r in old['panel']]
    refs=[c.PREV/'run_plan.json',c.PREV/'root_terminal_process_proof.json',c.PREV/'independent_verification.json',
          c.PREV/'execution/per_question.jsonl']
    refs += [Path(r['cache']['path']) for r in records]
    for r in records: assert c.sha(r['cache']['path'])==r['cache']['sha256']
    refs += [c.HERE/n for n in ('common.py','prepare.py','worker.py','observer.py','cpu_tests.py','README.md','controller.py','cpu_preflight.py','cpu_container_check.py','verify_terminal.py')]
    p=dict(run_id=c.RUN_ID,status='prepared_not_started',panel=old['panel'],base_recipe=old['recipe'],
        visual_token_num=1024,registered_model=False,benchmark_result=False,source_plan=str(c.PREV/'run_plan.json'),
        source_panel_records=str(c.PREV/'execution/per_question.jsonl'),files={str(f):c.sha(f) for f in refs},
        policy='Same EADP wrapper and same loaded native model/processor: explicitly call original FixedRes and inherited CDPruner generate_inner_transformers_fixed methods on same fixed16 inputs, 32 full native generations. No GT, scoring, benchmark or threshold based on outputs.',
        observation='Hooks snapshot GPU tensors at original forward/processor boundaries (clone operations add overhead but do not replace native computation); CPU reads and serialization follow each generation. Save actual vision output, prefill embedding, position_ids, visual masks, DeepStack, outputs, and processor hashes; preserve raw artifacts before comparisons.',
        limits=['1024 EADP instance is an explicit diagnostic and not an author-registered benchmark recipe.',
                'Two generation APIs may differ in DeepStack, mRoPE and other internals; this test does not isolate a single cause.',
                'Identical token budget and actual input feature do not imply equivalent model execution.',
                'Panel contains 16 fixed Doc questions, not a benchmark or accuracy estimate.',
                'Execution needs separate root allocation, controller, container preflight and independent review; this draft is not a launcher.'])
    c.save(c.PLAN,p); c.plan_checks(p)
    c.save(c.HERE/'READY.json',dict(status='CPU_prepared_only',plan_sha256=c.sha(c.PLAN),GPU_started=False,model_loaded=False,queue_started=False))
if __name__=='__main__':main()
