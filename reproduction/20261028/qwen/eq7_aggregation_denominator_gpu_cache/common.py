"""Shared stdlib guards. Importing or checking this module never imports torch."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CPU = HERE.parent / 'eq7_aggregation_denominator_cpu'
OLD = HERE.parent / 'gpu_formula_facility_panel_run02'
ROOT = HERE.parents[3]
PYTHON = ROOT / '_reproduction/envs/qwen_author/bin/python'
PLAN = HERE / 'plan.json'
EXEC = HERE / 'execution'
RUN_ID = 'Qwen_fixed16_Eq7_denominator_GPU_cache_only'
HELPER = CPU / 'run_cpu.py'
EXPECTED_HELPER_SHA = '014cb537da51beef92c1a6c0d602860a0eb5466bfbcda5a1fa5e7cdf430c34cd'
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''): h.update(block)
    return h.hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    else:
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)


def identity(pid):
    try:
        ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError: return None
    return dict(pid=pid, start_ticks=ticks,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def helper():
    assert sha(HELPER) == EXPECTED_HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_Eq7_source_AST_helpers', HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check(plan):
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    assert plan['run_id'] == RUN_ID and plan['n'] == 16
    assert plan['parameters'] == dict(q=.2, alpha=.5, beta=2., K=256,
        source_aggregation_denominator=.010001, eq7_aggregation_denominator=.01)
    assert plan['variants'] == ['source', 'eq7_exact_denominator']
    assert plan['positions'] == [(2*i+1)*5349//32 for i in range(16)]
    assert plan['source_AST_proof'] == helper().nodes()[-1]
    for entry in plan['references'] + plan['implementations'] + [r['cache'] for r in plan['records']]:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    assert [r['position'] for r in plan['records']] == plan['positions']
    assert 'torch' not in sys.modules


def validate_allocation(note, plan_sha, current_pids, current_identities):
    assert note['authorization'] == 'explicit_root_gpu_allocation' and note['run_id'] == RUN_ID
    assert note['plan_sha256'] == plan_sha and note['GPU_slot'] == 'A'
    assert note['CUDA_VISIBLE_DEVICES'] == '0'
    assert note['maximum_concurrent_models_including_loading'] == 2
    assert current_pids == sorted(note['allowed_other_gpu_pids']) and len(current_pids) < 2
    expected = note['allowed_other_gpu_processes']
    assert sorted(p['pid'] for p in expected) == current_pids
    for item in expected:
        assert set(item) == {'pid', 'start_ticks', 'boot_id'} and current_identities[item['pid']] == item


def forbidden_imports():
    return [name for name in sys.modules if name.startswith(('transformers', 'vlmeval', 'flash_attn'))]


def prepare():
    assert not PLAN.exists() and not EXEC.exists() and not (HERE/'launch.claim.json').exists()
    original = helper()
    for path, digest in original.LOCKED.items(): assert sha(path) == digest
    records = [json.loads(line) for line in (OLD/'execution/per_question.jsonl').read_text().splitlines()]
    references = [dict(path=str(p),sha256=d) for p,d in original.LOCKED.items()]
    references += [dict(path=str(p),sha256=sha(p)) for p in [HELPER, original.PAPER, OLD/'execution/runtime.json',
        CPU/'plan.json', CPU/'comparison.json', CPU/'CPU_completion.json', HERE/'CPU_import_probe.json']]
    actual = read(HERE/'CPU_import_probe.json')
    assert actual['torch_version'] == '2.8.0+cu128' and actual['torch_cuda_build'] == '12.8'
    assert actual['CUDA_initialized'] is False and actual['forbidden_imports'] == []
    old_runtime = read(OLD/'execution/runtime.json')
    assert old_runtime['default_dtype'] == 'torch.float32' and not old_runtime['autocast'] and not old_runtime['matmul_TF32']
    files = ['common.py','worker.py','run.py','verify_terminal.py','cpu_acceptance.py','cpu_import_probe.py','README.md']
    plan = dict(run_id=RUN_ID,n=16,prepared_utc=utc(),status='prepared_not_scheduled',python=str(PYTHON),
        positions=[(2*i+1)*5349//32 for i in range(16)],
        parameters=dict(q=.2,alpha=.5,beta=2.,K=256,source_aggregation_denominator=.010001,eq7_aggregation_denominator=.01),
        variants=['source','eq7_exact_denominator'],source_AST_proof=original.nodes()[-1],
        references=references,implementations=[dict(path=str(HERE/n),sha256=sha(HERE/n)) for n in files],
        records=[{k:r[k] for k in ['position','index','cache','tensors','actual_instruction_L']} for r in records],
        graph_cut='Saved native FP32 global/dense cross similarities and visual similarity; original BF16 projected features unchanged; no encoder, model, FA2, LLM or cross-modal GEMM',
        source_acceptance=['importance','ordered256','full256_gain_trace','final_spatial_BF16_sequence'],
        source_failure='Save source outputs and diagnostic deltas, mark question unmeasured, skip Eq7 arm; never substitute CPU results',
        expected_runtime=actual,observed_original_runtime=old_runtime,
        flag_boundary='Old runtime records default FP32/autocast false/matmul TF32 false. cuDNN flags were not recorded then; current defaults are disclosed, never tuned, and exact source anchors decide measurement.',
        scope=dict(model_loaded=False,FA2_imported=False,GT_read=False,answers_read=False,accuracy_computed=False,LLM_generated=False),
        launch_policy=dict(root_note_required=True,slot='A',CUDA_VISIBLE_DEVICES='0',single_attempt=True,automatic_retry=False,
            maximum_concurrent_models_including_loading=2,no_waiter_or_queue=True),
        limits='Only fixed16 feature-cache pruning. Complete can include explicitly unmeasured source anchors; success of terminal auditing does not mean all variants measured or paper reproduced.')
    save(PLAN,plan,exclusive=True);check(plan)
    save(HERE/'READY.json',dict(status='prepared_not_scheduled',plan_sha256=sha(PLAN),GPU_started=False,
        CUDA_initialized=False,model_loaded=False,watcher_started=False),exclusive=True)


if __name__ == '__main__':
    if len(sys.argv)==2 and sys.argv[1]=='prepare': prepare()
    else: raise SystemExit('Use prepare only; run.py defaults to a CPU-only check')
