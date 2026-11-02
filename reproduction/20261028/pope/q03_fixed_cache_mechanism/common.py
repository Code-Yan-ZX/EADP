"""Stdlib-only frozen cache/source guards; default check never imports torch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'gpu_formula_facility_panel'
PLAN = HERE / 'plan.json'
EXEC = HERE / 'execution'
PYTHON = Path('/home/dell/miniconda3/envs/llava_pruner/bin/python')
RUN_ID = 'POPE_fixed16_q02_to_q03_cache_only_alpha05_beta1_K32'
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: datetime.now(timezone.utc).isoformat()
LOCKED = {
    OLD/'run_plan.json': '355f020bb2bcaf9ac37f4daa061f504ec532e155c0cb61be5aa966fc3e2df4ff',
    OLD/'execution/loaded_runtime.json': '32fa78e5ec98a69b15dfa42568b6a2b3eef00c81546098568aaaeff8d249c10a',
    OLD/'execution/per_image.jsonl': '1948d294bb12027872995f1d6dd0d4e711bf9b4a64c40fdd00bfaf4a5b053ff1',
    OLD/'execution/summary.json': '3cb20b156ed8e9986a1326d9ff0c2b49ef997c49bea05fba31fca9533ec58e08',
    OLD/'independent_verification.json': 'ef6e790c133b9f247130f841ed7b93be6f41fc43844a3e4da8eed539b8ef7d46',
    OLD/'root_terminal_process_proof.json': 'c690cc0d0f7f24999031d4b7a90b9b6f3f3211e351cf89600b8b7743387cf581',
    OLD/'formula_helpers.py': 'ce0c928ee20c035739eb2376e8e057cc46b8ea1bf79a9334feb39bf6c02d2e6a',
    OLD/'diagnose_facility.py': 'c306488c601b9e04213010e76325fe845a28020b0446b3d20a01abbd863731c7',
    HERE.parent/'q03_cache_feasibility_20261029.json': '34a0a4b22185d9b27516fff2cd58cb5af205d3a7696fa54922f61723475296a7',
}


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
        temp = path.with_name('.' + path.name + '.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def identity(pid):
    try:
        ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError: return None
    return dict(pid=pid, start_ticks=ticks, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def check(plan):
    import helpers
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    assert plan['run_id'] == RUN_ID and plan['n'] == 16 and plan['positions'] == list(range(16))
    assert plan['parameters'] == dict(source_q=.2, variant_q=.3, alpha=.5, beta=1., K=32)
    assert plan['variants'] == ['source', 'q03'] and plan['source_AST_proof'] == helpers.nodes()[-1]
    for entry in plan['references'] + plan['implementations'] + [r['cache'] for r in plan['records']]:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    assert [r['position'] for r in plan['records']] == plan['positions']
    assert len({r['image'] for r in plan['records']}) == 16
    assert 'torch' not in sys.modules


def validate_allocation(note, plan_sha, current_pids, current_identities):
    assert note['authorization'] == 'explicit_root_gpu_allocation' and note['run_id'] == RUN_ID
    assert note['plan_sha256'] == plan_sha and note['GPU_slot'] == 'A' and note['CUDA_VISIBLE_DEVICES'] == '0'
    assert note['maximum_concurrent_models_including_loading'] == 2
    assert note['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert current_pids == sorted(note['allowed_other_gpu_pids']) and len(current_pids) < 2
    assert sorted(p['pid'] for p in note['allowed_other_gpu_processes']) == current_pids
    for item in note['allowed_other_gpu_processes']:
        assert set(item) == {'pid', 'start_ticks', 'boot_id'} and current_identities[item['pid']] == item


def forbidden_imports():
    return [name for name in sys.modules if name.startswith(('transformers', 'vlmeval', 'flash_attn', 'llava'))]


def prepare():
    import helpers
    assert not PLAN.exists() and not EXEC.exists() and not (HERE/'launch.claim.json').exists()
    for path, digest in LOCKED.items(): assert sha(path) == digest
    old_plan = read(OLD/'run_plan.json'); runtime = read(OLD/'execution/loaded_runtime.json')
    old_root = read(OLD/'root_terminal_process_proof.json'); old_audit = read(OLD/'independent_verification.json')
    assert old_root['root_exec_session'] == 49578 and old_root['observed_exec_terminal_exit_code'] == 0
    assert old_root['original_worker_exited'] is True and old_root['gpu_pid_absent'] is True
    assert old_root['summary_sha256'] == sha(OLD/'execution/summary.json')
    assert old_audit['success'] is True and old_audit['n_images'] == old_audit['source_anchor_images_exact'] == 16
    assert old_audit['n_cases'] == 192 and old_audit['benchmark_answers_read'] is False
    assert runtime['torch'] == '2.1.2+cu118' and runtime['CUDA'] == '11.8'
    assert runtime['default_dtype'] == 'torch.float32' and runtime['autocast_enabled'] is False
    rows = [json.loads(line) for line in (OLD/'execution/per_image.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and all(r['native_source_anchor_matches'] for r in rows)
    actual = read(HERE/'CPU_import_probe.json')
    assert actual['torch_version'] == runtime['torch'] and actual['torch_cuda_build'] == runtime['CUDA']
    assert actual['CUDA_initialized'] is False and actual['forbidden_imports'] == []
    records = []
    for row in rows:
        case = next(r for r in row['cases'] if r['recipe']['name'] == 'alpha05_beta1' and r['variant'] == 'source')
        assert case['recipe']['alpha'] == .5 and case['recipe']['beta'] == 1. and case['quota'] == [32]
        records.append(dict(position=row['index'], index=row['index'], image=row['panel']['image'],
            question_id=row['panel']['question_id'], image_sha256=row['panel']['image_sha256'],
            prompt_sha256=hashlib.sha256(row['panel']['prompt'].encode()).hexdigest(),
            cache=dict(path=row['cache_path'], sha256=row['cache_sha256']), reference_case=case))
    references = [dict(path=str(p), sha256=d) for p, d in LOCKED.items()]
    references += [dict(path=p, sha256=d) for p, d in old_plan['source_sha256'].items()]
    references += [dict(path=str(HERE/'CPU_import_probe.json'), sha256=sha(HERE/'CPU_import_probe.json'))]
    paper = read(HERE/'public_ablation_evidence.json')
    references += [dict(path=str(HERE/'public_ablation_evidence.json'), sha256=sha(HERE/'public_ablation_evidence.json'))] + paper['source_files']
    files = ['common.py', 'helpers.py', 'worker.py', 'run.py', 'verify_terminal.py', 'cpu_acceptance.py', 'cpu_import_probe.py', 'README.md']
    plan = dict(run_id=RUN_ID, prepared_utc=utc(), status='prepared_not_scheduled', n=16, positions=list(range(16)),
        parameters=dict(source_q=.2, variant_q=.3, alpha=.5, beta=1., K=32), variants=['source','q03'],
        python=str(PYTHON), references=references, implementations=[dict(path=str(HERE/n), sha256=sha(HERE/n)) for n in files],
        records=records, source_AST_proof=helpers.nodes()[-1], expected_runtime=actual, observed_original_runtime=runtime,
        graph_cut='Saved full FP32 global/dense77/visualsim, BEFORE entropy filtering. Each arm independently runs unchanged original filter/topk/floor/min1 then aggregation/fusion/global normalization/smoothing/power/facility. Original FP16 projected visual features reused losslessly; no cross-modal GEMM or feature extraction.',
        anchor_boundary='Original native alpha1/beta1 encode supplies common real cached features. Target alpha.5/beta1 importance/ordered32/full gains/final mask are same-run real GPU source-helper outputs, not a separate native encode or actual LLM prepare anchor. Final FP16 spatial sequence is losslessly derived from saved projected features and effective mask.',
        source_acceptance=['importance','ordered32','full32_gain_trace','finalmask','derived_spatial_FP16_sequence'],
        source_failure='Persist source tensors and differences, mark image unmeasured and skip q03. No CPU substitute, flag tuning, retry or selection by outcomes.',
        public_basis=paper,
        runtime_boundary='Original runtime recorded torch/CUDA, defaultFP32/autocastfalse/matmulTF32false/cudnnTF32true. Original cuDNN benchmark/deterministic were not recorded; disclose current defaults without tuning and require byteexact source anchors.',
        scope=dict(model_loaded=False,FA2_imported=False,GT_read=False,answers_read=False,accuracy_computed=False,LLM_generated=False),
        launch_policy=dict(root_note_required=True,slot='A',CUDA_VISIBLE_DEVICES='0',single_attempt=True,automatic_retry=False,
            maximum_concurrent_models_including_loading=2,no_waiter_or_queue=True),
        limits='One preregistered q=.2 to .3 fixed16 mechanism probe supported by public ablation, not main default, known paper argv, repair or full benchmark. Complete terminal audit may include explicitly unmeasured source failures. No Eq6/Eq7/Eq10 or class/environment variation.')
    save(PLAN, plan, exclusive=True); check(plan)
    save(HERE/'READY.json', dict(status='prepared_not_scheduled', plan_sha256=sha(PLAN), GPU_started=False,
        CUDA_initialized=False, model_loaded=False, watcher_started=False), exclusive=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] == 'prepare'; prepare()
