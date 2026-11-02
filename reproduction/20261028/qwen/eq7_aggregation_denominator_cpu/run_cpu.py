"""CPU-only replay of the fixed sixteen accepted Qwen feature caches.

The graph cut is the saved native FP32 global/dense cross similarities. Neither
the encoder nor cross-modal GEMMs are rerun. Only Eq7's aggregation denominator
changes; CPU reference results never become a claim about an unexecuted GPU arm.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from typing import Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'gpu_formula_facility_panel_run02'
SOURCE = ROOT / '_reproduction/author_eadp/Qwen_vl/model/pruner.py'
PAPER = ROOT / 'reproduction/20261028/pope/arxiv_release_check_20261028/official_latest_source.tar.gz'
PLAN = HERE / 'plan.json'
LOCKED = {
    SOURCE: 'bf809614c207142b8a4fc945f29a8828688c88a2896ebd349d55c4a0b1a917e6',
    OLD / 'run_plan.json': 'd752c876367803c5728ffa84a819fd63b4c129cfd0e2eb78931f7d94dae5d4ad',
    OLD / 'independent_verification.json': '3b9a0a0d582d972bfeccb4f765cd2754f2512c7ea43c1c5a985b419f2db61c46',
    OLD / 'root_terminal_process_proof.json': '03516e429818c7dc39a1db27e396b921ff8462ce4628dc64f7add23b4f6794ea',
    OLD / 'execution/per_question.jsonl': '9429b7ae99b3ca41bafafb1cc9f16e69460331be5dadc4160db59a60ce3fbaab',
}
read = lambda path: json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def adump(node):
    return ast.dump(node, include_attributes=False)


def nodes():
    tree = ast.parse(SOURCE.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VisualTokenPruner')
    compute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_importance')
    source = functions['_local_aggregation_impl']
    changed = copy.deepcopy(source)
    matches = [n for n in ast.walk(changed) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'logits'
               and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div)
               and isinstance(n.value.right, ast.BinOp) and isinstance(n.value.right.op, ast.Add)
               and isinstance(n.value.right.left, ast.Name) and n.value.right.left.id == 'M_temp'
               and isinstance(n.value.right.right, ast.Constant) and n.value.right.right.value == 1e-6]
    assert len(matches) == 1
    denominator = copy.deepcopy(matches[0].value.right)
    matches[0].value.right = copy.deepcopy(denominator.left)
    inverse = copy.deepcopy(changed)
    assignment = next(n for n in ast.walk(inverse) if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'logits'
                      and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div)
                      and isinstance(n.value.right, ast.Name) and n.value.right.id == 'M_temp')
    assignment.value.right = denominator
    assert adump(inverse) == adump(source)
    greedy = functions['_greed_select_impl']
    observed = copy.deepcopy(greedy)
    loop = next(n for n in observed.body if isinstance(n, ast.For))
    position = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'best_idx')
    loop.body.insert(position, ast.parse('gain_trace.append(total_gain.detach().clone())').body[0])
    inverse = copy.deepcopy(observed)
    del next(n for n in inverse.body if isinstance(n, ast.For)).body[position]
    assert adump(inverse) == adump(greedy)
    proof = dict(source_aggregation_AST_sha256=hashlib.sha256(adump(source).encode()).hexdigest(),
        eq7_aggregation_AST_sha256=hashlib.sha256(adump(changed).encode()).hexdigest(),
        only_change='negative_entropy: -(H)/(M_temp+1e-6) -> -(H)/M_temp',
        aggregation_inverse_restores_source=True, greedy_observer_inverse_restores_source=True,
        compute_importance_AST_sha256=hashlib.sha256(adump(compute).encode()).hexdigest())
    return functions, compute, changed, observed, proof


def prepare():
    assert not PLAN.exists() and not (HERE / 'execution').exists()
    for path, digest in LOCKED.items():
        assert sha(path) == digest
    records = [json.loads(line) for line in (OLD / 'execution/per_question.jsonl').read_text().splitlines()]
    assert len(records) == 16 and [r['position'] for r in records] == [(2*i+1)*5349//32 for i in range(16)]
    assert read(OLD / 'independent_verification.json')['success'] is True
    assert read(OLD / 'root_terminal_process_proof.json')['root_owned_exec_actual_exit_code'] == 0
    for record in records:
        assert sha(record['cache']['path']) == record['cache']['sha256']
    with tarfile.open(PAPER) as archive:
        method = archive.extractfile('sections/methodology.tex').read().decode()
        appendix = archive.extractfile('sections/appendix_or_supplementary.tex').read().decode()
    assert r'(-h_i / \gamma)' in method and r'\gamma=0.01' in appendix
    save_new(PLAN, dict(scope='Fixed16 CPU post-cross-similarity cache replay; no GPU or model',
        parameters=dict(q=.2, alpha=.5, beta=2., K=256, source_denominator=.010001, eq7_denominator=.01),
        source_math_proof=nodes()[-1], worker=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        references=[dict(path=str(p), sha256=d) for p,d in LOCKED.items()] + [dict(path=str(PAPER), sha256=sha(PAPER))],
        paper_spec=dict(methodology_lines=[119,124], appendix_default_gamma_line=349,
            method_sha256=hashlib.sha256(method.encode()).hexdigest(), appendix_sha256=hashlib.sha256(appendix.encode()).hexdigest()),
        records=records, no_GT_or_answers=True, CPU_source_anchor_policy='Always report bitwise equality and deltas against native GPU cache; conditional CPU results if any mismatch',
        expected_Python='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'))


def compiled(torch, functions, compute, aggregate, observed, global_sim, dense_sim):
    namespace = dict(torch=torch, F=torch.nn.functional, Tuple=Tuple, aggregation_capture={}, gain_trace=[])
    for node in list(functions.values()) + [compute]:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])), '<fresh_source_AST>', 'exec'), namespace)
    # Explicit graph input cut: both arms consume the same saved native FP32
    # similarities. This is not an approximation to a fresh cross-modal GEMM.
    namespace['_sim_cross'] = lambda *args: (global_sim, dense_sim)
    original_greedy = namespace['_greed_select_impl']
    expected_aggregate = copy.deepcopy(aggregate)
    aggregate = copy.deepcopy(aggregate)
    branch = next(n for n in ast.walk(aggregate) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare) and isinstance(n.test.comparators[0], ast.Constant)
                  and n.test.comparators[0].value == 'negative_entropy')
    branch.body.append(ast.parse('aggregation_capture.update(logits=logits, weights=weights, local_sim=local_sim, entropy=top_entropy_vals)').body[0])
    inverse = copy.deepcopy(aggregate)
    del next(n for n in ast.walk(inverse) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
             and isinstance(n.test.comparators[0], ast.Constant) and n.test.comparators[0].value == 'negative_entropy').body[-1]
    assert adump(inverse) == adump(expected_aggregate)
    for node in [aggregate, observed]:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<source_postcall_observers>', 'exec'), namespace)
    return namespace, original_greedy


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan = read(PLAN)
    assert sha(__file__) == plan['worker']['sha256']
    assert Path(sys.executable).resolve() == Path(plan['expected_Python']).resolve()
    for entry in plan['references']:
        assert sha(entry['path']) == entry['sha256']
    assert nodes()[-1] == plan['source_math_proof']
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = HERE / 'execution'
    out.mkdir(exist_ok=False)
    pid = os.getpid()
    ticks = int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    save_new(HERE / 'process_identity.json', dict(pid=pid, start_ticks=ticks, CPU_only=True,
        CUDA_VISIBLE_DEVICES='', plan_sha256=sha(PLAN), torch=torch.__version__, python=sys.version,
        threads=torch.get_num_threads(), cuda_initialized=False, models_loaded=0))
    functions, compute, changed, observed, _ = nodes()
    results = []
    def meta(value):
        array = value.detach().contiguous().numpy()
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
    def delta(a,b):
        difference=(a-b).abs()
        return dict(bitwise_equal=torch.equal(a,b), max_abs_delta=float(difference.max()),
            n_values_changed=int((a!=b).sum()), mean_abs_delta=float(difference.mean()))
    def margins(trace):
        top=trace.topk(2,dim=-1).values
        return top[...,0]-top[...,1]
    for index, record in enumerate(plan['records']):
        assert sha(record['cache']['path']) == record['cache']['sha256']
        arrays = np.load(record['cache']['path'], allow_pickle=False)
        def tensor(name):
            array=arrays[name]
            assert hashlib.sha256(array.tobytes()).hexdigest() == record['tensors'][name]['sha256']
            value=torch.from_numpy(array.copy())
            if record['tensors'][name]['dtype']=='torch.bfloat16': value=value.view(torch.bfloat16)
            assert str(value.dtype)==record['tensors'][name]['dtype'] and list(value.shape)==record['tensors'][name]['shape']
            assert bool(torch.isfinite(value).all())
            return value
        features=tensor('visual_features_BF16_bits');global_text=tensor('text_global_mean_BF16_bits');text=tensor('text_sequence_BF16_bits')
        global_sim=tensor('global_sim');dense=tensor('dense_sim');sim=tensor('visual_sim')
        gpu_importance=tensor('native_importance');gpu_ids=tensor('native_ordered_ids');gpu_gains=tensor('source__gain_trace')
        assert features.shape==(1024,4096) and global_sim.shape==(1,1024,1) and dense.shape==(1,1024,1,record['actual_instruction_L'])
        outcomes={};saved={}
        for name,aggregate in [('source',functions['_local_aggregation_impl']),('eq7_exact_denominator',changed)]:
            namespace,plain=compiled(torch,functions,compute,aggregate,observed,global_sim,dense)
            importance=namespace['compute_importance'](SimpleNamespace(alpha=.5,beta=2.),features.unsqueeze(0),global_text,text,32,32)
            ids,quotas=namespace['_greed_select_impl'](importance,sim,256)
            gains=torch.stack(namespace['gain_trace'])
            if name=='source':
                plain_ids,plain_quotas=plain(importance,sim,256)
                assert torch.equal(ids,plain_ids) and torch.equal(quotas,plain_quotas)
            assert quotas.tolist()==[256] and len(set(ids[0].tolist()))==256
            capture=namespace['aggregation_capture']
            value=dict(importance=importance,ids=ids,gains=gains,**capture)
            assert all(bool(torch.isfinite(v).all()) for v in value.values())
            outcomes[name]=value
            saved.update({name+'__'+key:v.detach().numpy() for key,v in value.items()})
        source,variant=outcomes['source'],outcomes['eq7_exact_denominator']
        spatial=lambda value:value['ids'][0].sort().values
        first=lambda a,b:next((i for i,(x,y) in enumerate(zip(a[0].tolist(),b[0].tolist())) if x!=y),None)
        source_first=first(source['ids'],gpu_ids);first_change=first(source['ids'],variant['ids'])
        source_anchor=dict(importance=delta(source['importance'],gpu_importance),
            ordered_ids_equal=torch.equal(source['ids'],gpu_ids), first_ordered_divergence=source_first,
            spatial_ids_equal=torch.equal(spatial(source),gpu_ids[0].sort().values),
            gain_trace=delta(source['gains'],gpu_gains),
            gain_comparison_scope='All steps have identical selection history' if source_first is None else 'After first divergence, gain history differs; full-trace delta is descriptive only')
        pair=dict(weights=delta(source['weights'],variant['weights']), local_sim=delta(source['local_sim'],variant['local_sim']),
            importance=delta(source['importance'],variant['importance']), gain_trace=delta(source['gains'],variant['gains']),
            ordered_ids_equal=torch.equal(source['ids'],variant['ids']), first_ordered_divergence=first_change,
            spatial_ids_equal=torch.equal(spatial(source),spatial(variant)),
            added=sorted(set(spatial(variant).tolist())-set(spatial(source).tolist())),
            removed=sorted(set(spatial(source).tolist())-set(spatial(variant).tolist())),
            CPU_source_min_argmax_margin=float(margins(source['gains']).min()),
            CPU_variant_min_argmax_margin=float(margins(variant['gains']).min()),
            archived_GPU_source_min_argmax_margin=float(margins(gpu_gains).min()),
            gain_comparison_scope='All steps have identical selection history' if first_change is None else 'Only steps through first divergence share history; full-trace delta is descriptive only')
        strict=source_anchor['importance']['bitwise_equal'] and source_anchor['ordered_ids_equal'] and source_anchor['gain_trace']['bitwise_equal']
        result=dict(position=record['position'],index=record['index'],cache=record['cache'],CPU_source_vs_archived_GPU=source_anchor,
            exact_GPU_source_anchor_closed_on_CPU=strict, interpretation='CPU conditional reference; no GPU variant has run' if not strict else 'Source anchor matches saved GPU exactly, but variant is still only CPU reference',
            source_vs_eq7=pair, tensors={name:{key:meta(v) for key,v in data.items()} for name,data in outcomes.items()},
            source_ordered_ids=source['ids'][0].tolist(),eq7_ordered_ids=variant['ids'][0].tolist())
        destination=out/f'question_{index:02d}.npz'
        np.savez_compressed(destination,**saved)
        result['CPU_output_arrays']=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size)
        save_new(out/f'question_{index:02d}.json',result)
        results.append(result)
        print(json.dumps(dict(completed=len(results),position=record['position'],source_exact_GPU_anchor=strict,
            CPU_ordered_ids_changed=not pair['ordered_ids_equal'],CPU_spatial_ids_changed=not pair['spatial_ids_equal'])),flush=True)
    for entry in plan['references']+[r['cache'] for r in plan['records']]: assert sha(entry['path'])==entry['sha256']
    assert not torch.cuda.is_initialized()
    summary=dict(status='complete',n_questions=16,plan_sha256=sha(PLAN),worker_sha256=sha(__file__),
        checked_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,torch=torch.__version__,CUDA_initialized=False,
        model_loaded=False,GT_or_answers_read=False,accuracy_computed=False,actual_GPU_variant_executed=False,
        exact_source_importance_vs_GPU=sum(r['CPU_source_vs_archived_GPU']['importance']['bitwise_equal'] for r in results),
        exact_source_order_vs_GPU=sum(r['CPU_source_vs_archived_GPU']['ordered_ids_equal'] for r in results),
        exact_source_spatial_set_vs_GPU=sum(r['CPU_source_vs_archived_GPU']['spatial_ids_equal'] for r in results),
        strict_source_all_array_anchors=sum(r['exact_GPU_source_anchor_closed_on_CPU'] for r in results),
        CPU_weights_changed=sum(not r['source_vs_eq7']['weights']['bitwise_equal'] for r in results),
        CPU_importance_changed=sum(not r['source_vs_eq7']['importance']['bitwise_equal'] for r in results),
        CPU_ordered_ids_changed=sum(not r['source_vs_eq7']['ordered_ids_equal'] for r in results),
        CPU_spatial_sets_changed=sum(not r['source_vs_eq7']['spatial_ids_equal'] for r in results),
        records=results,limitations=['Both arms start at saved native cross similarities, not recomputed encoder/text features.',
            'CPU kernels/reductions differ from archived GPU; failed bitwise source anchors make facility results conditional CPU references.',
            'This tests only the fixed16 cached questions, q.2/alpha.5/beta2/K256, and one epsilon; no paper recipe or accuracy claim.',
            'A later GPU cache-only replay could avoid loading8B, but must first reproduce saved source importance, all256 ordered IDs and gain trace bitwise. No GPU launcher is prepared here.'])
    save_new(HERE/'comparison.json',summary)
    print(json.dumps({key:value for key,value in summary.items() if key not in ('records','limitations')}),flush=True)


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='prepare': prepare()
    elif len(sys.argv)==2 and sys.argv[1]=='run': main()
    else: raise SystemExit('Use prepare (stdlib) or run (CUDA_VISIBLE_DEVICES must be empty)')
