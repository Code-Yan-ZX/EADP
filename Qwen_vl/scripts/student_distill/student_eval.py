"""Frozen held-out forward-only student QA and separate answer-aware reference.

Usage: python student_eval.py freeze --checkpoint-dir DIR
       python student_eval.py run
Test teacher and generation require a completed eval freeze.  No gold answer is
passed to the student selection function.  Each arm re-prepares and re-encodes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
OUT = QWEN / 'outputs/student_distill'
MULTI = QWEN / 'scripts/multistream_oracle'
for p in (str(QWEN), str(MULTI)):
    if p not in sys.path: sys.path.append(p)

import torch
from experiment_common import NativeEngine, common, resolution
from oracle import score_full_streams, generate_selection

TEACHERS = ('magnitude_v', 'magnitude_joint', 'signed_v', 'signed_joint')
SEEDS = (11, 23, 37)
BASELINES = ('full', 'rres512', 'b2', 'random')
STUDENTS = tuple(f'{teacher}__seed_{seed}' for teacher in TEACHERS for seed in SEEDS)
ORACLES = tuple('oracle_' + teacher for teacher in TEACHERS)
ARMS = BASELINES + STUDENTS + ORACLES
K = 256
MAX_TOKENS = 32
RANDOM_SEED = 20261109
MANIFEST = OUT / 'data/test_manifest.json'
DEV_MANIFEST = OUT / 'data/dev_manifest.json'
FREEZE = OUT / 'eval_freeze.json'
RESULTS = OUT / 'test_predictions.json'
INFERENCE_GATES = OUT / 'student_inference_gates.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def save_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def convert(obj):
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)): return [convert(x) for x in obj]
    return obj


def message(image_path, question):
    # The prediction path deliberately does not accept an item/answer object.
    return [dict(type='image', value=image_path),
            dict(type='text', value=question + '\nAnswer with the value only.')]


def stable_keep(scores, k=K):
    """Top K ties are broken by increasing original index; restore raster order."""
    scores = scores.reshape(-1).float()
    if not torch.isfinite(scores).all() or k <= 0 or k > scores.numel():
        raise ValueError('non-finite scores or invalid budget')
    return torch.argsort(scores, descending=True, stable=True)[:k].sort().values


def random_keep(item_id, n=1024, k=K):
    raw = hashlib.sha256(f'{RANDOM_SEED}:{item_id}'.encode()).digest()
    seed = int.from_bytes(raw[:8], 'big') % (2**63 - 1)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return seed, torch.randperm(n, generator=gen)[:k].sort().values


def input_features(V, DS, question):
    if V.shape != (1024, 4096) or len(DS) != 3 or any(x.shape != V.shape for x in DS):
        raise ValueError('expected [1024,4096] V and three aligned DS streams')
    if question.shape != (1, 4096):
        raise ValueError('expected one [1,4096] native question embedding')
    # Match cache serialization exactly: each stream and question are bf16.
    return torch.stack((V, *DS), dim=0).unsqueeze(0).to(torch.bfloat16), question.to(torch.bfloat16)


def select_student(model, V, DS, question):
    features, q = input_features(V, DS, question)
    scores = model(features, q)
    if tuple(scores.shape) != (1, 1024): raise ValueError('bad student score shape')
    scores = scores[0].float()
    return stable_keep(scores), scores


def checkpoint_paths(directory):
    directory = Path(directory).resolve()
    return {arm: directory / (arm + '.pt') for arm in STUDENTS}


def verify_runtime_model(eng, train_info=None):
    """Check the loaded runtime against the model frozen before teacher labels."""
    if train_info is None:
        train_info = json.loads((OUT/'train_freeze.json').read_text())
    model_info = train_info['model']
    revision = model_info['revision']
    snapshot = Path(model_info['snapshot']).resolve()
    if not snapshot.is_dir() or snapshot.name != revision:
        raise RuntimeError('frozen model snapshot/revision is missing or inconsistent')
    ref = snapshot.parent.parent/'refs/main'
    if not ref.is_file() or ref.read_text().strip() != revision:
        raise RuntimeError('model cached main revision differs from training freeze')
    for name, digest in model_info['metadata_sha256'].items():
        if sha256(snapshot/name) != digest:
            raise RuntimeError('frozen model metadata changed: '+name)
    model_path = str(eng.vlm.model_path)
    loaded_commit = getattr(eng.model.config, '_commit_hash', None)
    if model_path == 'Qwen/Qwen3-VL-8B-Instruct':
        if loaded_commit != revision:
            raise RuntimeError('loaded model commit differs from training revision')
    elif Path(model_path).resolve() != snapshot:
        raise RuntimeError('loaded model path differs from frozen snapshot')
    elif loaded_commit is not None and loaded_commit != revision:
        raise RuntimeError('loaded local snapshot commit differs from training revision')
    actual_dtype = str(next(eng.model.parameters()).dtype)
    if actual_dtype != model_info['dtype']:
        raise RuntimeError('runtime model dtype differs from training freeze')
    expected_backend = train_info['environment']['attention_backend']
    if eng.text.config._attn_implementation != expected_backend:
        raise RuntimeError('runtime attention backend differs from training freeze')
    if eng.model.training:
        raise RuntimeError('runtime model is in training mode')
    return dict(revision=revision, snapshot=str(snapshot), loaded_model_path=model_path,
                loaded_commit=loaded_commit, dtype=actual_dtype, attention_backend=expected_backend)


def source_paths():
    paths = sorted(HERE.glob('*.py'))
    paths += [ROOT / 'docs/student_distill_prereg.md']
    paths += [MULTI / name for name in ('experiment_common.py', 'oracle.py', 'analysis.py')]
    paths += [QWEN / name for name in ('model/native_qwen3.py', 'model/e0_selectors.py',
        'model/pruner.py', 'scripts/discovery/common.py', 'scripts/cf_mining/cf_mine.py',
        'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
        'VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py')]
    return paths


def freeze(checkpoint_dir, training_summary):
    if FREEZE.exists() or RESULTS.exists() or (OUT / 'latency.json').exists():
        raise RuntimeError('refusing to overwrite freeze or existing test results')
    train_freeze = OUT / 'train_freeze.json'
    summary_path = Path(training_summary).resolve()
    summary = json.loads(summary_path.read_text())
    if not summary.get('complete'):
        raise RuntimeError('training completion summary is not complete')
    if summary.get('teachers') != list(TEACHERS) or summary.get('seeds') != list(SEEDS):
        raise RuntimeError('training target/seed plan differs from protocol')
    if summary.get('updates_per_run') != 900:
        raise RuntimeError('training did not complete 900 updates per run')
    index_path = OUT / 'cache_index.json'
    if summary.get('train_freeze_sha256') != sha256(train_freeze):
        raise RuntimeError('training completion belongs to a different train freeze')
    if summary.get('cache_index_sha256') != sha256(index_path):
        raise RuntimeError('training completion belongs to a different cache index')
    prior = json.loads(train_freeze.read_text())
    index = json.loads(index_path.read_text())
    if not prior.get('pre_teacher') or not index.get('complete'):
        raise RuntimeError('missing frozen teacher protocol or complete training cache')
    if index.get('train_freeze_sha256') != sha256(train_freeze):
        raise RuntimeError('training cache belongs to a different train freeze')
    if {str(p) for p in HERE.glob('*.py')} != {
            path for path in prior['source_sha256'] if Path(path).parent == HERE}:
        raise RuntimeError('student source file set changed after train freeze')
    for path, digest in prior['source_sha256'].items():
        if sha256(path) != digest: raise RuntimeError('pre-teacher frozen source changed: '+path)
    for role in ('fit', 'val', 'test', 'dev'):
        if prior['manifest_sha256'].get(role) != sha256(OUT/f'data/{role}_manifest.json'):
            raise RuntimeError('pre-teacher frozen manifest changed: '+role)
    for role in ('fit', 'val'):
        if summary.get(role+'_manifest_sha256') != sha256(OUT / f'data/{role}_manifest.json'):
            raise RuntimeError(f'{role} training manifest mismatch')
    for name, digest in summary.get('sources_sha256', {}).items():
        if sha256(HERE/name) != digest: raise RuntimeError('training source changed: '+name)
    gates = json.loads(INFERENCE_GATES.read_text())
    if not gates.get('pass_all'):
        raise RuntimeError('student inference gates have not passed')
    if gates['train_complete_sha256'] != sha256(summary_path):
        raise RuntimeError('inference gate used different training run')
    for path, digest in gates['source_sha256'].items():
        if sha256(path) != digest: raise RuntimeError('gated source changed: '+path)
    manifest = json.loads(MANIFEST.read_text())
    dev = json.loads(DEV_MANIFEST.read_text())
    if manifest.get('role') != 'test' or dev.get('role') != 'dev':
        raise ValueError('wrong data split role')
    items = manifest['items']; devitems = dev['items']
    if not items or len({x['parent_id'] for x in items}) != len(items):
        raise ValueError('test parents not unique')
    if {x['parent_id'] for x in items} & {x['parent_id'] for x in devitems}:
        raise ValueError('test/dev overlap')
    for item in items + devitems:
        if sha256(item['image_path']) != item['image_sha256']:
            raise ValueError('frozen source image bytes changed')
    files = checkpoint_paths(checkpoint_dir)
    if set(summary.get('checkpoints', {})) != {p.name for p in files.values()}:
        raise RuntimeError('training checkpoint set differs from evaluation plan')
    metadata = {}
    for arm, path in files.items():
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if sha256(path) != summary['checkpoints'][path.name]:
            raise RuntimeError('checkpoint differs from completed training summary: '+str(path))
        if gates['checkpoint_sha256'].get(arm) != sha256(path):
            raise RuntimeError('inference gate checkpoint differs: '+arm)
        teacher, seedstr = arm.rsplit('__seed_', 1)
        if ckpt.get('teacher') != teacher or int(ckpt.get('seed', -1)) != int(seedstr):
            raise ValueError(f'checkpoint metadata mismatch: {path}')
        if not isinstance(ckpt.get('model_state_dict'), dict) or 'best_step' not in ckpt:
            raise ValueError(f'checkpoint missing state or validation selection: {path}')
        metadata[arm] = dict(path=str(path), sha256=sha256(path), teacher=teacher,
            seed=int(seedstr), best_step=ckpt['best_step'], val_head_recall8=ckpt.get('val_head_recall8'))
    src = {str(p.resolve()): sha256(p) for p in source_paths()}
    payload = dict(schema_version=1, test_manifest_sha256=sha256(MANIFEST),
        dev_manifest_sha256=sha256(DEV_MANIFEST), train_freeze_sha256=sha256(train_freeze),
        inference_gates_sha256=sha256(INFERENCE_GATES),
        cache_index_sha256=sha256(index_path),
        training_summary=dict(path=str(summary_path), sha256=sha256(summary_path)),
        checkpoints=metadata, source_sha256=src, arms=list(ARMS), K=K, seeds=list(SEEDS),
        generation=dict(suffix='\nAnswer with the value only.', max_new_tokens=MAX_TOKENS,
                        greedy=True, deepstack=True, normal_eos=True),
        teacher_scope='after checkpoint freeze only; gold/gradient oracle is offline, not deployable',
        random_seed=RANDOM_SEED, pre_test_teacher_and_qa=True)
    save_json(FREEZE, payload)
    print('Frozen held-out evaluation:', FREEZE, sha256(FREEZE), flush=True)


def verify_freeze():
    f = json.loads(FREEZE.read_text())
    for path, digest in [(MANIFEST, f['test_manifest_sha256']), (DEV_MANIFEST, f['dev_manifest_sha256']),
                         (OUT/'train_freeze.json', f['train_freeze_sha256']),
                         (OUT/'cache_index.json', f['cache_index_sha256']),
                         (INFERENCE_GATES, f['inference_gates_sha256']),
                         (f['training_summary']['path'], f['training_summary']['sha256'])]:
        if sha256(path) != digest: raise RuntimeError('frozen input changed: ' + str(path))
    for path, digest in f['source_sha256'].items():
        if sha256(path) != digest: raise RuntimeError('frozen source changed: ' + path)
    if f['arms'] != list(ARMS): raise RuntimeError('planned arms changed')
    for arm, info in f['checkpoints'].items():
        if sha256(info['path']) != info['sha256']: raise RuntimeError('checkpoint changed: '+arm)
    if set(f['checkpoints']) != set(STUDENTS): raise RuntimeError('checkpoint set changed')
    return f


def load_students(frozen, device):
    from student_model import StudentScorer
    models = {}
    for arm in STUDENTS:
        raw = torch.load(frozen['checkpoints'][arm]['path'], map_location='cpu', weights_only=False)
        model = StudentScorer(**raw['config']).to(device)
        model.load_state_dict(raw['model_state_dict'], strict=True)
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        models[arm] = model
    return models


@torch.inference_mode()
def fresh_inputs(eng, image_path, question):
    msg = message(image_path, question)
    prep = eng.prepare(msg, 'DocVQA_VAL')
    V, DS = eng.encode(prep)
    return msg, prep, V, DS


@torch.inference_mode()
def forward_only_arm(eng, image_path, question, item_id, arm, students, oracle_keep=None):
    """No labels accepted; optional oracle_keep is used only for offline arms."""
    if arm not in ARMS: raise ValueError('unknown arm')
    if arm.startswith('oracle_') != (oracle_keep is not None):
        raise ValueError('oracle selection only for explicitly offline arms')
    from contextlib import nullcontext
    with resolution(eng, 512) if arm == 'rres512' else nullcontext():
        msg, prep, V, DS = fresh_inputs(eng, image_path, question)
        expected_n = K if arm == 'rres512' else 1024
        if prep['n_vis'] != expected_n: raise ValueError('unexpected native visual token count')
        scores = None; randseed = None
        if arm in ('full', 'rres512'): keep = None
        elif arm == 'b2':
            tm, ts = eng.instruction_embeds(msg, 'DocVQA_VAL')
            keep = eng.select('b2', K, dict(prep=prep,V=V,DS=DS,K=K,engine=eng,
                              text_mean=tm,text_seq=ts,seed=RANDOM_SEED))
        elif arm == 'random':
            randseed, keep = random_keep(item_id)
        elif arm in STUDENTS:
            q, _ = eng.instruction_embeds(msg, 'DocVQA_VAL')
            keep, scores = select_student(students[arm], V, DS, q)
        else: keep = oracle_keep
        pred = generate_selection(eng, prep, V, DS, keep, max_tokens=MAX_TOKENS)
        actual = 1024 if arm == 'full' else K
        if pred['actual_visual_count'] != actual or pred['ds_lengths'] != [actual]*3:
            raise AssertionError('physical count or DS mismatch')
        if scores is not None: pred['student_scores'] = scores
        if randseed is not None: pred['random_seed'] = randseed
        pred['eos'] = not pred['truncated']
        return convert(pred)


def answer_teacher(eng, item):
    """Offline only: dense gold NLL backward after all checkpoints are frozen."""
    _, prep, V, DS = fresh_inputs(eng, item['image_path'], item['question'])
    result = score_full_streams(eng, prep, V, DS, item['answers'][0], K=K, checkpointing=False)
    return convert(result)


def inference_gates(checkpoint_dir, training_summary):
    """Fit-only cache/live parity and proof that answer mutations have no input path."""
    summary_path = Path(training_summary).resolve()
    summary = json.loads(summary_path.read_text())
    if not summary.get('complete'): raise RuntimeError('training not complete')
    paths = checkpoint_paths(checkpoint_dir)
    if set(summary.get('checkpoints', {})) != {p.name for p in paths.values()}:
        raise RuntimeError('checkpoint inventory mismatch')
    ckpt_info = {}
    for arm,path in paths.items():
        digest = sha256(path)
        if summary['checkpoints'][path.name] != digest: raise RuntimeError('checkpoint hash mismatch')
        ckpt_info[arm] = dict(path=str(path),sha256=digest)
    fit_manifest = OUT / 'data/fit_manifest.json'
    if summary['fit_manifest_sha256'] != sha256(fit_manifest): raise RuntimeError('fit manifest mismatch')
    item = json.loads(fit_manifest.read_text())['items'][0]
    if sha256(item['image_path']) != item['image_sha256']: raise RuntimeError('fit source image changed')
    cachepath = OUT/'cache'/f"{item['item_id']}.pt"
    cache = torch.load(cachepath, map_location='cpu', weights_only=False)
    if cache['item_id'] != item['item_id']: raise ValueError('cache item mismatch')
    eng = NativeEngine(common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS))
    runtime_model = verify_runtime_model(eng)
    students = load_students({'checkpoints':ckpt_info}, next(eng.model.parameters()).device)
    checks = []
    def check(name, passed, **details):
        checks.append(dict(name=name,passed=bool(passed),**details))
        if not passed:
            save_json(INFERENCE_GATES, dict(pass_all=False,checks=checks))
            raise AssertionError(name)
    _,prep,V,DS = fresh_inputs(eng,item['image_path'],item['question'])
    q,_ = eng.instruction_embeds(message(item['image_path'],item['question']),'DocVQA_VAL')
    features, liveq = input_features(V,DS,q)
    check('cached_features_equal_live_bfloat16',torch.equal(cache['features'],features[0].cpu()))
    check('cached_question_equal_live_bfloat16',torch.equal(cache['question'],liveq[0].cpu()))
    changed = dict(item, answers=['999999999999', 'synthetic changed alias'])
    _,_,v2,ds2 = fresh_inputs(eng,changed['image_path'],changed['question'])
    q2,_ = eng.instruction_embeds(message(changed['image_path'],changed['question']),'DocVQA_VAL')
    f2,q2 = input_features(v2,ds2,q2)
    check('changed_answers_same_features',torch.equal(features,f2))
    check('changed_answers_same_question_embedding',torch.equal(liveq,q2))
    with torch.inference_mode():
        for arm,model in students.items():
            live = model(features,liveq)
            cached = model(cache['features'].unsqueeze(0).to(features.device),
                           cache['question'].unsqueeze(0).to(features.device))
            changed_scores = model(f2,q2)
            check(arm+'_cache_scores_equal',torch.equal(live,cached))
            check(arm+'_changed_answer_scores_equal',torch.equal(live,changed_scores))
            check(arm+'_cache_keep_equal',torch.equal(stable_keep(live[0]),stable_keep(cached[0])))
            check(arm+'_changed_answer_keep_equal',torch.equal(stable_keep(live[0]),stable_keep(changed_scores[0])))
    result = dict(pass_all=True,checks=checks,fit_item_id=item['item_id'],
        fit_manifest_sha256=sha256(fit_manifest),cache_sha256=sha256(cachepath),
        train_complete_sha256=sha256(summary_path), runtime_model=runtime_model,
        checkpoint_sha256={arm:info['sha256'] for arm,info in ckpt_info.items()},
        source_sha256={str(p.resolve()):sha256(p) for p in source_paths()},
        note='Only fit sample used; repeated answer mutation never reaches image/question-only scorer.')
    save_json(INFERENCE_GATES,result)
    print(f"student inference gates passed: {len(checks)}",flush=True)


def run():
    f = verify_freeze()
    manifest = json.loads(MANIFEST.read_text()); items = manifest['items']
    config = dict(eval_freeze_sha256=sha256(FREEZE), manifest_sha256=sha256(MANIFEST),
                  planned_arms=list(ARMS), teachers=list(TEACHERS), seeds=list(SEEDS))
    if RESULTS.exists():
        result = json.loads(RESULTS.read_text())
        if result['config'] != config: raise ValueError('resume config mismatch')
    else:
        result = dict(config=config, records=[], complete=False)
    rows = {row['item_id']: row for row in result['records']}
    if len(rows) != len(result['records']) or set(rows) - {x['item_id'] for x in items}:
        raise ValueError('duplicate/unknown item in existing output')
    eng = NativeEngine(common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS))
    verify_runtime_model(eng)
    students = load_students(f, next(eng.model.parameters()).device)
    started = time.time()
    for i, item in enumerate(items):
        if sha256(item['image_path']) != item['image_sha256']: raise RuntimeError('test image changed')
        row = rows.get(item['item_id'])
        if row is None:
            row = dict(item_id=item['item_id'], parent_id=item['parent_id'],
                       image_sha256=item['image_sha256'], question=item['question'],
                       answers=item['answers'], predictions={})
            result['records'].append(row); rows[item['item_id']] = row
            save_json(RESULTS, result)
        if 'teacher' not in row:
            torch.cuda.synchronize(); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            row['teacher'] = answer_teacher(eng, item)
            torch.cuda.synchronize(); row['teacher']['elapsed_seconds'] = time.perf_counter()-start
            row['teacher']['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['teacher']['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            save_json(RESULTS, result)
        for arm in ARMS:
            if arm in row['predictions']: continue
            keep = row['teacher']['keep_indices'][arm[len('oracle_'):]] if arm in ORACLES else None
            if keep is not None: keep = torch.tensor(keep, dtype=torch.long)
            # Never pass the item/answer object to the forward-only selector.
            row['predictions'][arm] = forward_only_arm(eng,item['image_path'],item['question'],
                                                      item['item_id'],arm,students,keep)
            save_json(RESULTS, result)
        print(f'[{i+1}/{len(items)}] {item["item_id"]}: {len(row["predictions"])} arms, '
              f'elapsed {time.time()-started:.1f}s', flush=True)
    result['complete'] = len(result['records']) == len(items) and all(
        'teacher' in row and set(row['predictions']) == set(ARMS) for row in result['records'])
    save_json(RESULTS, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fr = sub.add_parser('freeze'); fr.add_argument('--checkpoint-dir', required=True)
    fr.add_argument('--training-summary', required=True)
    gate = sub.add_parser('gate'); gate.add_argument('--checkpoint-dir', required=True)
    gate.add_argument('--training-summary', required=True)
    sub.add_parser('run')
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.checkpoint_dir, args.training_summary)
    elif args.action == 'gate': inference_gates(args.checkpoint_dir,args.training_summary)
    else: run()


if __name__ == '__main__': main()
