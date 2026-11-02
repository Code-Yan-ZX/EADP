"""Pre-register, run and CPU-verify one public-default NeXT TextVQA recipe.

No scheduler is provided. `run` requires an explicitly allocated GPU slot.
Public alpha .5/beta1/K128 is not asserted to be the paper's actual argv.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parents[2] / '_reproduction/author_eadp'
OLD = Path('/media/disk2/YZX/research/EADP_amp')
MODEL = Path('/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b')
DATA = OLD / 'LLaVA/playground/data/eval/textvqa'
PLAN = HERE / 'full_default_K128.plan.json'
RUN = HERE / 'full_default_K128_alpha05_beta1_wait'
PARAMETERS = dict(visual_token_num=128, budget_nominal=640, alpha=.5,
                  beta=1., temperature=0., top_p=None, num_beams=1,
                  max_new_tokens=128, conv_mode='vicuna_v1',
                  attention_backend='sdpa', dtype='torch.float16')
WAITS = ['caller_stream = torch.cuda.current_stream(device=self.device)',
         'image_stream.wait_stream(caller_stream)',
         'text_stream.wait_stream(caller_stream)']
ARCH_SWITCHES = ['USE_LLAVA_ARCH_CDPRUNER', 'USE_LLAVA_ARCH_DIVPRUNE',
                 'USE_LLAVA_ARCH_HIPRUNE', 'USE_LLAVA_ARCH_ABLATION']


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def fingerprint(path):
    p = Path(path)
    stat = p.stat()
    return dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def parameter_dtypes(module):
    return dict(Counter(str(parameter.dtype) for parameter in module.parameters()))


def require_stat(record):
    current = fingerprint(record['path'])
    assert current['bytes'] == record['bytes'], ('size changed', current)
    assert current['mtime_ns'] == record['mtime_ns'], ('mtime changed', current)


def author_imports():
    assert all(os.environ.get(k) != '1' for k in ARCH_SWITCHES), 'Alternative pruning arch enabled'
    sys.path[:] = [str(AUTHOR / 'LLaVA')] + [p for p in sys.path
                      if '/EADP_amp' not in p and '/_reproduction/author_eadp/LLaVA' not in p]
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    return prompt_processor, TextVQAAccuracyEvaluator


def validate_rows(questions, answers, runtime, expected_n=5000):
    """Reject incomplete, reordered, duplicated or misaligned prediction traces."""
    assert len(questions) == len(answers) == len(runtime) == expected_n
    keys = [(str(q['question_id']), q['text']) for q in questions]
    assert len(set(keys)) == expected_n, 'canonical composite keys not unique'
    assert [(str(a['question_id']), a['prompt']) for a in answers] == keys
    for i, (q, a, r) in enumerate(zip(questions, answers, runtime)):
        assert isinstance(a['text'], str), (i, 'answer must be a string')
        assert r['position'] == i and str(r['question_id']) == str(q['question_id'])
        assert r['image'] == q['image']
        assert r['prompt_sha256'] == hashlib.sha256(q['text'].encode()).hexdigest()
        assert type(r['actual_visual_tokens_retained']) is int
        assert 0 < r['actual_visual_tokens_retained'] <= 3000
        assert r['image_tensor_shape'] == [1, 5, 3, 336, 336]
        assert isinstance(r['output_token_ids'], list) and len(r['output_token_ids']) == 1
    return dict(n=expected_n, canonical_composite_keys=expected_n,
                empty_answers=sum(not a['text'].strip() for a in answers),
                retained_count_histogram=dict(sorted(Counter(
                    r['actual_visual_tokens_retained'] for r in runtime).items())))


def prepare():
    assert not RUN.exists(), 'Do not rewrite a plan after its run directory exists'
    prompt_processor, _ = author_imports()
    import torch
    assert not torch.cuda.is_initialized()
    question_path = HERE / 'author_linked_textvqa_questions.jsonl'
    annotation_path = HERE / 'TextVQA_0.5.1_val.fresh.json'
    assert question_path.read_bytes() == (DATA / 'llava_textvqa_val_v051_ocr.jsonl').read_bytes()
    assert annotation_path.read_bytes() == (DATA / 'TextVQA_0.5.1_val.json').read_bytes()
    questions = rows(question_path)
    assert len(questions) == len({(str(q['question_id']), q['text']) for q in questions}) == 5000
    annotations = json.loads(annotation_path.read_text())['data']
    gt_keys = {(str(a['image_id']), a['question'].lower()) for a in annotations}
    assert len(annotations) == len(gt_keys) == 5000
    assert {(str(q['question_id']), prompt_processor(q['text'])) for q in questions} == gt_keys
    source_plan = json.loads((HERE / 'source_equivalence_plan.json').read_text())
    for source in source_plan['source_comparisons']:
        assert sha(source['author_path']) == source['author_sha256']
    weights = json.loads((HERE / 'current_weight_identity.json').read_text())
    assert weights['all_match'] and weights['total_bytes_hashed'] == 14125909456
    for record in weights['files']:
        assert record['sha256'] == record['official_lfs_sha256'] and record['stat_unchanged']
        require_stat(record)
    small = json.loads((HERE / 'current_small_model_identity.json').read_text())
    for record in small['files']:
        assert sha(MODEL / record['file']) == record['sha256']
    model_config = json.loads((MODEL / 'config.json').read_text())
    assert not model_config['mm_use_im_start_end'] and not model_config['mm_use_im_patch_token']
    clip_audit_path = HERE.parent / 'pope/model_identity/model_identity_audit.json'
    clip = json.loads(clip_audit_path.read_text())['models']['clip']
    assert clip['all_expected_runtime_files_present'] and clip['all_present_file_bytes_match']
    from transformers.utils.hub import cached_file
    clip_files = []
    for record in clip['files']:
        assert record['byte_identity_match']
        path = Path(cached_file(clip['repository'], record['filename'], local_files_only=True))
        assert path.resolve() == Path(record['resolved_local_path'])
        stat = fingerprint(path)
        assert stat['bytes'] == record['bytes']
        if record['filename'] != 'pytorch_model.bin':
            assert sha(path) == record['local_sha256']
        clip_files.append(dict(stat, sha256=record['local_sha256'],
            hash_scope='Current small file rehash' if record['filename'] != 'pytorch_model.bin'
                       else 'Shared concurrent POPE full-payload SHA audit; current path/size/stat bound here'))
    official_images = json.loads((HERE / 'official_dataset_identity.json').read_text())
    crc_by_name = {r['image']: r for r in official_images['image_crc_size_checks']}
    manifest_path = HERE / 'full_default_K128.images.json'
    if manifest_path.exists():
        images = json.loads(manifest_path.read_text())
        assert {Path(r['path']).name for r in images} == {q['image'] for q in questions}
        for record in images:
            require_stat(record)
        image_manifest_mode = 'Reuse prior one-pass local SHA256 + official CRC/size manifest; all stats unchanged'
    else:
        images = []
        for name in sorted({q['image'] for q in questions}):
            path = DATA / 'train_images' / name
            before = fingerprint(path)
            raw = path.read_bytes()
            official = crc_by_name[name]
            assert len(raw) == official['zip_size']
            assert zlib.crc32(raw) == official['zip_crc32']
            require_stat(before)
            images.append(dict(before, sha256=hashlib.sha256(raw).hexdigest()))
        image_manifest_mode = 'One-pass local SHA256 + official ZIP CRC/size; stable file stats'
    assert len(images) == 3166
    save(HERE / 'full_default_K128.images.json', images)
    evidence_files = ['current_weight_identity.json', 'current_small_model_identity.json',
                      'official_dataset_identity.json', 'full_default_K128.images.json',
                      'author_package_full_corpus_comparison.json',
                      'author_sentencepiece_causal_cpu_comparison.json',
                      'existing_beta1_metadata_audit.json', 'panel64_author_wait/result.json']
    plan = dict(prepared_utc=utc(), status='prepared_not_launched', n=5000,
        purpose='One previously unrun published-default K128 recipe, with disclosed same three stream waits; no parameter search.',
        author_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=AUTHOR, text=True).strip(),
        parameters=PARAMETERS, waits=WAITS, paper_argv_confirmed=False,
        source_comparisons=source_plan['source_comparisons'],
        model_path=str(MODEL), model_payloads=weights['files'],
        model_small_files=[dict(path=str(MODEL / r['file']), sha256=r['sha256']) for r in small['files']],
        questions=dict(path=str(question_path), sha256=sha(question_path)),
        annotation=dict(path=str(annotation_path), sha256=sha(annotation_path)),
        image_folder=str(DATA / 'train_images'), image_count=3166, image_manifest_mode=image_manifest_mode,
        clip_identity=dict(audit=dict(path=str(clip_audit_path), sha256=sha(clip_audit_path)),
            repository=clip['repository'], revision=clip['public_revision'], files=clip_files),
        worker=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        wait_helper=dict(path=str(HERE / 'repro_panel.py'), sha256=sha(HERE / 'repro_panel.py')),
        evidence=[dict(path=str(HERE / name), sha256=sha(HERE / name)) for name in evidence_files],
        prior_negative_control=dict(budget_cli=32, beta=1., n=5000, accuracy_percent=50.966,
            interpretation='Completed negative control preserved; not rerun or replaced.'),
        baseline_comparator=dict(budget_cli=128, beta=2., n=5000, accuracy_percent=57.866,
            interpretation='Compare protocol explicitly; do not select parameters according to accuracy.'),
        decision_rule='Report all5000 predictions and official score whatever the result. No score-based retry/tuning. Reassess unresolved paper residual honestly.',
        limits='Public shell defaults plus disclosed caller waits; not proof of the paper configuration. Full-image official identity is CRC32/size; only64 downloaded JPEGs have independent official SHA256 equivalence.',
        runtime_environment='Existing llava_pruner core author pins; whole environment not identical. CPU checks prove relevant Pillow/tokenizer input equivalence.',
        observation_only_fields=['Main LLM dtype is distinct from actual vision body, visual_projection, text_tower and mm_projector parameter dtypes',
            'torch default dtype and first actual generated input images dtype; no numerical casts or source/math changes added'],
        commands={'run_requires_parent_slot':[sys.executable, str(Path(__file__).resolve()), 'run', '--gpu-authorized'],
                  'verify_cpu':[sys.executable, str(Path(__file__).resolve()), 'verify']})
    save(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), n=5000, images=3166, CUDA_initialized=False, GPU_started=False)))


def check_plan(plan):
    assert all(os.environ.get(k) != '1' for k in ARCH_SWITCHES), 'Alternative pruning arch enabled'
    assert plan['parameters'] == PARAMETERS and plan['waits'] == WAITS
    assert plan['author_commit'] == 'e1a08801461c181871b9ff1cb0803b8e9966f083'
    for record in [plan['worker'], plan['wait_helper'], plan['questions'], plan['annotation'],
                   *plan['model_small_files'], *plan['evidence']]:
        assert sha(record['path']) == record['sha256'], ('hash changed', record['path'])
    for record in plan['source_comparisons']:
        assert sha(record['author_path']) == record['author_sha256']
    for record in plan['model_payloads']:
        require_stat(record)
    assert sha(plan['clip_identity']['audit']['path']) == plan['clip_identity']['audit']['sha256']
    for record in plan['clip_identity']['files']:
        require_stat(record)
    for record in json.loads((HERE / 'full_default_K128.images.json').read_text()):
        require_stat(record)


def verify(run_dir=RUN, write=True):
    """CPU-only full denominator, input alignment, runtime and official score gate."""
    plan = json.loads(PLAN.read_text())
    check_plan(plan)
    questions = rows(plan['questions']['path'])
    answers = rows(run_dir / 'answers.jsonl')
    runtime = rows(run_dir / 'runtime.jsonl')
    details = validate_rows(questions, answers, runtime)
    loaded = json.loads((run_dir / 'loaded_runtime.json').read_text())
    assert loaded['worker_sha256'] == plan['worker']['sha256'] and loaded['plan_sha256'] == sha(PLAN)
    actual = loaded['loaded']
    assert actual['alpha'] == .5 and actual['beta'] == 1. and actual['K'] == 128
    assert actual['backend'] == 'sdpa' and actual['dtype'] == 'torch.float16'
    assert actual['module_file'] == str(AUTHOR / 'LLaVA/llava/model/language_model/llava_llama.py')
    assert actual['builder_file'] == str(AUTHOR / 'LLaVA/llava/model/builder.py')
    assert actual['multimodal_method_file'] == str(AUTHOR / 'LLaVA/llava/model/llava_arch.py')
    assert actual['multimodal_method_sha256'] == sha(actual['multimodal_method_file'])
    assert set(actual['component_parameter_dtypes']) == {'vision_body','visual_projection','text_tower','mm_projector'}
    assert all(actual['component_parameter_dtypes'].values())
    assert loaded['first_generate_input']['images_dtype']
    assert loaded['torch_default_dtype']
    assert actual['clip_name'] == plan['clip_identity']['repository']
    if actual['clip_config_commit_hash'] is not None:
        assert actual['clip_config_commit_hash'] == plan['clip_identity']['revision']
    assert loaded['wait_patch']['inserted_statements'] == WAITS
    assert loaded['args']['anchorzip'] is False
    assert loaded['args']['model_path'] == plan['model_path']
    assert loaded['args']['question_file'] == plan['questions']['path']
    assert loaded['args']['image_folder'] == plan['image_folder']
    assert loaded['args']['answers_file'] == str(run_dir / 'answers.jsonl')
    for k in ['visual_token_num', 'alpha', 'beta', 'temperature', 'top_p', 'num_beams', 'max_new_tokens', 'conv_mode']:
        assert loaded['args'][k] == PARAMETERS[k], ('actual args mismatch', k)
    prompt_processor, evaluator_type = author_imports()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    decoded = tokenizer.batch_decode([r['output_token_ids'][0] for r in runtime], skip_special_tokens=True)
    assert [s.strip() for s in decoded] == [a['text'] for a in answers], 'raw answers do not decode from recorded output IDs'
    annotations = json.loads(Path(plan['annotation']['path']).read_text())['data']
    by_key = {(str(a['image_id']), a['question'].lower()): a for a in annotations}
    assert len(by_key) == len(annotations) == 5000
    pairs = [dict(pred_answer=a['text'], gt_answers=by_key[
        (str(a['question_id']), prompt_processor(a['prompt']))]['answers']) for a in answers]
    evaluator = evaluator_type()
    per_question = [evaluator.eval_pred_list([p]) for p in pairs]
    value = 100 * evaluator.eval_pred_list(pairs)
    assert abs(value - 100 * sum(per_question) / 5000) < 1e-10
    result = dict(origin='local_run', method='EADP', model='LLaVA-NeXT-7B',
        budget_cli=128, budget_nominal=640, metric='official TextVQA accuracy (%)',
        value=value, n=5000, token_decoding_checks=5000, parameters=PARAMETERS, paper_argv_confirmed=False,
        stream_wait_repair=WAITS, **{k:v for k,v in details.items() if k != 'n'},
        artifacts={name:dict(path=str(run_dir / name), sha256=sha(run_dir / name))
                   for name in ['answers.jsonl', 'runtime.jsonl', 'loaded_runtime.json']},
        plan=dict(path=str(PLAN), sha256=sha(PLAN)),
        annotation=plan['annotation'], evaluator=dict(path=inspect.getfile(evaluator_type), sha256=sha(inspect.getfile(evaluator_type))),
        limits=plan['limits'])
    if write:
        with (run_dir / 'per_question_scores.jsonl').open('w') as f:
            for i, (q, score) in enumerate(zip(questions, per_question)):
                f.write(json.dumps(dict(position=i, question_id=q['question_id'], image=q['image'],
                    prompt_sha256=hashlib.sha256(q['text'].encode()).hexdigest(), score=score)) + '\n')
        result['per_question_scores_sha256'] = sha(run_dir / 'per_question_scores.jsonl')
        save(run_dir / 'official.score.json', result)
    return result


def run(authorized):
    if not authorized:
        raise RuntimeError('Explicit parent GPU slot allocation required before --gpu-authorized')
    plan = json.loads(PLAN.read_text())
    check_plan(plan)
    RUN.mkdir(exist_ok=False)
    author_imports()
    import torch
    from repro_panel import install_wait
    native = AUTHOR / 'LLaVA/llava/eval/model_vqa_loader.py'
    namespace = runpy.run_path(str(native), run_name='eadp_full_default')
    fn = namespace['eval_model']
    original_loader = fn.__globals__['load_pretrained_model']
    args = argparse.Namespace(model_path=str(MODEL), model_base=None,
        image_folder=plan['image_folder'], question_file=plan['questions']['path'],
        answers_file=str(RUN / 'answers.jsonl'), num_chunks=1, chunk_idx=0,
        anchorzip=False, **{k:v for k,v in PARAMETERS.items()
                            if k not in ['budget_nominal', 'attention_backend', 'dtype']})
    questions = rows(plan['questions']['path'])
    observed = dict(entry_file=str(native), entry_sha256=sha(native),
        worker_sha256=sha(__file__), plan_sha256=sha(PLAN), args=vars(args),
        gpu=torch.cuda.get_device_name(0),
        torch_default_dtype=str(torch.get_default_dtype()),
        package_versions={k:importlib.metadata.version(k) for k in
            ['torch','transformers','accelerate','Pillow','tokenizers','sentencepiece','protobuf','safetensors','numpy']},
        backend_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic))
    position = 0
    trace = (RUN / 'runtime.jsonl').open('x')
    def loader(*a, **kw):
        tokenizer, model, processor, context = original_loader(*a, **kw)
        observed['loaded'] = dict(alpha=model.alpha, beta=model.beta, K=model.visual_token_num,
            backend=model.config._attn_implementation, dtype=str(model.dtype),
            module_file=inspect.getfile(type(model)), builder_file=inspect.getfile(original_loader),
            clip_name=model.get_vision_tower().vision_tower_name,
            clip_config_commit_hash=getattr(model.get_vision_tower().config, '_commit_hash', None))
        method_file = inspect.getfile(inspect.unwrap(model.prepare_inputs_labels_for_multimodal.__func__))
        assert method_file == str(AUTHOR / 'LLaVA/llava/model/llava_arch.py')
        observed['loaded'].update(multimodal_method_file=method_file,
            multimodal_method_sha256=sha(method_file), arch_environment={k:os.environ.get(k) for k in ARCH_SWITCHES})
        tower=model.get_vision_tower()
        observed['loaded']['component_parameter_dtypes']={name:parameter_dtypes(module) for name,module in {
            'vision_body':tower.vision_tower.vision_model,
            'visual_projection':tower.vision_tower.visual_projection,
            'text_tower':tower.text_tower,
            'mm_projector':model.get_model().mm_projector}.items()}
        assert observed['loaded']['backend'] == 'sdpa'
        observed['wait_patch'] = install_wait(model.get_vision_tower())
        assert observed['wait_patch']['inserted_statements'] == WAITS
        generate = model.generate
        def traced_generate(*ga, **gk):
            nonlocal position
            returned = generate(*ga, **gk)
            tokens, count = returned
            q = questions[position]
            if position == 0:
                observed['first_generate_input']=dict(images_dtype=str(gk['images'].dtype),
                    images_device=str(gk['images'].device),images_shape=list(gk['images'].shape),
                    scope='Actual tensor passed into native generation, before any native CLIP dtype cast')
                save(RUN / 'loaded_runtime.json', observed)
            trace.write(json.dumps(dict(position=position, question_id=q['question_id'], image=q['image'],
                prompt_sha256=hashlib.sha256(q['text'].encode()).hexdigest(),
                actual_visual_tokens_retained=int(count),
                image_tensor_shape=list(gk['images'].shape),
                images_dtype=str(gk['images'].dtype),
                output_token_ids=tokens.detach().cpu().tolist())) + '\n')
            trace.flush()
            position += 1
            return returned
        model.generate = traced_generate
        save(RUN / 'loaded_runtime.json', observed)
        return tokenizer, model, processor, context
    fn.__globals__['load_pretrained_model'] = loader
    fn.__globals__['args'] = args
    stat = Path('/proc/self/stat').read_text()
    save(RUN / 'state.json', dict(status='generating', pid=os.getpid(),
        startticks=int(stat[stat.rindex(')')+2:].split()[19]), started_utc=utc()))
    try:
        fn(args)
        trace.close()
        assert position == 5000
        result = verify()
        save(RUN / 'state.json', dict(status='complete', pid=os.getpid(), n=5000,
            finished_utc=utc(), score_sha256=sha(RUN / 'official.score.json')))
        print(json.dumps(result, ensure_ascii=False))
    finally:
        trace.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','verify'])
    parser.add_argument('--gpu-authorized', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'run':
            run(args.gpu_authorized)
        else:
            print(json.dumps(verify(), ensure_ascii=False))
    except BaseException as error:
        if args.action == 'run' and RUN.exists():
            save(RUN / 'failure.json', dict(type=type(error).__name__, error=str(error), utc=utc()))
        raise
