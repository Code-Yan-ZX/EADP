"""Prepare/run a fixed64 fresh-author vs prior-harness equivalence check.

Preparation is CPU-only. GPU execution requires the coordinating agent's slot.
The only runtime repair is the same three caller-stream dependencies already
used by the frozen reference. Author and prior source files stay unchanged.
"""
import argparse
import ast
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
import textwrap
import types

HERE = Path(__file__).resolve().parent
OLD = Path('/media/disk2/YZX/research/EADP_amp')
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
MODEL = Path('/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b')
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
DATA = OLD / 'LLaVA/playground/data/eval/textvqa'
PRIOR = OLD / 'Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/stability/next_text_K32_EADP_beta2_streamwait.jsonl'
CORE = ['llava/eval/model_vqa_loader.py', 'llava/eval/eval_textvqa.py', 'llava/eval/m4c_evaluator.py', 'llava/model/llava_arch.py', 'llava/model/multimodal_encoder/clip_encoder.py', 'llava/model/builder.py', 'llava/model/language_model/llava_llama.py', 'llava/mm_utils.py', 'llava/conversation.py', 'scripts/v1_6/eval/textvqa.sh', 'environment.yml', 'requirements.txt']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    Path(name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    question_path = HERE / 'author_linked_textvqa_questions.jsonl'
    assert question_path.read_bytes() == (DATA / 'llava_textvqa_val_v051_ocr.jsonl').read_bytes()
    questions = [json.loads(s) for s in question_path.read_text().splitlines() if s.strip()]
    indices = [int((i + .5) * len(questions) / 64) for i in range(64)]
    selected = [questions[i] for i in indices]
    previous_panel = json.loads((OLD / 'Qwen_vl/outputs/audit_followup_20261008/next_runtime_repair_20261009/panel64/protocol.json').read_text())
    assert indices == [r['position'] for r in previous_panel['samples']]
    original = [json.loads(s) for s in PRIOR.read_text().splitlines() if s.strip()]
    by_key = {(str(r['question_id']), r['prompt']): r for r in original}
    assert len(original) == len(by_key) == 5000
    references = [by_key[(str(q['question_id']), q['text'])] for q in selected]
    prior_runtime_path = Path(str(PRIOR) + '.runtime.jsonl')
    prior_runtime = [json.loads(s) for s in prior_runtime_path.read_text().splitlines() if s.strip()]
    assert len(prior_runtime) == 5000
    selected_runtime = [prior_runtime[i] for i in indices]
    assert all(r['question_position'] == i for r, i in zip(selected_runtime, indices))
    images = []
    for q, old in zip(selected, previous_panel['samples']):
        assert q['text'] == old['prompt'] and q['image'] == old['image']
        path = DATA / 'train_images' / q['image']
        assert sha(path) == old['image_sha256']
        images.append({'path': str(path), 'sha256': sha(path)})
    fresh_image_folder = HERE / 'fresh_panel_images'
    for q, image in zip(selected, images):
        assert sha(fresh_image_folder / q['image']) == image['sha256']
    for name, records in [('panel64.questions.jsonl', selected), ('panel64.prior_answers.jsonl', references)]:
        (HERE / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    save(HERE / 'panel64.prior_runtime.json', selected_runtime)
    comparisons = []
    for relative in CORE:
        fresh, prior = AUTHOR / 'LLaVA' / relative, OLD / 'LLaVA' / relative
        comparisons.append({'relative': relative, 'author_path': str(fresh), 'author_sha256': sha(fresh), 'prior_sha256': sha(prior), 'byte_identical': fresh.read_bytes() == prior.read_bytes()})
    assert all(r['byte_identical'] for r in comparisons)
    source_probe = "import inspect,json,torch;import llava.model.builder,llava.model.llava_arch,llava.eval.model_vqa_loader;print(json.dumps({'cuda_initialized':torch.cuda.is_initialized(),'module_files':{m:inspect.getfile(__import__(m,fromlist=[''])) for m in ['llava.model.builder','llava.model.llava_arch','llava.eval.model_vqa_loader']}}))"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(AUTHOR / 'LLaVA'))
    result = subprocess.run([PYTHON, '-c', source_probe], cwd=AUTHOR / 'LLaVA', env=env, text=True, capture_output=True, check=True)
    (HERE / 'cpu_import_probe.stdout').write_text(result.stdout)
    (HERE / 'cpu_import_probe.stderr').write_text(result.stderr)
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert not probe['cuda_initialized'] and all(str(AUTHOR) in p for p in probe['module_files'].values())
    spec = {'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'purpose': 'Source-entry equivalence, not full benchmark reproduction or parameter search.', 'author_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=AUTHOR, text=True).strip(), 'source_comparisons': comparisons, 'cpu_import_probe': probe, 'model_path': str(MODEL), 'model_config_sha256': sha(MODEL / 'config.json'), 'question_source': {'path': str(DATA / 'llava_textvqa_val_v051_ocr.jsonl'), 'sha256': sha(DATA / 'llava_textvqa_val_v051_ocr.jsonl')}, 'prior_predictions': {'path': str(PRIOR), 'sha256': sha(PRIOR)}, 'fixed_positions': indices, 'panel_question_sha256': sha(HERE / 'panel64.questions.jsonl'), 'panel_prior_answer_sha256': sha(HERE / 'panel64.prior_answers.jsonl'), 'images': images, 'parameters': {'budget_cli': 32, 'nominal_budget': 160, 'alpha': .5, 'beta': 2.0, 'max_new_tokens': 128, 'conv_mode': 'vicuna_v1', 'do_sample': False, 'backend': 'sdpa', 'dtype': 'float16'}, 'synchronization': 'Identical caller-stream wait on both image/text streams, matching frozen reference; no scorer/crop/selection math changes.', 'commands': {mode: [PYTHON, str(Path(__file__).resolve()), 'run', '--entry', mode, '--gpu-authorized'] for mode in ['author', 'prior']}, 'decision_rule': 'Run author64 first. If all answers/counts match frozen reference and imported sources are correct, source isolation did not repair the discrepancy; no need to regenerate prior64. Any divergence is retained and investigated, never a parameter-selection rule.'}
    save(HERE / 'source_equivalence_plan.json', spec)
    spec.update(fresh_question_source={'path': str(question_path), 'sha256': sha(question_path)}, prior_runtime_source={'path': str(prior_runtime_path), 'sha256': sha(prior_runtime_path)}, panel_prior_runtime_sha256=sha(HERE / 'panel64.prior_runtime.json'))
    spec['fresh_image_folder'] = str(fresh_image_folder)
    save(HERE / 'source_equivalence_plan.json', spec)
    print(json.dumps({'plan': str(HERE / 'source_equivalence_plan.json'), 'n': 64, 'core_source_files_equal': len(comparisons), 'CUDA_initialized': probe['cuda_initialized'], 'GPU_started': False}))


def install_wait(tower):
    original = inspect.unwrap(tower.forward.__func__)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef))
    nodes = ast.parse('caller_stream = torch.cuda.current_stream(device=self.device)\nimage_stream.wait_stream(caller_stream)\ntext_stream.wait_stream(caller_stream)').body
    def patch(statements):
        count = 0
        for index, node in enumerate(list(statements)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'text_stream' for t in node.targets):
                statements[index + 1:index + 1] = nodes
                count += 1
            for attr in ['body', 'orelse', 'finalbody']:
                if isinstance(getattr(node, attr, None), list):
                    count += patch(getattr(node, attr))
        return count
    assert patch(function.body) == 1
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(Path(__file__).resolve()), 'exec'), dict(original.__globals__), namespace)
    tower.forward = types.MethodType(namespace['forward'], tower)
    return {'original_source_file': inspect.getfile(original), 'original_source_sha256': sha(inspect.getfile(original)), 'inserted_statements': [ast.unparse(n) for n in nodes], 'source_file_modified': False}


def run(entry, authorized):
    if not authorized:
        raise RuntimeError('Coordinate GPU slot with the parent agent before --gpu-authorized')
    import torch
    plan = json.loads((HERE / 'source_equivalence_plan.json').read_text())
    assert sha(HERE / 'panel64.questions.jsonl') == plan['panel_question_sha256']
    assert sha(HERE / 'panel64.prior_runtime.json') == plan['panel_prior_runtime_sha256']
    for source in plan['source_comparisons']:
        assert sha(source['author_path']) == source['author_sha256']
    run_dir = HERE / ('panel64_' + entry + '_wait')
    run_dir.mkdir(exist_ok=False)
    source_root = AUTHOR if entry == 'author' else OLD
    sys.path[:] = [str(source_root / 'LLaVA')] + [p for p in sys.path if '/EADP_amp' not in p and '/_reproduction/author_eadp/LLaVA' not in p]
    native = source_root / ('LLaVA/llava/eval/model_vqa_loader.py' if entry == 'author' else 'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_eval_arm_loader.py')
    namespace = runpy.run_path(str(native), run_name='eadp_source_equivalence')
    fn = namespace['eval_model']
    original_loader = fn.__globals__['load_pretrained_model']
    args = argparse.Namespace(model_path=str(MODEL), model_base=None, image_folder=plan['fresh_image_folder'], question_file=str(HERE / 'panel64.questions.jsonl'), answers_file=str(run_dir / 'answers.jsonl'), conv_mode='vicuna_v1', num_chunks=1, chunk_idx=0, temperature=0.0, top_p=None, num_beams=1, max_new_tokens=128, visual_token_num=32, beta=2.0, alpha=.5, anchorzip=False)
    observation = {'entry': entry, 'entry_file': str(native), 'entry_sha256': sha(native), 'worker_sha256': sha(Path(__file__)), 'plan_sha256': sha(HERE / 'source_equivalence_plan.json'), 'args': vars(args), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'), 'runtime': []}
    observation['package_versions'] = {name: importlib.metadata.version(name) for name in ['Pillow', 'tokenizers', 'sentencepiece', 'protobuf', 'safetensors', 'accelerate', 'numpy']}
    observation['CUDA_backend_flags'] = {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}
    def loader(*a, **kwargs):
        tokenizer, model, processor, context = original_loader(*a, **kwargs)
        observation['loaded'] = {'class': type(model).__module__ + '.' + type(model).__name__, 'dtype': str(model.dtype), 'backend': model.config._attn_implementation, 'alpha': model.alpha, 'beta': model.beta, 'K': model.visual_token_num, 'module_file': inspect.getfile(type(model)), 'builder_file': inspect.getfile(original_loader)}
        assert observation['loaded']['backend'] == 'sdpa'
        assert str(source_root / 'LLaVA') in observation['loaded']['module_file']
        observation['wait_patch'] = install_wait(model.get_vision_tower())
        generate = model.generate
        def measured_generate(*ga, **gk):
            result = generate(*ga, **gk)
            token_ids, count = result
            # All GPU reads occur after native generation returns.
            observation['runtime'].append({'position': len(observation['runtime']), 'retained_return_value': int(count), 'output_token_ids': token_ids.detach().cpu().tolist()})
            save(run_dir / 'runtime.json', observation['runtime'])
            return result
        model.generate = measured_generate
        save(run_dir / 'loaded_runtime.json', {k: v for k, v in observation.items() if k != 'runtime'})
        return tokenizer, model, processor, context
    fn.__globals__['load_pretrained_model'] = loader
    fn.__globals__['args'] = args
    started = datetime.datetime.now(datetime.timezone.utc)
    save(run_dir / 'state.json', {'status': 'generating', 'started_utc': started.isoformat(), 'pid': os.getpid()})
    fn(args)
    answers = [json.loads(s) for s in (run_dir / 'answers.jsonl').read_text().splitlines() if s.strip()]
    reference = [json.loads(s) for s in (HERE / 'panel64.prior_answers.jsonl').read_text().splitlines() if s.strip()]
    assert len(answers) == len(reference) == len(observation['runtime']) == 64
    assert [(str(r['question_id']), r['prompt']) for r in answers] == [(str(r['question_id']), r['prompt']) for r in reference]
    prior_runtime = json.loads((HERE / 'panel64.prior_runtime.json').read_text())
    count_changes = [{'position': i, 'old': old['actual_visual_tokens_retained'], 'new': new['retained_return_value']} for i, (old, new) in enumerate(zip(prior_runtime, observation['runtime'])) if old['actual_visual_tokens_retained'] != new['retained_return_value']]
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    annotation_path = HERE / 'TextVQA_0.5.1_val.fresh.json'
    assert annotation_path.read_bytes() == (DATA / 'TextVQA_0.5.1_val.json').read_bytes()
    annotations = json.loads(annotation_path.read_text())['data']
    by_key = {(a['image_id'], a['question'].lower()): a for a in annotations}
    evaluator = TextVQAAccuracyEvaluator()
    def scores(rows):
        return [evaluator.eval_pred_list([{'pred_answer': r['text'], 'gt_answers': by_key[(r['question_id'], prompt_processor(r['prompt']))]['answers']}]) for r in rows]
    fresh, prior = scores(answers), scores(reference)
    changed = [{'position': i, 'question_id': a['question_id'], 'old': b['text'], 'new': a['text'], 'old_score': prior[i], 'new_score': fresh[i]} for i, (a, b) in enumerate(zip(answers, reference)) if a['text'] != b['text']]
    result = {'n': 64, 'source_entry': entry, 'accuracy_percent': 100 * sum(fresh) / 64, 'frozen_reference_accuracy_percent': 100 * sum(prior) / 64, 'answer_changes': changed, 'n_answer_changes': len(changed), 'loaded': observation['loaded'], 'answer_sha256': sha(run_dir / 'answers.jsonl'), 'reference_sha256': sha(HERE / 'panel64.prior_answers.jsonl'), 'annotation': {'path': str(annotation_path), 'sha256': sha(annotation_path)}, 'evaluator': {'path': inspect.getfile(TextVQAAccuracyEvaluator), 'sha256': sha(inspect.getfile(TextVQAAccuracyEvaluator))}, 'limits': 'Fixed64 source-equivalence diagnostic; not full5000 accuracy and not recovery of paper residual.'}
    result.update(retained_count_changes=count_changes, n_retained_count_changes=len(count_changes), prior_runtime_sha256=sha(HERE / 'panel64.prior_runtime.json'))
    save(run_dir / 'result.json', result)
    save(run_dir / 'state.json', {'status': 'complete', 'pid': os.getpid(), 'n': 64, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'result_sha256': sha(run_dir / 'result.json')})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--entry', choices=['author', 'prior'], default='author')
    p.add_argument('--gpu-authorized', action='store_true')
    args = p.parse_args()
    try:
        prepare() if args.action == 'prepare' else run(args.entry, args.gpu_authorized)
    except BaseException as error:
        directory = HERE / ('panel64_' + args.entry + '_wait')
        if args.action == 'run' and directory.is_dir():
            save(directory / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
