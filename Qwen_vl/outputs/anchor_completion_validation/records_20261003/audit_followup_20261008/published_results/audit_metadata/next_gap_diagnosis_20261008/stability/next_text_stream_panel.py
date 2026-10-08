"""Fixed NeXT stream-dependency mechanism panel; --prepare is CPU-only.

Nothing under the frozen model/wrapper/queue is edited. --run is an explicit
GPU operation to be invoked by the coordinator only after acquiring a slot.
This panel does not estimate the full TextVQA benchmark accuracy effect.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = Path('/media/disk2/YZX/research/EADP_amp')
LLAVA = ROOT / 'LLaVA'
AUDIT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
HELPER = AUDIT / 'rerun_fast/run_pope_streamwait_control.py'
ARCHIVE = LLAVA / 'playground/data/eval/anchorzip_p3/next_textvqa_official_20261008'
BASE = ARCHIVE / 'EGATHER_K32.jsonl.protocol.json'
SUFFIX = '\nAnswer the question using a single word or phrase.'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def patch_module():
    spec = importlib.util.spec_from_file_location('frozen_pope_wait_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.forward_patch()


def key(row):
    return row['question_id'], row.get('prompt', row.get('text'))


def paths(n, method, attempt=1):
    stem = f'next_text_K32_{method}_stream_panel{n}' + (f'_attempt{attempt}' if attempt != 1 else '')
    return dict(manifest=OUT / f'{stem}.manifest.json',
                protocol=OUT / f'{stem}.protocol.json',
                records=OUT / f'{stem}.records.jsonl',
                result=OUT / f'{stem}.result.json',
                state=OUT / f'{stem}.state.json', log=OUT / f'{stem}.log')


def prepare(n, method, trace_indices, attempt=1):
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE.read_text())
    if (base['question_rows'] != 5000 or base['visual_token_num'] != 32
            or base['alpha'] != .5 or base['beta'] != 2.
            or base['max_new_tokens'] != 128 or base['temperature'] != 0):
        raise RuntimeError('Frozen K32 baseline settings changed')
    questions_path = Path(base['question_file'])
    questions = [json.loads(line) for line in questions_path.open()]
    if len(questions) != 5000 or sha(questions_path) != base['question_file_sha256']:
        raise RuntimeError('Frozen official TextVQA input changed')
    if n < 1 or n > len(questions):
        raise ValueError('Panel size outside corpus')
    # Deterministic equal-width interval midpoints; neither GT nor output used.
    selected = [((2 * i + 1) * len(questions)) // (2 * n) for i in range(n)]
    samples = []
    for panel_position, dataset_position in enumerate(selected):
        row = questions[dataset_position]
        if not row['text'].endswith(SUFFIX):
            raise RuntimeError('Question lacks official answer suffix')
        order = (['as_is_1', 'wait_1', 'as_is_2', 'wait_2']
                 if panel_position % 2 == 0 else
                 ['wait_1', 'as_is_1', 'wait_2', 'as_is_2'])
        samples.append(dict(panel_position=panel_position, dataset_position=dataset_position,
                            question_id=row['question_id'], image=row['image'],
                            prompt=row['text'], prompt_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                            image_sha256=sha(Path(base['image_folder']) / row['image']), order=order))
    p = paths(n, method, attempt)
    manifest = dict(selection='Equal-width interval midpoint system sample from original 5000-row order; no GT/prediction used',
                    n=n, corpus_n=5000, question_file=str(questions_path),
                    question_file_sha256=sha(questions_path), samples=samples,
                    order='Even rows as-is/wait/as-is/wait, odd rows wait/as-is/wait/as-is',
                    composite_key=['question_id (image ID)', 'full original prompt'])
    if p['manifest'].exists():
        if json.loads(p['manifest'].read_text()) != manifest:
            raise RuntimeError('Existing fixed manifest differs; preserve it')
    else:
        dump(p['manifest'], manifest)
    frozen = {str(LLAVA / rel): digest for rel, digest in base['source_sha256'].items()}
    frozen.update({str(questions_path): sha(questions_path), str(BASE): sha(BASE),
                   str(HELPER): sha(HELPER),
                   str(ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_eval_arm_model_vqa.py'): base['wrapper_sha256']})
    for source in [LLAVA/'playground/data/eval/textvqa/TextVQA_0.5.1_val.json',
                   LLAVA/'llava/eval/eval_textvqa.py', LLAVA/'llava/eval/m4c_evaluator.py',
                   Path(base['model_path'])/'config.json', Path(base['model_path'])/'generation_config.json',
                   ARCHIVE/('EGATHER_K32.jsonl' if method=='EADP' else 'LRMAIN00625.jsonl')]:
        frozen[str(source)] = sha(source)
    if method == 'AZ':
        for rel in ['llava/model/llava_arch_anchorzip.py', '../Qwen_vl/scripts/anchor_merge_pilot/amp_common.py']:
            frozen[str((LLAVA / rel).resolve())] = sha(LLAVA / rel)
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source identity changed: ' + path)
    module, validation = patch_module()
    compile(module, 'fixed_neXT_wait_patch', 'exec')
    command = [PYTHON, str(Path(__file__).resolve()), '--run', '--n', str(n), '--method', method, '--attempt', str(attempt)]
    if not trace_indices:
        command.append('--no-index-trace')
    protocol = dict(protocol='NeXT_TextVQA_K32_fixed_stream_dependency_panel', prepared_utc=now(),
                    method=method, attempt=attempt, n=n, n_generations=4*n, manifest=str(p['manifest']), manifest_sha256=sha(p['manifest']),
                    parameters={k: base[k] for k in ['model_path','image_folder','conv_mode','visual_token_num','alpha','beta','temperature','top_p','num_beams','max_new_tokens']},
                    official_guidance='Remove exact answer suffix for texts= only; preserve full LLM prompt',
                    source_and_input_sha256=frozen, worker_sha256=sha(__file__), AST_validation=validation,
                    index_trace=trace_indices,
                    observation='Only after greed_select returns: selected index/quotas read to CPU; no vision/importance references kept' if trace_indices else 'Generate return count and decoded output only, read after generation',
                    observation_limit='Post-selection CPU reads can alter later allocator/timing; absence of failures does not prove stability',
                    pre_vision_GPU_hash_or_sync=False, new_record_stream=False, production_modified=False,
                    GPU_launch_authorized_by_prepare=False, generate_command=command,
                    output_paths={k:str(v) for k,v in p.items()},
                    interpretation='Mechanism panel; panel score change is not a 5000-question effect estimate')
    # A running or completed artifact must retain its exact protocol.
    if p['records'].exists() or p['result'].exists():
        previous = json.loads(p['protocol'].read_text())
        if previous['worker_sha256'] != protocol['worker_sha256']:
            raise RuntimeError('Existing artifact uses another worker revision')
        return previous, manifest, p
    dump(p['protocol'], protocol)
    return protocol, manifest, p


def safe_slot():
    entries = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip().splitlines()
    models = []
    exempt = []
    for entry in entries:
        pid, memory = entry.split(',')
        pid = int(pid.strip())
        memory_mib = int(memory.strip().split()[0])
        cmdline_path = Path('/proc') / str(pid) / 'cmdline'
        command = cmdline_path.read_bytes().replace(b'\0', b' ').decode(errors='replace') if cmdline_path.exists() else ''
        record = dict(pid=pid, used_memory=memory.strip(), command=command)
        if memory_mib == 0 and command.startswith('/usr/local/bin/ollama runner '):
            exempt.append(record)
        else:
            # Unknown or small CUDA contexts consume a slot conservatively.
            models.append(record)
    if len(models) > 1:
        raise RuntimeError('No model slot: two non-exempt CUDA processes already exist')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(gpu_rows) != 1 or 'A40' not in gpu_rows[0].split(',')[0]:
        raise RuntimeError('Expected the single authorized A40')
    free_mib = int(gpu_rows[0].split(',')[1].strip().split()[0])
    if free_mib < 22500:
        raise RuntimeError(f'Insufficient free memory for diagnostic: {free_mib} MiB < 22500 MiB')
    return dict(other_gpu_models=models, explicit_zero_memory_ollama_exemption=exempt,
                free_memory_mib=free_mib, required_free_memory_mib=22500, observed_utc=now())


def run(protocol, manifest, p):
    if p['records'].exists() or p['result'].exists():
        raise FileExistsError('Preserve existing panel artifacts; choose another panel size/method')
    slot = safe_slot()
    for path, digest in protocol['source_and_input_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source/input changed before generation: ' + path)
    for sample in manifest['samples']:
        if sha(Path(protocol['parameters']['image_folder']) / sample['image']) != sample['image_sha256']:
            raise RuntimeError('Image bytes changed')
    for name in ['USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION']:
        if os.environ.get(name) == '1':
            raise RuntimeError('Unexpected architecture environment: ' + name)
    sys.path.insert(0, str(LLAVA))
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.model import llava_arch as LA
    from llava.model.multimodal_encoder import clip_encoder
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.utils import disable_torch_init
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    import types
    original_forward = clip_encoder.CLIPVisionTower.forward
    module, validation = patch_module()
    if validation != protocol['AST_validation']:
        raise RuntimeError('Wait-only AST changed')
    namespace = dict(clip_encoder.__dict__)
    exec(compile(module, 'NeXT_fixed_stream_wait', 'exec'), namespace)
    patched_forward = namespace['forward']
    disable_torch_init()
    settings = protocol['parameters']
    tokenizer, model, image_processor, _ = load_pretrained_model(settings['model_path'], None,
        get_model_name_from_path(settings['model_path']), visual_token_num=32, alpha=.5, beta=2.)
    if protocol['method'] == 'AZ':
        sys.path.insert(0, str(LLAVA / 'llava/model'))
        sys.path.insert(0, str(ROOT / 'Qwen_vl/scripts/anchor_merge_pilot'))
        import llava_arch_anchorzip as AZ
        AZ.MODE = 'rtg'
        AZ.install()
    captures = {}
    original_greed = model.greed_select
    if protocol['index_trace']:
        def traced_greed(self, *args, **kwargs):
            idx, quotas = original_greed(*args, **kwargs)
            # Vision consumption is already complete by this stage. Never
            # inspect inputs/features or synchronize before vision forward.
            index_cpu = idx.detach().cpu().tolist()
            quota_cpu = quotas.detach().cpu().tolist()
            selected = [sorted(index_cpu[i][:q]) for i,q in enumerate(quota_cpu)]
            captures.update(quotas=quota_cpu, keep_sha256=hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest())
            return idx, quotas
        model.greed_select = types.MethodType(traced_greed, model)
    annotations_path = LLAVA / 'playground/data/eval/textvqa/TextVQA_0.5.1_val.json'
    annotations = {(r['image_id'],r['question'].lower()):r for r in json.loads(annotations_path.read_text())['data']}
    evaluator = TextVQAAccuracyEvaluator()
    archive_path = ARCHIVE / ('EGATHER_K32.jsonl' if protocol['method']=='EADP' else 'LRMAIN00625.jsonl')
    archive = {key(r):r['text'] for r in (json.loads(line) for line in archive_path.open())}
    records = []
    run_metadata = dict(started_utc=now(), pid=os.getpid(), concurrency=slot,
                        archive=str(archive_path), archive_sha256=sha(archive_path),
                        GT_sha256=sha(annotations_path), torch=torch.__version__,
                        transformers=__import__('transformers').__version__)
    dump(p['state'], dict(status='generating', complete=False, completed_samples=0, **run_metadata))
    with p['records'].open('x') as output:
        for sample in manifest['samples']:
            full_prompt = sample['prompt']
            qs = (DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN)+'\n'+full_prompt
            conv = conv_templates[settings['conv_mode']].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            # CPU token/image preparation is reused; each arm performs the exact
            # original blocking .cuda() transfers immediately before generate.
            ids_cpu = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            image = Image.open(Path(settings['image_folder']) / sample['image']).convert('RGB')
            image_cpu = process_images([image], image_processor, model.config)[0]
            annotation = annotations[(sample['question_id'],prompt_processor(full_prompt))]
            row = dict(panel_position=sample['panel_position'], dataset_position=sample['dataset_position'],
                       question_id=sample['question_id'], image=sample['image'], prompt=full_prompt,
                       prompt_sha256=sample['prompt_sha256'], image_sha256=sample['image_sha256'],
                       order=sample['order'], arms={})
            for variant in sample['order']:
                clip_encoder.CLIPVisionTower.forward = patched_forward if variant.startswith('wait') else original_forward
                captures.clear()
                input_ids = ids_cpu.unsqueeze(0).cuda()
                with torch.inference_mode():
                    output_ids, actual_count = model.generate(input_ids,
                        images=image_cpu.unsqueeze(0).half().cuda(), image_sizes=[image.size],
                        texts=full_prompt.replace(SUFFIX,''), do_sample=False,
                        temperature=0, top_p=None, num_beams=1, max_new_tokens=128, use_cache=True)
                prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                score = float(evaluator.eval_pred_list([dict(pred_answer=prediction,gt_answers=annotation['answers'])]))
                row['arms'][variant] = dict(text=prediction, actual_visual_tokens_retained=int(actual_count),
                    generation_ids_sha256=hashlib.sha256(json.dumps(output_ids.detach().cpu().tolist()).encode()).hexdigest(),
                    score=score, matches_full_archive=prediction==archive[(sample['question_id'],full_prompt)], **captures)
            row['same_input_prediction_disagreement'] = {name:row['arms'][name+'_1']['text']!=row['arms'][name+'_2']['text'] for name in ('as_is','wait')}
            row['same_input_count_disagreement'] = {name:row['arms'][name+'_1']['actual_visual_tokens_retained']!=row['arms'][name+'_2']['actual_visual_tokens_retained'] for name in ('as_is','wait')}
            if protocol['index_trace']:
                row['same_input_keep_disagreement'] = {name:row['arms'][name+'_1']['keep_sha256']!=row['arms'][name+'_2']['keep_sha256'] for name in ('as_is','wait')}
            records.append(row)
            output.write(json.dumps(row,ensure_ascii=False)+'\n')
            output.flush()
            dump(p['state'], dict(status='generating',complete=False,completed_samples=len(records),total_samples=manifest['n'], **run_metadata))
            print(json.dumps(dict(panel_position=row['panel_position'], dataset_position=row['dataset_position'],
                prediction_disagreement=row['same_input_prediction_disagreement'], count_disagreement=row['same_input_count_disagreement'])),flush=True)
    clip_encoder.CLIPVisionTower.forward = original_forward
    summary = dict(n=len(records), n_generations=4*len(records),
        repeated_prediction_disagreements={name:sum(r['same_input_prediction_disagreement'][name] for r in records) for name in ('as_is','wait')},
        repeated_count_disagreements={name:sum(r['same_input_count_disagreement'][name] for r in records) for name in ('as_is','wait')},
        panel_accuracy_percent={variant:100*sum(r['arms'][variant]['score'] for r in records)/len(records) for variant in ['as_is_1','wait_1','as_is_2','wait_2']},
        archive_mismatches={variant:sum(not r['arms'][variant]['matches_full_archive'] for r in records) for variant in ['as_is_1','wait_1','as_is_2','wait_2']},
        interpretation='Fixed small mechanism panel; does not establish the full benchmark accuracy effect')
    if protocol['index_trace']:
        summary['repeated_keep_disagreements']={name:sum(r['same_input_keep_disagreement'][name] for r in records) for name in ('as_is','wait')}
    dump(p['result'],dict(success=True,complete=True,finished_utc=now(),protocol=str(p['protocol']),
        protocol_sha256=sha(p['protocol']), manifest_sha256=sha(p['manifest']),records_sha256=sha(p['records']),
        run_metadata=run_metadata,summary=summary))
    dump(p['state'], dict(status='complete',complete=True,completed_samples=len(records), **run_metadata))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--n',type=int,default=128)
    parser.add_argument('--method',choices=['EADP','AZ'],default='EADP')
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--no-index-trace',dest='trace_indices',action='store_false',default=True)
    args=parser.parse_args()
    if args.attempt < 1:
        parser.error('--attempt must be positive')
    protocol,manifest,p=prepare(args.n,args.method,args.trace_indices,args.attempt)
    if args.run:
        run(protocol,manifest,p)
    else:
        print(json.dumps(dict(prepared=True,CPU_only=True,protocol=str(p['protocol']),
            n=args.n,n_generations=4*args.n,generate_command=protocol['generate_command'],
            restoration_ast_identical=protocol['AST_validation']['restoration_ast_identical']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
