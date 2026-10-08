"""Read-only CPU provenance and official-loader input identity audit.

Writes only within this diagnosis directory. Never constructs a model or uses
CUDA; large checkpoints use size/header identity to avoid urgent-run I/O load.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OFFICIAL = Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official')
MODEL = Path('/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b')
SNAPSHOT = Path('/home/dell/.cache/huggingface/hub/models--liuhaotian--llava-v1.6-vicuna-7b/snapshots/deae57a8c0ccb0da4c2661cc1891cc9d06503d11')
CLIP_SNAPSHOT = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
OUT = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for part in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import sys
    sys.path.insert(0, str(ROOT / 'LLaVA'))
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTokenizerFast
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates

    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    report = {'cpu_only': True, 'source_files': {}, 'checkpoint_files': {}}
    for rel in ('LLaVA/llava/model/llava_arch.py', 'LLaVA/llava/model/builder.py',
                'LLaVA/llava/model/multimodal_encoder/clip_encoder.py',
                'LLaVA/llava/mm_utils.py', 'LLaVA/llava/conversation.py',
                'LLaVA/llava/model/language_model/llava_llama.py',
                'LLaVA/llava/eval/model_vqa_loader.py'):
        report['source_files'][rel] = dict(local_sha256=sha(ROOT/rel), official_sha256=sha(OFFICIAL/rel), byte_identical=(ROOT/rel).read_bytes()==(OFFICIAL/rel).read_bytes())

    local_cfg = json.loads((MODEL/'config.json').read_text())
    cached_cfg = json.loads((SNAPSHOT/'config.json').read_text())
    report['config_semantic_differences'] = {k: {'local':local_cfg.get(k), 'hf_snapshot':cached_cfg.get(k)} for k in set(local_cfg)|set(cached_cfg) if local_cfg.get(k)!=cached_cfg.get(k)}
    assert report['config_semantic_differences'] == {'model_type': {'local':'llava_llama','hf_snapshot':'llava'}}

    api_url = 'https://huggingface.co/api/models/liuhaotian/llava-v1.6-vicuna-7b/revision/deae57a8c0ccb0da4c2661cc1891cc9d06503d11?blobs=true'
    expected_remote = {}
    try:
        with urllib.request.urlopen(api_url, timeout=20) as response:
            metadata = json.load(response)
        expected_remote = {r['rfilename']: r.get('lfs',{}).get('sha256') for r in metadata['siblings']}
        report['remote_hf_api'] = {'url':api_url, 'sha':metadata['sha'], 'verified':True}
        (OUT/'hf_revision_metadata.json').write_text(json.dumps(metadata,indent=2))
    except Exception as exc:
        report['remote_hf_api'] = {'url':api_url,'verified':False,'error':f'{type(exc).__name__}: {exc}'}
    for file in sorted(MODEL.iterdir()):
        counterpart = SNAPSHOT/file.name
        if not counterpart.is_file():
            continue
        expected_name = counterpart.resolve().name
        lfs_expected = expected_remote.get(file.name) or (expected_name if len(expected_name)==64 else None)
        if file.stat().st_size > 10 * 1024 * 1024:
            def header_sha(path):
                with path.open('rb') as handle:
                    raw_len=handle.read(8)
                    header_len=int.from_bytes(raw_len,'little')
                    assert header_len<10*1024*1024
                    return hashlib.sha256(raw_len+handle.read(header_len)).hexdigest()
            report['checkpoint_files'][file.name]={'size':file.stat().st_size,'hf_snapshot_size':counterpart.stat().st_size,'resolved_local_path':str(file.resolve()),'resolved_hf_path':str(counterpart.resolve()),'same_inode':file.stat().st_ino==counterpart.stat().st_ino,'safetensors_header_equal':header_sha(file)==header_sha(counterpart),'hf_snapshot_blob_id':expected_name,'lfs_sha256':lfs_expected,'remote_lfs_verified':expected_remote.get(file.name)==lfs_expected,'full_payload_sha_verified':False,'boundary':'Avoid reading 14GB concurrently with urgent GPU queues; equal size/header does not prove weight payload identity'}
            print('Checkpoint metadata:',file.name,'header_same=',report['checkpoint_files'][file.name]['safetensors_header_equal'],flush=True)
            continue
        actual_sha = sha(file)
        if lfs_expected:
            equal = actual_sha==lfs_expected
        else:
            equal = file.read_bytes()==counterpart.read_bytes()
        report['checkpoint_files'][file.name] = {'size':file.stat().st_size,'local_sha256':actual_sha,'hf_snapshot_blob_id':expected_name,'lfs_sha256':lfs_expected,'remote_lfs_verified':expected_remote.get(file.name)==lfs_expected if lfs_expected else None,'byte_identical_to_hf_snapshot':equal}
        print('Checkpoint:',file.name,'same=',equal,flush=True)
    assert all(r.get('byte_identical_to_hf_snapshot',r.get('safetensors_header_equal')) for k,r in report['checkpoint_files'].items() if k!='config.json')

    question_path = ROOT/'LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl'
    questions = [json.loads(l) for l in question_path.open()]
    with zipfile.ZipFile(ROOT/'LLaVA/playground/data/eval/eval.zip') as archive:
        entry = 'textvqa/llava_textvqa_val_v051_ocr.jsonl'
        contents = archive.read(entry)
        report['question_eval_zip_identity'] = {'entry':entry,'sha256':hashlib.sha256(contents).hexdigest(),'local_sha256':sha(question_path),'byte_identical':contents==question_path.read_bytes()}
    assert report['question_eval_zip_identity']['byte_identical']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_SNAPSHOT, local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(CLIP_SNAPSHOT, local_files_only=True)
    config = SimpleNamespace(**local_cfg)
    loader = load_module(OFFICIAL/'LLaVA/llava/eval/model_vqa_loader.py','official_loader_readonly_audit')
    loader.args = SimpleNamespace(conv_mode='vicuna_v1')
    dataset = loader.CustomDataset(questions,str(ROOT/'LLaVA/playground/data/eval/textvqa/train_images'),tokenizer,processor,config)

    panel = json.loads((ROOT/'Qwen_vl/outputs/audit_followup_20261008/textvqa_guidance_manifest.json').read_text())['samples']
    panel_rows = {int(r['dataset_row']) for r in panel}
    matches = []
    lengths = []
    segments = {}
    for position,row in enumerate(questions):
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0],'<image>\n'+row['text'])
        conv.append_message(conv.roles[1],None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt,tokenizer,return_tensors='pt').unsqueeze(0)
        lengths.append(len(input_ids[0]))
        guide = row['text'].replace('\nAnswer the question using a single word or phrase.','')
        segment_count = (len(clip_tokenizer(guide).input_ids)-1)//77+1
        segments[str(segment_count)] = segments.get(str(segment_count),0)+1
        if position in panel_rows:
            original = Image.open(ROOT/'LLaVA/playground/data/eval/textvqa/train_images'/row['image']).convert('RGB')
            ours = process_images([original],processor,config)[0].unsqueeze(0)
            ids_off,pixels_off,size_off = loader.collate_fn([dataset[position]])
            matches.append({'position':position,'question_id':row['question_id'],'image':row['image'],'image_sha256':sha(ROOT/'LLaVA/playground/data/eval/textvqa/train_images'/row['image']),'size':list(original.size),'llm_input_ids_equal':torch.equal(input_ids,ids_off),'pixels_equal':torch.equal(ours,pixels_off),'pixels_shape':list(ours.shape),'size_value_equal':tuple(original.size)==tuple(size_off[0]),'float16_pixels_equal':torch.equal(ours.half(),pixels_off.half()),'pixels_sha256':hashlib.sha256(ours.numpy().tobytes()).hexdigest()})
    report['cpu_panel_official_loader_identity'] = {'selection':'Reuse prior seed20261008 CLIP length-stratified panel, selected without answers or predictions','samples':matches,'all_equal':all(r['llm_input_ids_equal'] and r['pixels_equal'] and r['float16_pixels_equal'] and r['size_value_equal'] for r in matches),'execution_differences':['Official DataLoader 4 workers, custom wrapper serial PIL preprocessing','Official non_blocking=True GPU transfers, wrapper half().cuda() blocking transfers','Official image_sizes is ((w,h),), wrapper [(w,h)]; identical indexed values','Official metadata empty, wrapper protocol metadata and runtime observer; observer returns same object']}
    assert report['cpu_panel_official_loader_identity']['all_equal']
    report['all_5000_text_identity']={'n_questions':len(questions),'n_unique_images':len({r['image'] for r in questions}),'guidance_suffix_removed':True,'clip_segment_distribution':segments,'llm_token_lengths':{'minimum':min(lengths),'maximum':max(lengths)},'full_multimodal_length_over4096':sum(l-1+2880>4096 for l in lengths),'k32_nominal_multimodal_length_over4096':sum(l-1+160>4096 for l in lengths)}
    report['fixed_generation']={'conv_mode':'vicuna_v1','temperature':0,'do_sample':False,'num_beams':1,'top_p':None,'max_new_tokens':128,'use_cache':True,'image_tensors_dtype':'float16','model_loader_torch_dtype':'float16','model_loader_use_flash_attn':False,'alpha':.5,'beta':2.,'AZ_install_baseline':False}
    report['release_full_merge']={'mm_patch_merge_config':'spatial_unpad','unconditional_effective_merge':'spatial','n_crops':5,'clip_tokens_per_crop':576,'full_visual_tokens':2880,'unpad':False,'newline_tokens':False,'matches_official_release':True,'paper_full_vanilla_identity_unproven':True}
    report['cuda_initialized'] = torch.cuda.is_initialized()
    assert not report['cuda_initialized']
    (OUT/'identity_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'report':str(OUT/'identity_report.json'),'checkpoint_metadata_mismatches':[k for k,r in report['checkpoint_files'].items() if not r.get('byte_identical_to_hf_snapshot',r.get('safetensors_header_equal'))],'large_payload_sha_verified':False,'panel_equal':report['cpu_panel_official_loader_identity']['all_equal'],'all_5000':report['all_5000_text_identity']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
