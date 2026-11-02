#!/usr/bin/env python3
"""Prepare all4377 GT-free MMB runtime inputs; never load a model or score."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MMB = HERE.parent
ROOT = MMB.parents[2]
AUTHOR = ROOT / '_reproduction/author_eadp/LLaVA'
MODEL = Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b')
DATA = MMB / 'data/mmbench_dev_20230712.tsv'
SOURCE = AUTHOR / 'llava/eval/model_vqa_mmbench.py'
OLD = MMB / 'streamwait_control'
FIXED_GROUP_IDS = [72, 217, 370, 498, 632, 767, 909, 1030, 1143, 1255, 1364, 1480, 1591, 1706, 1821, 1936]
FIXED = MMB / 'question_only_guidance_panel/inputs/manifest.json'
FIXED_SHA = '920a465f64e220e519b41e13664a5959abb5cf55c1b4c5ab2410c539f721ebfb'
COLUMNS = ['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'image']
LOCKED = {
    DATA: '52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324',
    SOURCE: '957de2b23d0eae76afabb8e3e9d87ae62898185c046e8dd0c1bd398ccab7ca3e',
    MMB / 'cpu_inputs_current.json': 'b35a2b60bababb1c99884b248c90e64a409bc8c7ef72b5c83f6bf8faf205108e',
    MMB / 'cpu_inputs_author_pins.json': 'd8bdb3070ff9ca42d06633fb0d0d68f8ecb8e98f708b1640876d6ff3aa58f82e',
    MMB / 'cpu_llm_tokens_current.json': '05f7245c2b248a85f0fa56b6d3a298d1b6ed038b22a0661b80cb6832158be80c',
    MMB / 'cpu_llm_tokens_author_pins.json': '4751d27ed404a498bc39a7e6b5e03c5679e162b83225ea83c49177f649f2fb87',
    MMB / 'image_manifest.json': 'dc8d1a96cffc85847e2d881d2499e38fe13b8efed15af685393a487d762c729a',
    OLD / 'protocol.json': 'd339f8a41a0babc160389f1e0c908694e92293c488f3b646d00f40b245768863',
    OLD / 'independent_verification.json': '074f1ef2403a5c07182032e4d244b56b28fa28b064bc83a0ce0d74424a97027f',
    OLD / 'paired_full_comparison.json': 'c9dad4d7af1eaf25b804f42b9a72aed1df41b0e02c50bcf95879017d3c26a85f',
    OLD / 'supervisor_state.json': 'fc0d81f98a8b8486acc77e3b64650d954343b80d08a53b50ef40758daf432810',
    OLD / 'CPU_finalizer_state.json': 'd75db8be6b163864ae90fba8ad386bb9943fdaaaedbdf234f2b0c0bcd6f5bb2b',
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def validate_columns(columns):
    assert list(columns) == COLUMNS, 'Only the eight GT-free native input columns are allowed'


def validate_panel_structure(rows, universe_ids):
    ids = [int(row['index']) for row in rows]
    assert len(ids) == len(set(ids)) == 4377, 'Exactly4377 unique pre-expanded rows required'
    assert ids == [int(idx) for idx in universe_ids], 'Preserve all canonical source rows in order'
    groups = sorted({idx % 1000000 for idx in ids})
    assert len(groups) == 1176
    assert [row['position'] for row in rows] == [row['original_position'] for row in rows] == list(range(4377))
    assert all(row['group_id'] == row['index'] % 1000000 and row['round_id'] == 0 for row in rows)
    assert Counter(sum(idx % 1000000 == group for idx in ids) for group in groups) == {4: 938, 3: 149, 2: 89}


def fixed60_links(rows, fixed):
    assert fixed['n_rows'] == 60 and fixed['n_groups'] == 16
    assert fixed['group_ids'] == FIXED_GROUP_IDS
    result = []
    for old in fixed['rows']:
        position = old['original_position']
        current = rows[position]
        # File paths and panel-local positions are metadata, not model inputs.
        normalized = json.loads(json.dumps(current))
        normalized['position'] = old['position']
        normalized['image']['path'] = old['image']['path']
        assert normalized == old, 'Fixed60 original input record changed'
        canonical = lambda value: digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        result.append(dict(panel_position=old['position'], full_position=position, index=old['index'],
                           original_row_sha256=canonical(old), normalized_full_row_sha256=canonical(normalized)))
    assert len(result) == len({row['index'] for row in result}) == 60
    return result


def validate_anchor(record, old_prediction, old_runtime, clip_reference, llm_reference, image_reference):
    idx = record['index']
    assert idx == int(old_prediction['question_id']) == clip_reference['index'] == llm_reference['index'] == image_reference['index']
    assert old_prediction['round_id'] == 0
    assert old_prediction['prompt'] == record['full_guidance'] == old_runtime['guidance_text']
    assert digest_bytes(record['full_guidance'].encode()) == clip_reference['prompt_sha256'] == old_runtime['guidance_text_sha256']
    assert record['source_clip']['input_ids_sha256'] == clip_reference['input_ids_sha256']
    assert record['source_clip']['attention_mask_sha256'] == clip_reference['attention_mask_sha256']
    assert record['source_clip']['n_tokens'] == clip_reference['original_token_shape'][1]
    assert digest_bytes(record['llm_prompt'].encode()) == llm_reference['prompt_sha256']
    assert record['llm_input_ids_sha256'] == llm_reference['token_ids_sha256']
    assert record['llm_n_tokens'] == llm_reference['n_tokens']
    assert old_runtime['call'] == record['original_position'] + 1
    assert old_runtime['input_shape'] == [1, record['llm_n_tokens']]
    assert old_runtime['image_shape'] == record['image']['pixel_shape'] == [1, 3, 336, 336]
    assert old_runtime['image_dtype'] == record['image']['pixel_dtype'] == 'torch.float16'
    assert old_runtime['image_sizes'] == [record['image']['size']]
    assert record['image']['payload_sha256'] == clip_reference['image_payload_sha256'] == image_reference['payload_sha256']
    assert record['image']['decoded_rgb_sha256'] == image_reference['decoded_rgb_sha256']
    assert record['image']['size'] == image_reference['size'] and record['image']['mode'] == image_reference['mode']
    assert old_runtime['returned_visual_token_num'] == 128
    assert old_runtime['do_sample'] is False and old_runtime['temperature'] == 0
    assert old_runtime['num_beams'] == 1 and old_runtime['max_new_tokens'] == 1024
    assert isinstance(old_prediction['text'], str) and old_prediction['text'].strip()
    assert record['accepted_source']['text'] == old_prediction['text']


def prepare(output):
    # Imports below load only tokenizers/processors and CPU tensor utilities.
    # No llava package/model registry is imported.
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTokenizerFast
    assert not torch.cuda.is_initialized() and not output.exists(), 'Do not overwrite a prepared panel'
    torch.set_num_threads(2)
    for path, expected in LOCKED.items():
        assert sha(path) == expected, str(path)
    assert sha(FIXED) == FIXED_SHA
    fixed = read(FIXED)
    protocol = read(OLD / 'protocol.json')
    verification = read(OLD / 'independent_verification.json')
    paired = read(OLD / 'paired_full_comparison.json')
    assert verification['success'] and verification['all_rows_checked']
    assert verification['n_rows'] == 4377 and verification['n_circular_groups'] == 1176
    assert paired['success'] and all(row['original_worker_exited'] for row in paired['worker_exit'])
    assert read(OLD / 'supervisor_state.json')['status'] == read(OLD / 'CPU_finalizer_state.json')['status'] == 'complete'
    assert read(OLD / 'supervisor_state.json')['exit_code'] == 0
    assert read(OLD / 'CPU_finalizer_state.json')['comparison_sha256'] == sha(OLD / 'paired_full_comparison.json')
    for item in protocol['source_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    command = protocol['command']
    predictions_path = Path(command[command.index('--answers-file') + 1])
    runtime_path = OLD / 'observed/runtime.jsonl'
    assert sha(predictions_path) == verification['prediction_sha256']
    assert sha(runtime_path) == verification['runtime_sha256']
    raw_lines = predictions_path.read_bytes().splitlines()
    predictions = [json.loads(line) for line in raw_lines]
    runtimes = [json.loads(line) for line in runtime_path.read_text().splitlines()]
    assert len(predictions) == len(runtimes) == 4377
    assert len({int(row['question_id']) for row in predictions}) == 4377
    # Hashing/parsing the TSV physically reads source bytes containing GT.
    # Only these input columns are parsed; answers are never loaded or used.
    frame = pd.read_table(DATA, usecols=COLUMNS)[COLUMNS]
    validate_columns(frame.columns)
    assert len(frame) == 4377 and frame['index'].is_unique
    universe = [int(idx) for idx in frame['index']]
    groups = sorted({idx % 1000000 for idx in universe})
    assert len(groups) == 1176
    assert [groups[((2 * i + 1) * 1176) // 32] for i in range(16)] == FIXED_GROUP_IDS
    selected = frame.copy()
    assert len(selected) == 4377
    references = read(MMB / 'cpu_inputs_current.json')
    pinned = read(MMB / 'cpu_inputs_author_pins.json')
    clip_refs = {row['index']: row for row in references['guidance']}
    pin_clip_refs = {row['index']: row for row in pinned['guidance']}
    llm_refs = {row['index']: row for row in read(MMB / 'cpu_llm_tokens_current.json')['records']}
    pin_llm_refs = {row['index']: row for row in read(MMB / 'cpu_llm_tokens_author_pins.json')['records']}
    image_refs = {row['index']: row for row in read(MMB / 'image_manifest.json')}
    config = SimpleNamespace(**read(MODEL / 'config.json'))
    assert config.image_aspect_ratio == 'pad'
    constants = runpy.run_path(str(AUTHOR / 'llava/constants.py'))
    conversation = runpy.run_path(str(AUTHOR / 'llava/conversation.py'))
    namespace = dict(Image=Image, torch=torch, math=math, IMAGE_TOKEN_INDEX=constants['IMAGE_TOKEN_INDEX'])
    for source, names in [(SOURCE, ('is_none', 'get_options')),
                          (AUTHOR / 'llava/mm_utils.py', ('expand2square', 'process_images', 'tokenizer_image_token'))]:
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    clip = CLIPTokenizerFast.from_pretrained('openai/clip-vit-large-patch14-336', local_files_only=True)
    llm = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336', local_files_only=True)
    fingerprint = lambda array: digest_bytes(np.ascontiguousarray(array).tobytes())

    def clip_input(text):
        value = clip(text=text, return_tensors='np')
        count = value['input_ids'].shape[1]
        segments = (count - 1) // 77 + 1
        padded = {key: np.concatenate([array, np.zeros((1, 77 * segments - count), dtype=array.dtype)], axis=1).reshape(-1, 77)
                  for key, array in value.items()}
        return dict(text_sha256=digest_bytes(text.encode()), n_tokens=count, n_segments=segments,
                    input_ids=padded['input_ids'].tolist(), attention_mask=padded['attention_mask'].tolist(),
                    input_ids_sha256=fingerprint(padded['input_ids']), attention_mask_sha256=fingerprint(padded['attention_mask']))

    records, payloads, image_cache = [], {}, {}
    for original_position, row in selected.iterrows():
        idx = int(row['index'])
        full = row['question']
        if not namespace['is_none'](row['hint']):
            full = row['hint'] + '\n' + full
        options = namespace['get_options'](row, list('ABCD'))
        for letter, option in zip('ABCD', options):
            full += '\n' + letter + '. ' + option
        marker = constants['DEFAULT_IMAGE_TOKEN']
        if config.mm_use_im_start_end:
            marker = constants['DEFAULT_IM_START_TOKEN'] + marker + constants['DEFAULT_IM_END_TOKEN']
        conv = conversation['conv_templates']['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], marker + '\n' + full + "\nAnswer with the option's letter from the given choices directly.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = namespace['tokenizer_image_token'](prompt, llm, constants['IMAGE_TOKEN_INDEX'], return_tensors='pt')
        payload = base64.b64decode(row['image'], validate=True)
        payload_sha = digest_bytes(payload)
        if payload_sha not in image_cache:
            image = Image.open(io.BytesIO(payload)); image.load()
            pixels = namespace['process_images']([image], processor, config)[0].unsqueeze(0)
            reference_image = references['images'][payload_sha]
            pin_image = pinned['images'][payload_sha]
            assert reference_image == pin_image
            assert fingerprint(pixels.numpy()) == reference_image['float32_sha256']
            assert fingerprint(pixels.half().numpy()) == reference_image['float16_sha256']
            image_cache[payload_sha] = dict(path=str((output / 'images' / (payload_sha + '.image')).resolve()),
                payload_sha256=payload_sha, decoded_rgb_sha256=fingerprint(np.asarray(image.convert('RGB'))),
                size=list(image.size), mode=image.mode, pixel_shape=list(pixels.shape), pixel_dtype='torch.float16',
                pixel_float16_sha256=reference_image['float16_sha256'], pixel_float32_sha256=reference_image['float32_sha256'])
            payloads[payload_sha] = payload
            if len(image_cache) % 128 == 0:
                print(json.dumps(dict(CPU_images_verified=len(image_cache))), flush=True)
        record = dict(position=len(records), original_position=int(original_position), index=idx,
            group_id=idx % 1000000, round_id=0, full_guidance=full, question_only_guidance=row['question'],
            llm_prompt=prompt, llm_input_ids=ids.tolist(), llm_input_ids_sha256=fingerprint(ids.numpy()),
            llm_n_tokens=int(ids.numel()), source_clip=clip_input(full), question_only_clip=clip_input(row['question']),
            image=dict(image_cache[payload_sha]), accepted_source=dict(text=predictions[original_position]['text'],
                raw_line_sha256=digest_bytes(raw_lines[original_position]), raw_prediction_sha256=verification['prediction_sha256'],
                source_runtime_call=original_position + 1))
        assert clip_refs[idx] == pin_clip_refs[idx] and llm_refs[idx] == pin_llm_refs[idx]
        validate_anchor(record, predictions[original_position], runtimes[original_position], clip_refs[idx], llm_refs[idx], image_refs[idx])
        assert predictions[original_position]['options'] == options
        assert predictions[original_position]['option_char'] == list('ABCD')[:len(options)]
        records.append(record)
        if len(records) % 512 == 0:
            print(json.dumps(dict(CPU_rows_verified=len(records))), flush=True)
    validate_panel_structure(records, universe)
    links = fixed60_links(records, fixed)
    assert not torch.cuda.is_initialized() and not any(name.startswith('llava') for name in sys.modules)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    for digest, payload in payloads.items():
        (output / 'images' / (digest + '.image')).write_bytes(payload)
    panel = output / 'panel.tsv'
    selected.to_csv(panel, sep='\t', index=False, columns=COLUMNS)
    sanitized = pd.read_table(panel)
    validate_columns(sanitized.columns)
    assert sanitized['index'].tolist() == [record['index'] for record in records]
    for (_, row), record in zip(sanitized.iterrows(), records):
        full = row['question']
        if not namespace['is_none'](row['hint']): full = row['hint'] + '\n' + full
        for letter, option in zip('ABCD', namespace['get_options'](row, list('ABCD'))): full += '\n' + letter + '. ' + option
        assert full == record['full_guidance']
        assert digest_bytes(base64.b64decode(row['image'], validate=True)) == record['image']['payload_sha256']
    # The original answer-bearing TSV is provenance-only and excluded from
    # executable runtime guards and the sources map. Never auto-walk that file.
    sources = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): artifact(path)
               for path in LOCKED if path != DATA}
    sources.update(source_predictions=artifact(predictions_path), source_runtime=artifact(runtime_path),
                   fixed60_manifest=artifact(FIXED), preparer=artifact(__file__))
    manifest = dict(schema_version=1, prepared_UTC=datetime.now(timezone.utc).isoformat(), n_rows=4377, n_groups=1176,
        group_ids=groups, all_rounds=False, round_id=0, panel_tsv=artifact(panel), rows=records, sources=sources,
        provenance_only_sources=[dict(**artifact(DATA), contains_GT=True, preparation_bytes_read=True,
                                      GT_columns_parsed=False, runtime_access_forbidden=True)],
        source_TSV_contains_GT_bytes=True, source_TSV_bytes_read_including_GT=True, GT_columns_parsed=False,
        columns_parsed=COLUMNS, contains_historical_model_responses=True, historical_responses_are_GT=False,
        fixed60_reference=dict(manifest=artifact(FIXED), panel_tsv=fixed['panel_tsv'], rows=fixed['rows'], links=links,
                               only_normalized_fields=['position', 'image.path']),
        model_loaded=False, CUDA_initialized=False, accuracy_computed=False,
        checks=dict(source_clip_fingerprints=4377, source_LLM_fingerprints=4377, source_raw_text_bindings=4377,
                    source_runtime_bindings=4377, source_image_pixel_fingerprints=4377, unique_images=len(image_cache),
                    complete_preexpanded_groups=1176, sanitized_roundtrip_rows=4377, fixed60_original_row_anchors=60),
        limits=['Preparation physically read original TSV bytes containing GT; only the eight input columns were parsed.',
                'provenance_only_sources is documentary: the original GT TSV must not be read by generation or validation guards.',
                'Historical model responses are provenance/replay anchors, not GT or score inputs; do not require all new nonanchor responses to match them.',
                'Fixed60 original input rows are preserved verbatim; full rows differ only in panel-local position and copied image path.',
                'Historical masks, importance, embeddings and output IDs were absent from the old full run; the completed fixed60 GPU panel is a separate anchor.',
                'No model, selector, GPU, generation or accuracy computation was used for preparation.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(success=True, n_rows=4377, n_groups=1176, unique_images=len(image_cache),
                         manifest=artifact(output / 'manifest.json'), panel_tsv=artifact(panel),
                         source_bytes_contain_GT=True, GT_columns_parsed=False, model_loaded=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'inputs')
    arguments = parser.parse_args()
    prepare(arguments.output_dir.resolve())
