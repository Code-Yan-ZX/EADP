#!/usr/bin/env python3
"""Audit MMB guidance inputs only; no model, answers, GT, or scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import CLIPTokenizerFast

HERE = Path(__file__).resolve().parent
MMB = HERE.parent
ROOT = MMB.parents[2]
AUTHOR = ROOT / '_reproduction/author_eadp/LLaVA'
SOURCE = AUTHOR / 'llava/eval/model_vqa_mmbench.py'
DATA = MMB / 'data/mmbench_dev_20230712.tsv'
REFERENCE = MMB / 'cpu_inputs_current.json'
EXPECTED = {
    SOURCE: '957de2b23d0eae76afabb8e3e9d87ae62898185c046e8dd0c1bd398ccab7ca3e',
    DATA: '52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324',
    REFERENCE: 'b35a2b60bababb1c99884b248c90e64a409bc8c7ef72b5c83f6bf8faf205108e',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return {'path': str(path), 'sha256': sha(path)}


def main():
    assert not torch.cuda.is_initialized()
    for path, digest in EXPECTED.items():
        assert sha(path) == digest, str(path)
    tree = ast.parse(SOURCE.read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ('is_none', 'get_options')]
    assert len(helpers) == 2
    namespace = {'math': math}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    # Deliberately exclude the answer and image columns. Image identity is reused
    # from the already accepted preprocessing fingerprint reference.
    frame = pd.read_table(DATA, usecols=['index', 'question', 'hint', 'A', 'B', 'C', 'D'])
    reference = json.loads(REFERENCE.read_text())
    assert len(frame) == reference['n_rows'] == 4377
    refs = {row['index']: row for row in reference['guidance']}
    assert len(refs) == 4377 and frame['index'].is_unique
    tokenizer = CLIPTokenizerFast.from_pretrained(
        'openai/clip-vit-large-patch14-336', local_files_only=True)

    def tokenize(text):
        inputs = tokenizer(text=text, return_tensors='np')
        length = inputs['input_ids'].shape[1]
        segments = (length - 1) // 77 + 1
        padding = 77 * segments - length
        padded = {key: np.concatenate([value, np.zeros((value.shape[0], padding), dtype=value.dtype)],
                                     axis=1).reshape(-1, 77)
                  for key, value in inputs.items()}
        return dict(n_tokens=length, n_segments=segments,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    **{key + '_sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                       for key, value in padded.items()})

    records, groups = [], defaultdict(list)
    hints = 0
    for _, row in frame.iterrows():
        idx = int(row['index'])
        question = row['question']
        full = question
        if not namespace['is_none'](row['hint']):
            full = row['hint'] + '\n' + full
            hints += 1
        for letter, option in zip('ABCD', namespace['get_options'](row, list('ABCD'))):
            full += '\n' + letter + '. ' + option
        source, variant = tokenize(full), tokenize(question)
        old = refs[idx]
        assert source['text_sha256'] == old['prompt_sha256']
        assert source['input_ids_sha256'] == old['input_ids_sha256']
        assert source['attention_mask_sha256'] == old['attention_mask_sha256']
        assert source['n_tokens'] == old['original_token_shape'][1]
        record = dict(index=idx, group=idx % 1000000,
                      image_payload_sha256=old['image_payload_sha256'],
                      full_guidance=source, question_only_guidance=variant)
        records.append(record)
        groups[record['group']].append(record)
    assert len(groups) == 1176
    signature = lambda row, key: (row[key]['input_ids_sha256'], row[key]['attention_mask_sha256'])
    counts = dict(
        groups_same_question=sum(len({row['question_only_guidance']['text_sha256'] for row in group}) == 1
                                 for group in groups.values()),
        groups_same_image_payload=sum(len({row['image_payload_sha256'] for row in group}) == 1
                                      for group in groups.values()),
        groups_current_full_guidance_varies=sum(len({signature(row, 'full_guidance') for row in group}) > 1
                                               for group in groups.values()),
        groups_question_only_guidance_constant=sum(len({signature(row, 'question_only_guidance') for row in group}) == 1
                                                  for group in groups.values()))
    assert all(value == 1176 for value in counts.values())
    lengths = {}
    for key in ('full_guidance', 'question_only_guidance'):
        sizes = [row[key]['n_tokens'] for row in records]
        lengths[key] = dict(min=min(sizes), max=max(sizes), over77=sum(size > 77 for size in sizes),
                            segment_histogram=dict(Counter(row[key]['n_segments'] for row in records)))
    changed = sum(signature(row, 'full_guidance') != signature(row, 'question_only_guidance') for row in records)
    assert changed == 4377
    assert not any(name.startswith('llava') for name in sys.modules)
    assert not torch.cuda.is_initialized()
    fingerprints = HERE / 'token_fingerprints.jsonl'
    fingerprints.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    report = dict(success=True, created_UTC=datetime.now(timezone.utc).isoformat(),
                  scope='Native source and inactive question-only CLIP-guidance boundary; input tokenization only.',
                  n_rows=4377, n_circular_groups=1176, source_fingerprint_checks=4377,
                  changed_guidance_rows=changed, rows_with_hint=hints, token_lengths=lengths,
                  group_identity=counts,
                  source=artifact(SOURCE), dataset=artifact(DATA), reference=artifact(REFERENCE),
                  clip_encoder=artifact(AUTHOR / 'llava/model/multimodal_encoder/clip_encoder.py'),
                  script=artifact(__file__), fingerprints=artifact(fingerprints),
                  runtime=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                               CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], CUDA_initialized=False),
                  model_loaded=False, GT_columns_loaded=False, predictions_or_scores_loaded=False,
                  actual_features_or_masks_recomputed=False, paper_argv_confirmed=False,
                  source_locations={'prompt_with_hint_options': '92-100', 'guidance_boundary': '118-119',
                                    'texts_argument': '130'},
                  interpretation='All circular rotations share question and image, but source guidance changes with rotated options. '
                                 'Question-only guidance removes that rotation dependence at the tokenizer input level; '
                                 'no actual importance, selection, embeddings, generated answers or accuracy effect has been tested.',
                  limits=['The inactive comment is not proof of the paper main-table guidance recipe.',
                          'This change removes hints as well as options and changes segmentation; these effects are not separated.',
                          'Tokenization equality does not guarantee deterministic GPU equality or unchanged selected tokens.',
                          'The existing image identity comes from accepted fingerprints, not a new image payload audit.',
                          'No pruner numerical cache was reconstructed; no model or selector was run.'])
    (HERE / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('success', 'n_rows', 'n_circular_groups',
                     'source_fingerprint_checks', 'changed_guidance_rows', 'rows_with_hint',
                     'token_lengths', 'group_identity', 'model_loaded', 'GT_columns_loaded')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
