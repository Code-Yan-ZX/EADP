"""Test a transparent question-token mapping, without predictions or weights.

All 384 frozen DEV prompts use the tokenizer; three anchors additionally use
the real image processor. The official author selector itself is not modified.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path('/media/disk2/YZX/research/EADP_v2')
BASE = ROOT / 'reproduction/20261104/deepstack_compression'
HERE = BASE / 'e6/ziprerank_question_scope'
AUTHOR = ROOT / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
MODEL = Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path.insert(0, str(AUTHOR))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'plan001.json').read_text())
    for p, expected in plan['sources'].items():
        if sha(p) != expected:
            raise ValueError('source mismatch: ' + p)
    import torch
    import transformers
    from transformers import AutoProcessor, AutoConfig
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    wrapper = object.__new__(Qwen3VLChatFixedRes)
    wrapper.system_prompt = None
    wrapper.min_pixels = wrapper.max_pixels = 1048576
    wrapper.total_pixels = None
    wrapper.processor = processor
    wrapper.model = SimpleNamespace(device='cpu', dtype=torch.bfloat16)
    rows = [json.loads(x) for x in (BASE / 'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    if len(rows) != 384:
        raise ValueError('unexpected panel size')
    anchors = {(x['task'], str(x['qid'])) for x in plan['processor_anchors']}
    records = []
    for row in rows:
        if [x['type'] for x in row['message']] != ['image', 'text']:
            raise ValueError('unexpected message order')
        text = row['message'][1]['value']
        messages = wrapper._build_messages(row['message'], dataset=row['task'])
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered.count(text) != 1:
            raise ValueError('question text not uniquely located in template')
        start = rendered.index(text); end = start + len(text)
        encoded = processor.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']; ids = encoded['input_ids']
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError('noncontiguous question tokens')
        if any(offsets[i][0] < start or offsets[i][1] > end for i in indices):
            raise ValueError('question boundary shares a token with template')
        image_indices = [i for i, token in enumerate(ids) if token == config.image_token_id]
        if len(image_indices) != 1 or indices[0] <= image_indices[0]:
            raise ValueError('unexpected rendered image ordering')
        record = {'task': row['task'], 'qid': str(row['qid']), 'message_sha256': row['message_sha256'],
                  'question_token_count': len(indices), 'question_ids_sha256': hashlib.sha256(
                      json.dumps([ids[i] for i in indices], separators=(',', ':')).encode()).hexdigest(),
                  'question_is_contiguous_and_after_image': True, 'boundaries_do_not_cross_template': True,
                  'real_processor_verified': False}
        if (row['task'], str(row['qid'])) in anchors:
            if sha(row['image_path']) != row['image_sha256']:
                raise ValueError('anchor image mismatch')
            inputs = wrapper._processor_inputs(messages)
            actual = inputs['input_ids'][0].tolist()
            actual_images = [i for i, token in enumerate(actual) if token == config.image_token_id]
            if len(actual_images) != 1024:
                raise ValueError('unexpected visual token count')
            shift = len(actual_images) - 1
            mapped = [i + shift for i in indices]
            if [actual[i] for i in mapped] != [ids[i] for i in indices]:
                raise ValueError('expanded processor token mapping mismatch')
            record.update(real_processor_verified=True, expanded_question_start=mapped[0],
                          expanded_question_end_exclusive=mapped[-1]+1, image_tokens=len(actual_images))
        records.append(record)
    if sum(r['real_processor_verified'] for r in records) != 3 or torch.cuda.is_initialized():
        raise ValueError('anchor count/CUDA mismatch')
    out = {'status': 'completed', 'scope': '384 DEV tokenizer templates, 3 real image processor checks; no labels, weights, GPU or scoring',
           'plan_sha256': sha(HERE/'plan001.json'), 'torch': torch.__version__,
           'transformers': transformers.__version__, 'cuda_initialized': False, 'records': records}
    with (HERE / 'result001.json').open('x') as f:
        json.dump(out, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


if __name__ == '__main__':
    main()
