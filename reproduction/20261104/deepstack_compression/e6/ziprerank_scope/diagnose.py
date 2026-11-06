"""Inspect the author's QI-Early text-selection scope on three DEV inputs.

No model weights, generation, labels, or CUDA are used.  The wrapper and
selector are imported from their fixed local source trees without alteration.
"""
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path('/media/disk2/YZX/research/EADP_v2')
BASE = ROOT / 'reproduction/20261104/deepstack_compression'
SCOPE = BASE / 'e6/ziprerank_scope'
AUTHOR = ROOT / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
ZIP = BASE / 'e6/external_sources/ziprerank_f367b09'
MODEL = Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path[:0] = [str(AUTHOR), str(ZIP)]

import torch
import transformers
from transformers import AutoConfig, AutoProcessor
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    plan = json.loads((SCOPE / 'plan001.json').read_text())
    for name, expected in plan['source_sha256'].items():
        assert sha(name) == expected, name
    rows = [json.loads(x) for x in (BASE / 'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    selected = []
    for expected in plan['anchors']:
        matches = [r for r in rows if r['task'] == expected['task'] and r['qid'] == expected['qid']]
        assert len(matches) == 1
        row = matches[0]
        assert row['message_sha256'] == expected['message_sha256']
        assert sha(row['image_path']) == row['image_sha256'] == expected['image_sha256']
        assert [x['type'] for x in row['message']] == ['image', 'text']
        selected.append(row)

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    wrapper = object.__new__(Qwen3VLChatFixedRes)
    wrapper.system_prompt = None
    wrapper.min_pixels = wrapper.max_pixels = 1048576
    wrapper.total_pixels = None
    wrapper.processor = processor
    wrapper.model = SimpleNamespace(device='cpu', dtype=torch.bfloat16)
    selector = SimpleNamespace(config=config, qi_early_text_mode='all_before_image', query_emb_token_id=None)
    out = []
    for row in selected:
        messages = wrapper._build_messages(row['message'], dataset=row['task'])
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        marker = '<|image_pad|>'
        marker_pos = rendered.index(marker)
        question = row['message'][1]['value']
        question_pos = rendered.index(question)
        assert marker_pos < question_pos
        inputs = wrapper._processor_inputs(messages)
        ids = inputs['input_ids']
        image_pos = int(torch.where(ids[0] == config.image_token_id)[0][0])
        prefix_ids = processor.tokenizer(rendered[:marker_pos], add_special_tokens=False)['input_ids']
        assert prefix_ids == ids[0, :image_pos].tolist()
        embeds = torch.arange(ids.shape[1], dtype=torch.float32).reshape(1, -1, 1)
        selected_embeds = Qwen3VLWithQIEarly._extract_query_text_embeds(selector, embeds, ids)
        chosen = selected_embeds[:, 0].to(torch.int64).tolist()
        assert chosen and max(chosen) < image_pos
        # The wrapper's rendered question is strictly after the image placeholder.
        # Selected token positions are strictly before the first image token.
        decoded_prefix = processor.tokenizer.decode(ids[0, :image_pos].tolist(), skip_special_tokens=False)
        assert decoded_prefix == rendered[:marker_pos]
        out.append({'task': row['task'], 'qid': row['qid'], 'image_sha256': row['image_sha256'],
                    'message_sha256': row['message_sha256'], 'input_token_count': ids.shape[1],
                    'image_token_count': int((ids == config.image_token_id).sum()),
                    'first_image_token_index': image_pos, 'selected_token_indices': chosen,
                    'selected_token_count': len(chosen), 'question_after_image_in_template': True,
                    'prefix_tokenization_matches_processor': True,
                    'question_text_in_selected_scope': False,
                    'selected_prefix_sha256': hashlib.sha256(decoded_prefix.encode()).hexdigest(),
                    'image_grid_thw': inputs['image_grid_thw'].tolist()})
    assert not torch.cuda.is_initialized()
    result = {'description': 'Three real DEV inputs; author default text mode; no labels or model weights',
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'author_wrapper_path': inspect.getfile(Qwen3VLChatFixedRes),
              'author_selector_path': inspect.getfile(Qwen3VLWithQIEarly),
              'cuda_initialized': False, 'plan_sha256': sha(SCOPE / 'plan001.json'), 'anchors': out}
    output = SCOPE / 'result001.json'
    with open(output, 'x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')


if __name__ == '__main__':
    main()
