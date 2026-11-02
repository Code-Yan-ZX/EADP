#!/usr/bin/env python3
"""CPU-only full POPE slow-tokenizer check across author SentencePiece pins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--author-pins', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, '/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA')
    import sentencepiece
    import google.protobuf
    import torch
    from transformers import AutoTokenizer
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    if args.author_pins:
        assert sentencepiece.__version__ == '0.1.99' and google.protobuf.__version__ == '3.20.3'
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained('/media/disk2/YZX/doct/FastV/llava-v1.5-7b', use_fast=False, local_files_only=True)
    questions = [json.loads(s) for s in (root / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    cache, rows = {}, []
    for i, q in enumerate(questions):
        if q['text'] not in cache:
            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + q['text'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            cache[q['text']] = {'full_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'LLM_input_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(), 'LLM_input_length': ids.numel()}
        rows.append({'position': i, 'question_id': q['question_id'], **cache[q['text']]})
    result = {'sentencepiece': sentencepiece.__version__, 'sentencepiece_module': sentencepiece.__file__,
              'protobuf': google.protobuf.__version__, 'protobuf_module': google.protobuf.__file__,
              'rows': rows, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
