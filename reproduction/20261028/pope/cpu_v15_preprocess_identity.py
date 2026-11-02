#!/usr/bin/env python3
"""Hash all actual POPE preprocessing under current or isolated author pins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--author-pins', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    source = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA')
    sys.path.insert(0, str(source))
    import PIL
    from PIL import Image
    import numpy as np
    import tokenizers
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTokenizerFast
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    if args.author_pins:
        assert PIL.__version__ == '10.4.0' and tokenizers.__version__ == '0.15.1'
        assert 'vendor_author_pins' in PIL.__file__ and 'vendor_author_pins' in tokenizers.__file__
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    model = Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b')
    clip = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
    config = types.SimpleNamespace(**json.loads((model / 'config.json').read_text()))
    processor = CLIPImageProcessor.from_pretrained(clip, local_files_only=True)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip, local_files_only=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=True)
    questions = [json.loads(x) for x in (out / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    image_rows = []
    image_root = Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/pope/val2014')
    for name in sorted({q['image'] for q in questions}):
        image = Image.open(image_root / name).convert('RGB')
        values = process_images([image], processor, config)[0]
        image_rows.append({'image': name, 'image_size': list(image.size), 'shape': list(values.shape),
            'decoded_RGB_sha256': digest(np.asarray(image).tobytes()), 'FP32_sha256': digest(values.numpy().tobytes()),
            'FP16_sha256': digest(values.half().numpy().tobytes())})
    cache = {}
    text_rows = []
    for position, q in enumerate(questions):
        if q['text'] not in cache:
            guidance = q['text'].replace('\nAnswer the question using a single word or phrase.', '')
            encoded = clip_tokenizer(text=guidance, return_tensors='pt')
            segments = (encoded.input_ids.shape[1] - 1) // 77 + 1
            padding = 77 * segments - encoded.input_ids.shape[1]
            padded = {key: torch.cat([value, value.new_zeros((value.shape[0], padding))], dim=1).reshape(-1, 77)
                      for key, value in encoded.items()}
            prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], prefix + '\n' + q['text'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            llm_ids = tokenizer_image_token(prompt, llm_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            cache[q['text']] = {'raw_prompt_sha256': digest(q['text'].encode()), 'full_LLM_prompt_sha256': digest(prompt.encode()),
                'LLM_input_ids_sha256': digest(llm_ids.numpy().tobytes()), 'LLM_input_length': llm_ids.numel(),
                'CLIP_input_ids_sha256': digest(encoded.input_ids.numpy().tobytes()),
                'CLIP_attention_mask_sha256': digest(encoded.attention_mask.numpy().tobytes()),
                'CLIP_padded_input_ids_sha256': digest(padded['input_ids'].numpy().tobytes()),
                'CLIP_padded_attention_mask_sha256': digest(padded['attention_mask'].numpy().tobytes())}
        text_rows.append({'position': position, 'question_id': q['question_id'], **cache[q['text']]})
    report = {'Pillow': PIL.__version__, 'Pillow_module': PIL.__file__, 'tokenizers': tokenizers.__version__,
              'tokenizers_module': tokenizers.__file__, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
              'image_rows': image_rows, 'text_rows': text_rows, 'unique_texts': len(cache),
              'image_scope': 'All500usedPOPEimages, actual source pad+CLIP336 preprocessing, RGB/FP32/FP16 hashes.',
              'text_scope': 'All8910 canonical source prompts, full vicuna_v1 LLM IDs and native suffix-removed CLIP guidance IDs/masks including source77zero-padding.',
              'cuda_initialized': torch.cuda.is_initialized()}
    assert not report['cuda_initialized']
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'images': len(image_rows), 'prompts': len(text_rows),
                      'unique_texts': len(cache), 'Pillow': PIL.__version__, 'tokenizers': tokenizers.__version__, 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
