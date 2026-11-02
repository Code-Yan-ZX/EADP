"""Hash real NeXT preprocessing under current or isolated author package pins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-used-images', action='store_true')
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    author = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA')
    sys.path.insert(0, str(author))
    import PIL
    from PIL import Image
    import numpy as np
    import tokenizers
    import torch
    from transformers import CLIPImageProcessor, CLIPTokenizerFast
    from transformers.utils import logging
    from llava.mm_utils import process_images
    logging.set_verbosity_error()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    clip = Path('/data/models/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
    processor = CLIPImageProcessor.from_pretrained(clip, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(clip, local_files_only=True)
    config = types.SimpleNamespace(**json.loads(Path('/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b/config.json').read_text()))
    questions = [json.loads(s) for s in (here / 'author_linked_textvqa_questions.jsonl').read_text().splitlines() if s.strip()]
    panel = [json.loads(s) for s in (here / 'panel64.questions.jsonl').read_text().splitlines() if s.strip()]
    if args.all_used_images:
        by_image = {q['image']: q for q in questions}
        panel = [by_image[name] for name in sorted(by_image)]
    image_rows = []
    for q in panel:
        path = (Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/textvqa/train_images') if args.all_used_images else here / 'fresh_panel_images') / q['image']
        image = Image.open(path).convert('RGB')
        values = process_images([image], processor, config)[0]
        image_rows.append({'question_id': q['question_id'], 'image': q['image'], 'image_size': image.size, 'shape': list(values.shape), 'decoded_RGB_sha256': digest(np.asarray(image).tobytes()), 'FP32_sha256': digest(values.numpy().tobytes()), 'FP16_sha256': digest(values.half().numpy().tobytes())})
    text_rows = []
    for i, q in enumerate(questions):
        guidance = q['text'].replace('\nAnswer the question using a single word or phrase.', '')
        result = tokenizer(text=guidance, return_tensors='pt')
        text_rows.append({'position': i, 'question_id': q['question_id'], 'ids_length': result.input_ids.shape[-1], 'input_ids_sha256': digest(result.input_ids.numpy().tobytes()), 'attention_mask_sha256': digest(result.attention_mask.numpy().tobytes())})
    report = {'Pillow': PIL.__version__, 'Pillow_module': PIL.__file__, 'tokenizers': tokenizers.__version__, 'tokenizers_module': tokenizers.__file__, 'numpy': np.__version__, 'torch': torch.__version__, 'CLIP_snapshot': str(clip), 'image_rows': image_rows, 'CLIP_text_rows': text_rows, 'cuda_initialized': torch.cuda.is_initialized(), 'image_scope': '64 fixed samples / 64 unique official downloaded JPEGs, actual AnyRes process_images and FP16 deployment values.', 'tokenization_scope': 'All5000 official OCR prompts, exact native guidance transform and CLIPTokenizerFast.'}
    if args.all_used_images:
        report['image_scope'] = 'All3166 unique images used by full5000 official questions, previously CRC32+size-verified against official ZIP; actual AnyRes preprocessing.'
    assert not report['cuda_initialized']
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'Pillow': PIL.__version__, 'tokenizers': tokenizers.__version__, 'images': len(image_rows), 'texts': len(text_rows), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
