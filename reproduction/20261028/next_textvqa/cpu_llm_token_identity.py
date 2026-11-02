"""Compare native rendered LLaVA prompt IDs across author SentencePiece pins."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, '/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA')
    import sentencepiece
    import torch
    from transformers import AutoTokenizer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    model = Path('/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b')
    config = json.loads((model / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=True)
    questions = [json.loads(s) for s in (here / 'author_linked_textvqa_questions.jsonl').read_text().splitlines() if s.strip()]
    rows = []
    for i, q in enumerate(questions):
        image = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if config.get('mm_use_im_start_end') else DEFAULT_IMAGE_TOKEN
        conversation = conv_templates['vicuna_v1'].copy()
        conversation.append_message(conversation.roles[0], image + '\n' + q['text'])
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        rows.append({'position': i, 'question_id': q['question_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'n_tokens': ids.shape[-1], 'token_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()})
    assert not torch.cuda.is_initialized()
    report = {'sentencepiece': sentencepiece.__version__, 'sentencepiece_module': sentencepiece.__file__, 'protobuf': importlib.metadata.version('protobuf'), 'tokenizer_class': type(tokenizer).__name__, 'rows': rows, 'cuda_initialized': False, 'scope': 'All5000 full officialOCRquestions rendered by nativevicuna_v1 template + image token; AutoTokenizer(use_fast=False) and native tokenizer_image_token.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sentencepiece': sentencepiece.__version__, 'protobuf': report['protobuf'], 'n': len(rows), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
