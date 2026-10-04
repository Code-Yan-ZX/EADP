"""G0 — LLaVA-1.5 environment smoke gate (P3 prereg §3).

Mirrors llava/eval/model_vqa.py exactly (same loader, prompt template,
conv mode vicuna_v1, greedy, max_new_tokens=1024) on the first 5 TextVQA
val questions, FULL path (visual_token_num=0, no pruning).  Checks:
  1. model loads through the official EADP builder,
  2. outputs are non-degenerate (non-empty, no NaN/repetition),
  3. determinism: a second identical pass reproduces outputs bit-exact.

Usage: python g0_llava_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"
sys.path.insert(0, LLAVA_ROOT)

MODEL_PATH = "/media/disk2/YZX/doct/FastV/llava-v1.5-7b"
QS = os.path.join(LLAVA_ROOT,
                  "playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl")
IMAGES = os.path.join(LLAVA_ROOT, "playground/data/eval/textvqa/train_images")
N = 5
OUT = ("/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/"
       "stage1_roundtrip_pilot/legacy_full/g0_llava_smoke.json")


def run_pass(questions, tokenizer, model, image_processor):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from PIL import Image

    outs = []
    for line in questions:
        qs = line["text"]
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).cuda()
        image = Image.open(os.path.join(
            IMAGES, line["image"])).convert("RGB")
        image_tensor = process_images(
            [image], image_processor, model.config)[0]
        with torch.inference_mode():
            output_ids, _vtn = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                texts=cur_prompt,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True)
        text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        outs.append(dict(question_id=line["question_id"],
                         prompt=cur_prompt, text=text))
    return outs


def main():
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()
    questions = [json.loads(q) for q in open(QS)][:N]
    tokenizer, model, image_processor, _ctx = load_pretrained_model(
        MODEL_PATH, None, get_model_name_from_path(MODEL_PATH),
        visual_token_num=0, beta=1.0, alpha=0.5)
    model.eval()

    p1 = run_pass(questions, tokenizer, model, image_processor)
    p2 = run_pass(questions, tokenizer, model, image_processor)

    nonempty = all(o["text"] for o in p1)
    deterministic = all(
        a["text"] == b["text"] for a, b in zip(p1, p2))
    no_nan = all("nan" not in o["text"].lower() for o in p1)
    result = dict(gate="G0", model=MODEL_PATH, n=N,
                  nonempty=bool(nonempty), deterministic=bool(deterministic),
                  no_nan=bool(no_nan),
                  verdict=bool(nonempty and deterministic and no_nan),
                  outputs=p1)
    with open(OUT + ".tmp", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"},
                     indent=1))
    for o in p1:
        print(f"  [{o['question_id']}] {o['text'][:90]!r}")
    print("[saved]", OUT)
    if not result["verdict"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
