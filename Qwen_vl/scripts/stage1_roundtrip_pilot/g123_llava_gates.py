"""G1-G3 gates — LLaVA AnchorZip port identity and mechanics.

G1 identity: the ported function in MODE='official_replay' must
reproduce the OFFICIAL encode_images keep masks bit-exactly on 10
TextVQA images (visual_token_num=128).
G2 N/K: the RTG path keeps exactly 128 tokens per image (v1.5).
G3 lam-0 identity: Completion with lam=0 equals pure gather (bitwise).
G4 (separate script): 100-question main-method generation sanity.

Usage: python g123_llava_gates.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"
sys.path.insert(0, LLAVA_ROOT)
sys.path.insert(0, LLAVA_ROOT + "/llava/model")
QWEN_SCRIPTS = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts"
for _p in ("anchor_merge_pilot",):
    sys.path.insert(0, f"{QWEN_SCRIPTS}/{_p}")

MODEL_PATH = "/media/disk2/YZX/doct/FastV/llava-v1.5-7b"
QS = os.path.join(LLAVA_ROOT,
                  "playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl")
IMAGES = os.path.join(LLAVA_ROOT, "playground/data/eval/textvqa/train_images")
N = 10
K = 128
OUT = ("/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/"
       "stage1_roundtrip_pilot/legacy_full/g123_llava_gates.json")


def build_inputs(questions, tokenizer, image_processor, model):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from PIL import Image

    items = []
    for line in questions:
        qs = line["text"]
        cur = qs
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).cuda()
        image = Image.open(os.path.join(IMAGES, line["image"])) \
            .convert("RGB")
        tensor = process_images([image], image_processor,
                                model.config)[0].half().cuda()
        items.append(dict(cur_prompt=cur, input_ids=input_ids,
                          images=tensor.unsqueeze(0),
                          image_sizes=[image.size]))
    return items


def main():
    import llava_arch_anchorzip as AZ
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
    import llava.model.llava_arch as LA

    disable_torch_init()
    questions = [json.loads(q) for q in open(QS)][:N]
    tokenizer, model, image_processor, _ctx = load_pretrained_model(
        MODEL_PATH, None, get_model_name_from_path(MODEL_PATH),
        visual_token_num=K, beta=2.0, alpha=0.5)
    items = build_inputs(questions, tokenizer, image_processor, model)

    orig_fn = LA.LlavaMetaForCausalLM.encode_images
    per_img = lambda fn, mode: [
        (lambda fm: fm[1].reshape(-1))(fn(model, it["images"],
                                          texts=it["cur_prompt"]))
        for it in items]

    # G1: official vs official_replay (bitwise masks per image)
    AZ.MODE = "official_replay"
    masks_o, masks_r = [], []
    for it in items:
        mo = orig_fn(model, it["images"], texts=it["cur_prompt"])[1]
        mr = AZ._anchorzip_encode_images(model, it["images"],
                                         texts=it["cur_prompt"])[1]
        masks_o.append(mo.reshape(-1))
        masks_r.append(mr.reshape(-1))
    g1_masks = all(bool(torch.equal(a, b))
                   for a, b in zip(masks_o, masks_r))

    # G2: RTG path keeps exactly K per image
    AZ.MODE = "rtg"
    masks_a = [AZ._anchorzip_encode_images(model, it["images"],
                                           texts=it["cur_prompt"])[1]
               .reshape(-1) for it in items]
    kept = [int(m.sum()) for m in masks_a]
    g2 = all(k == K for k in kept)

    # G3: lam=0 == gather (compare merged stream against masked gather)
    from amp_common import compute_assignment, merge_stream
    f0 = AZ._anchorzip_encode_images(model, items[0]["images"],
                                     texts=items[0]["cur_prompt"])[0][0]
    keep0 = torch.nonzero(masks_a[0], as_tuple=False).squeeze(1)
    dropped, gid, _s = compute_assignment(f0, keep0)
    y0 = merge_stream(f0, keep0, dropped, gid, "uniform", 0.0)
    gather0 = f0[keep0]
    g3 = bool(torch.equal(y0, gather0))

    g1_keep = g1_masks
    result = dict(g1_masks_equal=g1_masks, g1_keep_equal=g1_keep,
                  g2_kept_per_image=kept, g2_all_K=bool(g2),
                  g3_lam0_equals_gather=g3,
                  verdict=bool(g1_masks and g1_keep and g2 and g3))
    with open(OUT + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps(result, indent=1))
    if not result["verdict"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
