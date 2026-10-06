"""AnchorZip online efficiency mini-panel for LLaVA-1.5 / NeXT (P3, 2026-10-06).

Scope (user, 2026-10-05): TTFT, full generation wall time, output length,
module-level times, peak VRAM.  Arms: FULL(keep-all) + AnchorZip at the two
frozen budgets (v1.5: K128/K64 single image; NeXT: 128/64 per crop,
nominal total K640/K320 -- actual per-crop quotas follow the official
importance allocation, e.g. 188/98/113/104/134=637).

All probes are external wrappers; the frozen port
(llava_arch_anchorzip.py) and the official llava_arch.py are NOT modified.
Timed modules: vision tower, mm_projector, sim_cross, greed_select
(facility), Completion (compute_assignment+merge_stream), encode_images
(total), and the LM forward (first call == prefill).  TTFT is measured to
the end of the first LM forward (greedy, natural EOS, no length cap beyond
max_new_tokens=512).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict

import torch

LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)
QWEN_SCRIPTS = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts"
for _p in ("anchor_merge_pilot", "anchor_completion_validation",
           "stage1_roundtrip_pilot"):
    if f"{QWEN_SCRIPTS}/{_p}" not in sys.path:
        sys.path.insert(0, f"{QWEN_SCRIPTS}/{_p}")

from PIL import Image  # noqa: E402

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from llava.mm_utils import (tokenizer_image_token, process_images,  # noqa: E402
                            get_model_name_from_path)

T = defaultdict(float)   # cumulative seconds per probe
C = defaultdict(int)


def _timed(orig, key):
    # functools.wraps keeps the original signature visible: HF generate()
    # inspects signature(self.forward) for "attention_mask" and skips
    # creating the mask when the wrapper hides it (4.37.2 utils.py:1352).
    @functools.wraps(orig)
    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig(*a, **k)
        torch.cuda.synchronize()
        T[key] += time.perf_counter() - t0
        C[key] += 1
        return r
    return wrapper


def install_probes(model):
    md = model.model  # LlavaMetaModel: vision tower + projector live here
    vt = md.get_vision_tower()
    vt.forward = _timed(vt.forward, "vision_tower")
    md.mm_projector.forward = _timed(md.mm_projector.forward, "projector")
    # these live on LlavaMetaForCausalLM (model itself); guard anyway
    if hasattr(model, "sim_cross"):
        model.sim_cross = _timed(model.sim_cross, "sim_cross")
    if hasattr(model, "greed_select"):
        model.greed_select = _timed(model.greed_select, "greed_select")
    model.encode_images = _timed(model.encode_images, "encode_images_total")
    try:
        import amp_common  # Completion operators used by the frozen port
        amp_common.compute_assignment = _timed(amp_common.compute_assignment,
                                               "completion_assignment")
        amp_common.merge_stream = _timed(amp_common.merge_stream,
                                         "completion_merge")
    except ImportError:
        pass
    fwd = model.forward

    @functools.wraps(fwd)
    def fwd_timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fwd(*a, **k)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        C["lm_forward"] += 1
        if C["lm_forward"] == 1:
            T["lm_prefill_first"] = dt  # per-sample overwrite
        else:
            T["lm_decode_rest"] += dt
        return r
    model.forward = fwd_timed


def run_arm(args, samples):
    sys.path.insert(0, LLAVA_ROOT + "/llava/model")
    from llava_arch_anchorzip import install as az_install  # noqa: E402
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name,
        visual_token_num=args.visual_token_num,
        beta=args.beta, alpha=args.alpha)
    if args.anchorzip:
        import llava_arch_anchorzip as _AZ
        _AZ.MODE = args.az_mode
        if args.az_lam is not None:
            _AZ.LAM = args.az_lam
        az_install()
    install_probes(model)

    T.clear(); C.clear()
    records = []
    for si, s in enumerate(samples):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + s["question"]
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(s["image_path"]).convert('RGB')
        image_tensor = process_images([image], image_processor,
                                      model.config)[0].unsqueeze(0).half().cuda()

        T.clear(); C.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=image_tensor, image_sizes=[image.size],
                texts=s["question"], do_sample=False, num_beams=1,
                max_new_tokens=512, use_cache=True)
        # llava generate returns (output_ids, visual_token_num)
        output_ids = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        ttft = T.get("lm_prefill_first", 0.0)
        n_new = int(output_ids.shape[1])
        rec = {
            "i": si, "question_id": s["question_id"],
            "ttft_s": round(ttft, 4), "total_s": round(total, 4),
            "out_tokens": n_new,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "vision_tower_s": round(T.get("vision_tower", 0.0), 4),
            "projector_s": round(T.get("projector", 0.0), 4),
            "sim_cross_s": round(T.get("sim_cross", 0.0), 4),
            "greed_select_s": round(T.get("greed_select", 0.0), 4),
            "completion_assignment_s":
                round(T.get("completion_assignment", 0.0), 4),
            "completion_merge_s": round(T.get("completion_merge", 0.0), 4),
            "encode_images_total_s":
                round(T.get("encode_images_total", 0.0), 4),
            "lm_prefill_first_s": round(T.get("lm_prefill_first", 0.0), 4),
            "lm_decode_rest_s": round(T.get("lm_decode_rest", 0.0), 4),
        }
        records.append(rec)
        print(f"[{args.tag}] {si+1}/{len(samples)} ttft={ttft*1e3:.0f}ms "
              f"total={total:.2f}s out={n_new}", flush=True)

    import statistics as st
    summary = {
        "tag": args.tag, "model": args.model_path,
        "visual_token_num": args.visual_token_num, "anchorzip": args.anchorzip,
        "beta": args.beta, "alpha": args.alpha,
        "n": len(records),
        "ttft_ms_mean": round(st.mean(r["ttft_s"] for r in records) * 1e3, 1),
        "total_s_mean": round(st.mean(r["total_s"] for r in records), 3),
        "out_tokens_mean": round(st.mean(r["out_tokens"] for r in records), 1),
        "peak_vram_mb_max": max(r["peak_vram_mb"] for r in records),
        "vision_tower_s_mean": round(st.mean(r["vision_tower_s"] for r in records), 4),
        "projector_s_mean": round(st.mean(r["projector_s"] for r in records), 4),
        "sim_cross_s_mean": round(st.mean(r["sim_cross_s"] for r in records), 4),
        "greed_select_s_mean": round(st.mean(r["greed_select_s"] for r in records), 4),
        "completion_assignment_s_mean":
            round(st.mean(r["completion_assignment_s"] for r in records), 4),
        "completion_merge_s_mean":
            round(st.mean(r["completion_merge_s"] for r in records), 4),
        "encode_images_total_s_mean":
            round(st.mean(r["encode_images_total_s"] for r in records), 4),
        "lm_prefill_first_s_mean":
            round(st.mean(r["lm_prefill_first_s"] for r in records), 4),
    }
    return {"summary": summary, "samples": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--question-file", required=True)
    p.add_argument("--conv-mode", default="vicuna_v1")
    p.add_argument("--visual-token-num", type=int, default=0)
    p.add_argument("--anchorzip", action="store_true")
    p.add_argument("--az-mode", default="rtg", choices=["rtg", "official_replay"],
                   help="rtg=AnchorZip main; official_replay=official EADP "
                        "steps (G1-verified) for the E-side efficiency row")
    p.add_argument("--az-lam", type=float, default=None,
                   help="override Completion lam; 0.0 = hard pruning (no "
                        "completion), used with --az-mode official_replay")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--tag", required=True)
    p.add_argument("--json-out", required=True)
    args = p.parse_args()

    questions = [json.loads(q)
                 for q in open(os.path.expanduser(args.question_file))]
    samples = [{"question_id": q["question_id"],
                "question": q["text"],
                "image_path": os.path.join(args.image_folder, q["image"])}
               for q in questions[:args.n + args.warmup]]
    warm = samples[:args.warmup]
    meas = samples[args.warmup:]

    # warmup pass (separate small run, results discarded)
    res = run_arm(args, warm)
    res = run_arm(args, meas)
    res["summary"]["warmup_n"] = args.warmup
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(res, open(args.json_out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
