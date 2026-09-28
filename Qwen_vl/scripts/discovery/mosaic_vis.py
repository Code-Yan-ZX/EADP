#!/usr/bin/env python3
"""
MosaicPrune phase-1 diagnostic visualization.

For a handful of deterministic samples per dataset, rebuilds the adaptive
partition on the real post-merger features and renders the leaf map over the
preprocessed 1024x1024 image: each leaf outlined, filled by its side length
(coarse = dark, native-res = bright), so you can see blank areas collapsing to
big blocks and text/detail areas refined to 1x1 leaves.

No beautification -- diagnostic only. Output:
    outputs/discovery/mosaic/vis/{dataset}_{idx}_{mode}_{K}.png

Usage:
    python scripts/discovery/mosaic_vis.py \
        --datasets DocVQA_VAL OCRBench TextVQA_VAL --per-dataset 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from model.mosaic import build_partition, leaves_to_mask  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import (  # noqa: E402
    QWEN3_FIXED_RESOLUTION,
    expand2square,
    unwrap_visual_output,
)

OUT_DIR = os.path.join(common.OUTPUT_DIR, "mosaic", "vis")

MODES = ("dispersion", "uniform", "random")
K = 256

# leaf side length -> grayscale value (bigger leaf = coarser = darker)
SIDE_SHADE = {1: 220, 2: 160, 4: 110, 8: 70, 16: 40, 32: 20}


def leaf_shade(lf) -> int:
    side = max(lf.h, lf.w)
    return SIDE_SHADE.get(side, 20)


def render(image: Image.Image, leaves, grid_h: int, grid_w: int,
           title: str) -> Image.Image:
    mask = leaves_to_mask(leaves, grid_h, grid_w)
    shade = np.zeros_like(mask, dtype=np.uint8)
    for i, lf in enumerate(leaves):
        shade[mask == i] = leaf_shade(lf)

    overlay = Image.fromarray(shade, mode="L").resize(
        (QWEN3_FIXED_RESOLUTION, QWEN3_FIXED_RESOLUTION), Image.NEAREST
    ).convert("RGB")
    base = image.convert("RGB")
    blended = Image.blend(base, overlay, alpha=0.45)

    draw = ImageDraw.Draw(blended)
    scale = QWEN3_FIXED_RESOLUTION / grid_w
    scale_y = QWEN3_FIXED_RESOLUTION / grid_h
    for lf in leaves:
        x0, y0 = lf.left * scale, lf.top * scale_y
        x1, y1 = (lf.left + lf.w) * scale, (lf.top + lf.h) * scale_y
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 40, 40), width=1)
    draw.text((8, 8), title, fill=(255, 255, 0))
    return blended


def get_processed_image(model, message, dataset_name: str):
    """Reproduce the wrapper's preprocessing and run the vision tower."""
    messages = model._build_messages(message, dataset=dataset_name)
    inputs = model._processor_inputs(messages)
    pixel_values = inputs["pixel_values"].type(model.model.visual.dtype)
    grid_thw = inputs["image_grid_thw"]
    feats = unwrap_visual_output(model.model.visual(pixel_values, grid_thw=grid_thw))
    return inputs, feats, grid_thw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["DocVQA_VAL", "OCRBench", "TextVQA_VAL"])
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--budget", type=int, default=K)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = common.load_model("Qwen3-VL-8B-Mosaic-256-dispersion",
                              max_new_tokens=32)
    common.build_dataset(args.datasets[0])  # warm the dataset cache paths

    for dataset_name in args.datasets:
        dataset, idxs, rows, messages = common.collect_samples(
            model, dataset_name, args.per_dataset, offset=args.offset)
        for row_i, (idx, message) in enumerate(zip(idxs, messages)):
            inputs, feats, grid_thw = get_processed_image(
                model, message, dataset_name)
            t_i, h_i, w_i = grid_thw[0].tolist()
            grid_h, grid_w = int(h_i) // 2, int(w_i) // 2
            img_feats = feats

            # recover the preprocessed 1024x1024 image for the overlay
            from qwen_vl_utils import process_vision_info
            msgs = model._build_messages(message, dataset=dataset_name)
            images, _, _ = process_vision_info(msgs, image_patch_size=16)
            from vlmeval.vlm.qwen3_vl.model_fixed_res import get_spatial_merge_size  # noqa
            proc_img = expand2square(images[0]).resize(
                (QWEN3_FIXED_RESOLUTION, QWEN3_FIXED_RESOLUTION))

            for mode in MODES:
                leaves = build_partition(img_feats, grid_h, grid_w,
                                         args.budget, mode, seed=0)
                canvas = render(proc_img, leaves, grid_h, grid_w,
                                f"{dataset_name} idx={idx} {mode} K={args.budget}")
                out = os.path.join(
                    OUT_DIR, f"{dataset_name}_{idx}_{mode}_K{args.budget}.png")
                canvas.save(out)
                sizes = {}
                for lf in leaves:
                    sizes[max(lf.h, lf.w)] = sizes.get(max(lf.h, lf.w), 0) + 1
                print(f"{out}  sides={dict(sorted(sizes.items()))}")


if __name__ == "__main__":
    main()
