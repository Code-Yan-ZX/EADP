"""Save ||v_i||_2 of the 1024 visual tokens for each causal case (vision tower only)."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name
from s1_audit import OUT


def main():
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    probe = json.load(open(os.path.join(OUT, "fa_probe.json"), encoding="utf-8"))
    cases = sorted((ds, int(k)) for ds, per in probe.items()
                   for k, v in per.items() if v["needed_blocks"])

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=8)
    out = {}
    by_ds = {}
    for ds, idx in cases:
        by_ds.setdefault(ds, []).append(idx)
    for ds, idxs in by_ds.items():
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for idx in idxs:
            row = dataset.data.iloc[idx]
            msg = common.build_message(model, dataset, ds, row)
            inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
            with torch.no_grad():
                pv = inputs["pixel_values"].type(model.model.visual.dtype)
                vis = unwrap_visual_output(
                    model.model.visual(pv, grid_thw=inputs["image_grid_thw"]))
            out[f"{ds}_{idx}"] = vis.float().norm(dim=-1).cpu().numpy()
            print(f"{ds}_{idx}: {tuple(vis.shape)} norm mean {out[f'{ds}_{idx}'].mean():.3f}")
    np.savez(os.path.join(OUT, "s2a_input_norm.npz"), **out)
    print(f"[saved] {os.path.join(OUT, 's2a_input_norm.npz')}")


if __name__ == "__main__":
    main()
