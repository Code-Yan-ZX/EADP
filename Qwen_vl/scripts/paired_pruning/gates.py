"""Native model checks run by the launcher before expensive label generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import torch

from .artifacts import atomic_json, load_protocol, sha256
from . import runtime


def run(run_dir):
    root = Path(run_dir).resolve()
    protocol = load_protocol(root)
    digest = sha256(root / "protocol.json")
    output = root / "gates/gpu.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("protocol_sha256") != digest:
            raise RuntimeError("GPU gate belongs to another protocol")
        if previous.get("status") == "PASS":
            print("native GPU gates already passed", flush=True)
            return previous
        # Retain the first failure before retrying the same frozen setup.
        attempt = 1
        while output.with_name(f"gpu.failed_{attempt}.json").exists():
            attempt += 1
        output.rename(output.with_name(f"gpu.failed_{attempt}.json"))
    payload = dict(protocol_sha256=digest, status="RUNNING", checks=[])
    try:
        eng = runtime.load_engine(protocol["model_path"])
        sys.path.insert(0, str(runtime.QWEN / "scripts/multistream_oracle"))
        from scripts.multistream_oracle.gates import Gates, stock_identity
        from scripts.multistream_oracle.oracle import teacher_native_agreement
        page = json.loads((root / "data/manifest.json").read_text())["splits"]["dev"][0]
        g = Gates()
        contexts = []
        for question in page["questions"][:2]:
            ctx = runtime.prepare(eng, page, question, protocol["max_question_tokens"])
            contexts.append(ctx)
        a, b = contexts
        g.check("same_page_two_questions_identical_vision", torch.equal(a["V"], b["V"])
                and all(torch.equal(x, y) for x, y in zip(a["DS"], b["DS"])))
        g.check("question_embedding_preserves_original_sequence", not torch.equal(
            a["question_token_ids"], b["question_token_ids"]))
        stock_identity(g, eng, a["prep"], a["V"], a["DS"], forced_steps=2)
        agreement = teacher_native_agreement(eng, a["prep"], a["V"], a["DS"],
                                             page["questions"][0]["answers"][0])
        g.check("teacher_native_gold_prefix", agreement["same_shape"]["max_abs"] <= .5
                and agreement["cached"]["max_abs"] <= .5
                and agreement["cached_nll_abs_difference"] <= .03,
                same_shape_max_abs=agreement["same_shape"]["max_abs"],
                cached_max_abs=agreement["cached"]["max_abs"],
                nll_difference=agreement["cached_nll_abs_difference"])
        teacher = runtime.teacher_scores(eng, a, page["questions"][0]["answers"][0], protocol["k"])
        g.check("dense_teacher_frozen_and_finite", teacher["finite_gradients"]
                and teacher["parameter_versions_unchanged"] and teacher["parameter_gradients_none"])
        for name in ("b1", "b2", "random"):
            keep = runtime.baseline_keep(eng, a, name, protocol["k"])
            prediction = runtime.generate(eng, a, keep, max_tokens=4, ignore_eos=True)
            g.check(f"{name}_native_K256_DS_positions_decode", prediction["actual_visual_count"] == 256
                    and prediction["ds_lengths"] == [256]*3
                    and len(prediction["keep_idx"]) == len(set(prediction["keep_idx"])) == 256
                    and len(prediction["token_ids"]) == 4,
                    actual_visual_count=prediction["actual_visual_count"],
                    ds_lengths=prediction["ds_lengths"], positions_sha256=prediction["positions_sha256"])
        with runtime.low_resolution(eng, 512):
            low = runtime.prepare(eng, page, page["questions"][0], protocol["max_question_tokens"])
            prediction = runtime.generate(eng, low, max_tokens=2, ignore_eos=True)
            g.check("Rres512_actual_K256", low["prep"]["n_vis"] == 256
                    and prediction["actual_visual_count"] == 256 and prediction["ds_lengths"] == [256]*3)
        g.check("native_model_has_no_gradients", all(p.grad is None for p in eng.model.parameters()))
        payload["checks"] = g.rows
        payload["environment"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), backend=eng.text.config._attn_implementation,
            model_path=protocol["model_path"])
        payload["status"] = "PASS" if all(row["passed"] for row in g.rows) else "FAIL"
        atomic_json(output, payload)
        if payload["status"] != "PASS":
            raise RuntimeError("Native GPU gate failed; see gates/gpu.json")
        return payload
    except Exception:
        payload.update(status="FAIL", error=traceback.format_exc())
        atomic_json(output, payload)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()
