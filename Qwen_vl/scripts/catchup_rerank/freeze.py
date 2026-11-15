"""Record the fixed exploratory configuration before running panel answers."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
OUT = QWEN / "outputs/catchup_rerank"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import transformers
    data_path = OUT / "data/manifest.json"
    gates_path = OUT / "gates.json"
    gate = json.loads(gates_path.read_text())
    if not gate.get("pass_all") or gate["route_sha256"] != sha(HERE / "route.py"):
        raise RuntimeError("Missing or stale passing gates")
    dest = OUT / "run_manifest.json"
    if dest.exists() or (OUT / "predictions.json").exists():
        raise RuntimeError("Existing freeze or panel predictions; do not overwrite")
    sources = ["data.py", "route.py", "run.py", "gates.py", "analyze.py", "benchmark.py", "audit_ltc.py", "audit_mask.py"]
    payload = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(data_path), gate_sha256=sha(gates_path),
        script_sha256={s: sha(HERE / s) for s in sources},
        python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), dtype="bfloat16", attention="sdpa",
        model="Qwen/Qwen3-VL-8B-Instruct",
        model_revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        official_code_revision="18374ec779df0eaab07b788b777a75e1ecd812a3",
        reproduction="Two-stage Qwen3 adaptation; not official LLaVA four-stage reproduction",
        source_split="16 unused DocVQA TRAIN parents, one fixed human numeric QA per parent",
        input_visual_tokens=1024, active_tokens=256, route_layers_zero_based=[2, 7],
        candidate_count=512,
        router="Last prompt token native q/k including q_norm/k_norm and mRoPE, float32 per-head softmax over all prompt keys, head-mean visual attention, torch.topk; sort selected indices in raster order",
        arms={"baseline":"rank current stale full pool",
              "dense_full":"same sparse query/active/nonvisual keys, full-model same-depth keys for deferred visuals, rank full pool; offline reference only",
              "dense_2k":"same full-pool dense-reference scores, but choose within baseline stale top 512",
              "ltc_2k":"correct deferred keys within baseline stale top 512 using LTC then choose within that shortlist; no dense-reference dependence"},
        ltc="8 nearest by cosine(reference) minus 0.15 normalized spatial distance; softmax temperature .07; add .8 weighted anchor transition, no norm rescaling; common anchors frozen to baseline S2 intersect baseline S7; selected states corrected once after choice",
        dense_reference="separate full-token input-only prefill to layer 7; replacement used only to score deferred keys; neither output state nor answer labels injected",
        prompt_suffix="\nAnswer with the value only.", generation="greedy, EOS stop, max 24 new tokens",
        exact="CF pilot Decimal-based numeric normalization: commas/currency/whitespace removed, sign and percent retained; leading zero identifiers may normalize; literal exact reported separately",
        data_filter="no model output, compression failure, teacher correctness, or truncation filtering",
        latency="batch 1, three alternating repeats on one outside-panel image, one warmup per arm; synchronized wall-clock preprocessing+vision+route+first token; model load excluded",
    )
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n")
    print(dest)


if __name__ == "__main__":
    main()
