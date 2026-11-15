"""CPU audit of the native adapter against the pinned official LTC kernel.

This verifies the numerical local transition operator, not an end-to-end
CatchUp reproduction or an improvement in downstream answers. No model
weights are downloaded and no CUDA operation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
PINNED = "18374ec779df0eaab07b788b777a75e1ecd812a3"


def load_official(root):
    """Load the original package under an isolated name, without patching it."""
    name = "_catchup_official_cpu_audit"
    package = types.ModuleType(name)
    package.__path__ = [str(root / "models")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".patching", root / "models/patching.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, check_name, ok, **detail):
        self.rows.append(dict(name=check_name, passed=bool(ok), **detail))


def ltc_cases(adapter, official, checks):
    # Fixed independently chosen synthetic inputs. Targets and anchors are
    # disjoint, matching newly readmitted versus already active tokens.
    cases = [
        ("many_anchors", torch.float32, 49, 12, 8),
        ("few_anchors", torch.float32, 35, 3, 8),
        ("one_anchor", torch.float32, 35, 1, 8),
        ("empty_anchors", torch.float32, 35, 0, 8),
        ("empty_targets", torch.float32, 35, 12, 0),
        ("float64_features", torch.float64, 49, 12, 8),
        ("bfloat16_features", torch.bfloat16, 49, 12, 8),
    ]
    for case_i, (name, dtype, n, na, nt) in enumerate(cases):
        rng = torch.Generator(device="cpu").manual_seed(20261108 + case_i)
        reference = torch.randn(n, 64, generator=rng, dtype=torch.float32).to(dtype)
        current = reference + .25 * torch.randn(n, 64, generator=rng).to(dtype)
        targets = torch.arange(n - nt, n, dtype=torch.long)
        anchors = torch.arange(na, dtype=torch.long)
        # Rectangular or square grid, normalized on both axes. The official
        # kernel consumes supplied coordinates and need not infer a grid.
        width = 7
        height = (n + width - 1) // width
        ids = torch.arange(n)
        xy = torch.stack(((ids // width).float() / max(height - 1, 1),
                          (ids % width).float() / max(width - 1, 1)), -1)
        before = [x.clone() for x in (reference, current, targets, anchors, xy)]
        got = adapter.ltc_states(reference, current, targets, anchors, xy,
                                 topm=8, spatial_lambda=.15, tau=.07, gamma=.8)
        readmit = torch.zeros(n, dtype=torch.bool)
        readmit[targets] = True
        source = torch.zeros(n, dtype=torch.bool)
        source[anchors] = True
        oi, ov = official._compute_local_transition_correction_for_batch(
            h_vis_b=current, readmit_mask_b=readmit, source_mask_b=source,
            coords=xy, layer_gap_b=torch.full((n,), 5),
            last_active_b=torch.full((n,), 2),
            layer_snapshots={2: reference.unsqueeze(0)}, batch_idx=0,
            topm=8, spatial_lambda=.15, tau=.07, gamma_max=.8,
            norm_preserve=False, norm_target="none", anchor_mode="semantic_spatial",
            query_relevance_b=None, query_eta=0.)
        expected_all = current.clone()
        if oi is not None:
            expected_all[oi] = ov
        expected = expected_all[targets]
        tolerance = .02 if dtype == torch.bfloat16 else 2e-6
        max_abs = float((got.float() - expected.float()).abs().max()) if nt else 0.
        same = got.shape == expected.shape and torch.allclose(
            got.float(), expected.float(), atol=tolerance, rtol=tolerance)
        checks.check("ltc_" + name, same, dtype=str(dtype), max_abs_error=max_abs,
                     atol=tolerance, rtol=tolerance, n_targets=nt, n_anchors=na)
        checks.check("immutable_" + name,
                     all(torch.equal(x, old) for x, old in
                         zip((reference, current, targets, anchors, xy), before)))
        checks.check("cpu_" + name, got.device.type == "cpu")


@torch.no_grad()
def score_cases(adapter, checks):
    """Compare the adapter to stock eager attention, including replaced keys.

    This is an independent actual Qwen3 text layer on CPU, with GQA and
    deliberately different temporal/height/width positions. It does not
    certify large-model bfloat16 rounding or image preprocessing.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261108)
        cfg = Qwen3VLTextConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, attention_dropout=0.,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]})
        cfg._attn_implementation = "eager"
        model = Qwen3VLTextModel(cfg).eval()
        eng = types.SimpleNamespace(text=model)
        hidden = torch.randn(1, 17, 64)
        replacement = hidden.clone()
        replacement[:, [4, 7, 10]] = torch.randn(1, 3, 64)
        positions = torch.stack((torch.arange(17),
                                 torch.arange(17) * 2 + 3,
                                 torch.arange(17) * 3 + 7))[:, None]
        old_hidden, old_replacement = hidden.clone(), replacement.clone()
        layer = model.layers[0]
        normed = layer.input_layernorm(hidden)
        pe = model.rotary_emb(hidden, positions)
        mask = torch.full((1, 1, 17, 17), torch.finfo(hidden.dtype).min).triu(1)
        for name, keys in (("native_keys", None), ("replaced_keys_query_fixed", replacement)):
            got = adapter.attention_scores(eng, 0, hidden, positions, 3, 13,
                                           key_hidden=keys)
            hook = None
            if keys is not None:
                # Change only the K projection's input. Stock attention still
                # computes Q from the original hidden and its q_norm.
                key_normed = layer.input_layernorm(keys)
                hook = layer.self_attn.k_proj.register_forward_pre_hook(
                    lambda module, inputs: (key_normed,))
            try:
                _, weights = layer.self_attn(normed, position_embeddings=pe,
                                             attention_mask=mask)
            finally:
                if hook is not None:
                    hook.remove()
            expected = weights.float().mean(1)[0, -1, 3:13]
            checks.check("score_" + name,
                         torch.allclose(got, expected, atol=2e-7, rtol=2e-6),
                         max_abs_error=float((got - expected).abs().max()),
                         gqa=True, distinct_three_axis_positions=True,
                         full_prompt_denominator=True)
        checks.check("score_inputs_immutable", torch.equal(hidden, old_hidden)
                     and torch.equal(replacement, old_replacement))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=QWEN / "third_party/CatchUp")
    parser.add_argument("--output", type=Path, default=QWEN / "outputs/catchup_rerank/audit_ltc.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = Checks()
    commit = subprocess.check_output(["git", "-C", str(args.official_root),
                                      "rev-parse", "HEAD"], text=True).strip()
    checks.check("official_revision", commit == PINNED, observed=commit, expected=PINNED)
    official = load_official(args.official_root)
    sys.path.insert(0, str(HERE))
    import route
    ltc_cases(route, official, checks)
    score_cases(route, checks)
    result = dict(status="PASS" if all(r["passed"] for r in checks.rows) else "FAIL",
                  seed=20261108, device="cpu", official_revision=commit,
                  source_sha256={"official_patching": sha(args.official_root / "models/patching.py"),
                                 "adapter": sha(HERE / "route.py"), "audit": sha(__file__)},
                  checks=checks.rows,
                  scope="Single L2 reference depth, no-norm, query_eta=0; LTC kernel and tiny stock Qwen3 attention scores.",
                  limitations=[
                      "Does not certify hidden-state snapshot timing, anchor freshness or routing caches.",
                      "Tiny float32 stock attention checks q_norm/k_norm/GQA/mRoPE only; does not certify real-model bfloat16 behavior, DeepStack or decode cache positions.",
                      "Mixed reference depths and end-to-end CatchUp reproduction are outside this adapter audit.",
                      "Synthetic agreement is not downstream QA or an efficiency result."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], n_checks=len(checks.rows),
                          failures=[r for r in checks.rows if not r["passed"]],
                          output=str(args.output)), indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
