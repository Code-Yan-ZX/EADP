"""Frozen arms for the native-Qwen3 fixed-selection calibration pilot."""
from __future__ import annotations

import math
import torch

from calibration import AttentionLayout, BiasSpec, attention_calibration
from rope_bias import distance_log_bias
from runtime import native_axis_map
from experiment_common import K, MAX_TOKENS

ARMS = ("full", "rres512", "baseline", "vz_maskzero", "count", "effective",
        "bias_const", "restore_prefill", "restore_continuous", "b2_baseline",
        "b2_maskzero", "b2_bias_const", "b2_restore_prefill", "b2_restore_continuous")


class RestoreDistance:
    """Cache a single position-dependent matrix for all decoder layers.

    On a given prefill/decode step every layer has identical native positions.
    The cache belongs to one single-sample inference context and never crosses
    samples. Its cost is included when timing the method end to end.
    """
    def __init__(self, rotary):
        self.inv_freq = rotary.inv_freq.detach()
        self.axes = native_axis_map(rotary).detach()
        self.signature = None
        self.value = None

    def __call__(self, frame):
        signature = (frame.query_length, frame.key_length)
        if signature != self.signature:
            self.value = distance_log_bias(frame.query_positions_3d, frame.key_positions_3d,
                                           self.inv_freq, self.axes)
            self.signature = signature
        return self.value


def carrier(sample, name):
    """Return common keep/V/DS and corresponding key multiplicities."""
    if name.startswith("b2_"):
        return sample["b2"], None, None, torch.ones(K, device=sample["V"].device)
    keep, vs, ds, _ = sample["vz"]
    weights = sample["effective"] if name == "effective" else sample["count"]
    return keep, vs, ds, weights


def specification(eng, name):
    """Return the exact, preregistered additive-bias intervention."""
    if name in ("baseline", "b2_baseline"):
        return BiasSpec()
    if name in ("vz_maskzero", "b2_maskzero"):
        return BiasSpec(force_mask=True)
    if name in ("count", "effective"):
        return BiasSpec(log_count_scale=1.0, force_mask=True)
    if name in ("bias_const", "b2_bias_const"):
        return BiasSpec(visual_bias=math.log(4.0), force_mask=True)
    if name in ("restore_prefill", "restore_continuous",
                "b2_restore_prefill", "b2_restore_continuous"):
        return BiasSpec(log_count_scale=1.0, pair_bias=RestoreDistance(eng.text.rotary_emb),
                        force_mask=True,
                        prefill_only=name.endswith("prefill"))
    raise ValueError("unregistered arm: " + name)


@torch.inference_mode()
def predict_calibrated(eng, sample, name, max_tokens=MAX_TOKENS, observer=None,
                       ignore_eos=False):
    if name not in ARMS or name in ("full", "rres512"):
        raise ValueError("not a calibration carrier arm")
    keep, vs, ds, weights = carrier(sample, name)
    layout = AttentionLayout.from_engine(eng, sample["prep"], keep, key_weights=weights)
    spec = specification(eng, name)
    with attention_calibration(eng.text, layout, spec, observer=observer):
        state = eng.prefill(sample["prep"], sample["V"], sample["DS"], keep,
                            V_sel=vs, DS_sel=ds)
        ids, text = eng.decode(state, max_tokens, ignore_eos=ignore_eos)
    if state.n_vis != K or state.ds_lengths != [K] * 3:
        raise AssertionError("compressed visual/DeepStack counts changed")
    if state.decode_positions and state.decode_positions[0] != state.prefill_max_pos + 1:
        raise AssertionError("native decode position drift")
    if int(state.cache.get_seq_length()) != state.lk + len(state.decode_positions):
        raise AssertionError("cache length does not match executed decode steps")
    return dict(text=text, token_ids=ids, truncated=len(ids) >= max_tokens,
                eos=(len(ids) < max_tokens if not ignore_eos else None),
                n_vis=state.n_vis, ds_lengths=state.ds_lengths,
                prefill_seq_length=state.lk, decode_positions=state.decode_positions,
                cache_length=int(state.cache.get_seq_length()))
