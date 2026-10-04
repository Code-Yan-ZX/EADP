"""AnchorZip port for LLaVA (P3, 2026-10-04).

Frozen math, per docs/anchorzip_llava_port_prereg.md §2 — mapping
decisions are FROZEN there and must not be changed here:

  * RTG importance replaces ONLY the official (2.1) entropy filter,
    (2.2) local aggregation and (2.3) alpha fusion.  A (negated-cosine
    text-token-to-visual sim, [N, L]) and g (global, [N]) come from the
    official sim_cross on PRE-projector image_embeds (same as the
    Qwen RTG implementation, which scores raw features).  Steps
    (2.4) min-max, (2.5) spatial smoothing, (2.6) beta and the official
    facility / quota selection are kept VERBATIM from the official
    encode_images.
  * Completion (uniform group mean, lam=0.25) runs on the
    POST-projector feature stream (the only visual stream entering the
    LLM), per image crop (split_sizes semantics), with the frozen
    compute_assignment (cos-argmax dropped->keep, gid = anchor rank)
    and merge_stream operators imported from the Qwen round.
  * FULL (visual_token_num<=0) and E_GATHER (official EADP, lam=0) are
    the official paths untouched.

The official llava_arch.py is NOT modified; this module monkey-patches
LlavaMetaForCausalLM.encode_images at load time when enabled.
"""

from __future__ import annotations

import sys

LLAVA_ROOT = "/media/disk2/YZX/research/EADP_amp/LLaVA"
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)
QWEN_SCRIPTS = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts"
for _p in ("anchor_merge_pilot", "anchor_completion_validation",
           "stage1_roundtrip_pilot"):
    if f"{QWEN_SCRIPTS}/{_p}" not in sys.path:
        sys.path.insert(0, f"{QWEN_SCRIPTS}/{_p}")

import torch  # noqa: E402

from llava.model import llava_arch as LA  # noqa: E402

INV_TEMP = 100.0            # frozen (rtg_common.py:66)
ALPHA = 0.5                 # official fusion weight
LAM = 0.25                  # frozen Completion strength
# "rtg" = AnchorZip main method; "official_replay" = the port replays the
# OFFICIAL entropy/aggregation/fusion steps inline (G1 identity gate:
# must reproduce the official encode_images keep masks bit-exactly).
MODE = "rtg"


def _text_weight_rtg(A: torch.Tensor) -> torch.Tensor:
    """Frozen rtg text weight (rtg_common.text_weight, mode='rtg'):
    A [N, L] fp32 negated-cosine -> w [L] summing to 1."""
    Af = A.float()
    logP = torch.log_softmax(INV_TEMP * Af, dim=0)          # P(i|t)
    logB = logP - torch.logsumexp(logP, dim=1, keepdim=True)  # B(t|i)
    c = torch.exp(logP + logB).sum(dim=0)                   # [L]
    return c / c.sum()


def _completion_stream(feat: torch.Tensor, keep: torch.Tensor,
                       from_amp_common) -> torch.Tensor:
    """Frozen merge_stream(kind='uniform', lam) on ONE crop stream,
    reordered to keep-index order so it matches boolean-mask gather."""
    from amp_common import compute_assignment, merge_stream
    dropped_idx, gid, _sim = compute_assignment(feat, keep)
    y = merge_stream(feat, keep, dropped_idx, gid, "uniform", LAM)
    order = torch.argsort(keep)
    return y[order].to(feat.dtype)


@torch.no_grad()
def _anchorzip_encode_images(self, images, texts=None, split_sizes=None):
    """RTG importance + official facility + Completion(0.25).  Mirrors
    the official encode_images up to stage (2.3), which is replaced."""
    model = self.get_model()
    visual_token_num = getattr(self, "visual_token_num", 0)
    beta = getattr(self, "beta", 1.0)

    # ---- official feature extraction (pre/post projector) -------------
    image_features, image_embeds, text_embeds, text_embeds_seq = \
        self.get_model().get_vision_tower()(images, texts=texts)
    B, N, C = image_features.shape
    device = image_features.device
    image_features = model.mm_projector(image_features)

    self.is_joint = False
    original_B, original_N = B, N
    K_joint, B_joint, N_joint, image_features_joint, image_embeds_joint = \
        self.joint_reshape(B, N, C, image_features, image_embeds,
                           split_sizes)
    K = K_joint
    B_f, N_f = B, N

    sim_matrix = self.sim_visual(image_features)
    global_sim, local_sim_all = self.sim_cross(
        text_embeds, text_embeds_seq, image_embeds, original_B, original_N)

    # ---- (2.1)+(2.2)+(2.3): RTG stage, or official replay for G1 ------
    if MODE == "official_replay":
        # verbatim official three steps (same ops the official
        # encode_images runs) — used only by the G1 identity gate
        filtered, top_vals = self.entrpy_filter(
            local_sim_all, T=100.0, entropy_keep_ratio=0.2)
        local_agg = self.local_aggregation(
            filtered_local_sim=filtered, top_entropy_vals=top_vals,
            M_temp=0.01, strategy="negative_entropy")
        text_sim_all = ALPHA * global_sim + (1.0 - ALPHA) * local_agg
        text_sim = text_sim_all.mean(dim=-1)
    else:
        # frozen RTG stage, PER CROP (Qwen round applies the text
        # weight per image; B here = crops of one image)
        rows = []
        for b in range(B_f):
            A_b = local_sim_all[b, :, 0, :]          # [N, L]
            g_b = global_sim[b, :, 0]                # [N]
            w_b = _text_weight_rtg(A_b)
            rows.append(ALPHA * g_b + (1.0 - ALPHA)
                        * (A_b * w_b.unsqueeze(0)).sum(dim=1))
        text_sim = torch.stack(rows)                 # [B, N]

    # ---- official (2.4)-(2.6) + facility, VERBATIM ---------------------
    text_sim = text_sim.view(B_joint, N_joint)
    importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values
                  + 1e-6) / (text_sim.max(dim=-1, keepdim=True).values
                             - text_sim.min(dim=-1, keepdim=True).values
                             + 1e-6)
    importance = importance.view(B_f, N_f)
    importance = self.spatial_smoothing(importance, kernel_size=3,
                                        sigma=1.0)
    importance = importance ** beta
    select_idx, token_quotas = self.greed_select(
        importance, sim_matrix, K=K, strategy="importance_based")

    # ---- official mask construction, VERBATIM --------------------------
    device = select_idx.device
    range_mask = torch.arange(select_idx.shape[1], device=device) \
        < token_quotas.unsqueeze(1)
    src = range_mask.float()
    index_masks = torch.zeros(B_f, N_f, dtype=torch.float32,
                              device=device)
    index_masks.scatter_(1, select_idx, src)
    index_masks = index_masks.bool()

    if self.is_joint:
        image_features = image_features.view(original_B, original_N, -1)
        index_masks = index_masks.view(original_B, original_N)

    # ---- frozen Completion on the post-projector stream ----------------
    # per crop (split_sizes); kept features are replaced by the merged
    # stream in keep-index order == boolean-mask gather order.
    if split_sizes is None:
        split_sizes = [B_f]
    feat_flat = image_features.view(original_B, original_N, -1)
    mask_flat = index_masks.view(original_B, original_N)
    out_parts, off = [], 0
    for n in split_sizes:
        f_i, m_i = feat_flat[off:off + n], mask_flat[off:off + n]
        keep = torch.nonzero(m_i.reshape(-1), as_tuple=False) \
            .squeeze(1).to(f_i.device)
        tot = f_i.shape[0] * f_i.shape[1]
        if int(keep.numel()) in (0, tot):
            out_parts.append(f_i)
        else:
            merged = _completion_stream(
                f_i.reshape(-1, f_i.shape[-1]),
                keep.to(f_i.device), None)
            f_new = f_i.reshape(-1, f_i.shape[-1]).clone()
            f_new[keep] = merged
            out_parts.append(f_new.view_as(f_i))
        off += n
    image_features = torch.cat(out_parts, dim=0)
    return image_features, index_masks


def install():
    """Monkey-patch LlavaMetaForCausalLM.encode_images with the AnchorZip
    path.  Returns a restore function."""
    orig = LA.LlavaMetaForCausalLM.encode_images
    LA.LlavaMetaForCausalLM.encode_images = _anchorzip_encode_images
    return lambda: setattr(LA.LlavaMetaForCausalLM, "encode_images", orig)
