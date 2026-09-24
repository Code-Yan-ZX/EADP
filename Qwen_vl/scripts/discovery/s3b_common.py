"""
S3-B shared machinery: the two bases, the frozen F(k) intervention family, the
Part-B arm plans, and the Part-C structure features.

MECHANISM DISCOVERY ONLY. Nothing here trains, scores or selects a new method;
every quantity is either read from a frozen cache or produced by the
budget-frozen swap protocol whose validity was established in S2-C2 / S3-A.

Protocol (docs/s3b_prereg_rules.md section 2, frozen before any number):

    F(k) = (S  minus  S_only[m-k:])  union  T_only[:k]        |F(k)| = 256

    T_only = T \\ S  in teacher-rank order      (the "missed tokens" queue)
    S_only = S \\ T  in student-rank order      (the displaced-token queue)
    m      = |T_only| = |S_only|

so every arm is fully described by the pair (adds, drops) applied to S: the
prefix family is adds = T_only[:k], drops = S_only[m-k:], and every null in
Part B keeps `drops` identical and permutes only `adds` -- which is what makes
"bundle vs same-window alternative" an instance-matched, size-matched,
removal-matched comparison.

Two bases, both delivered pre-LLM by the S3A harness (gates B2/B3 tie them to
the frozen artifacts they are supposed to reproduce):

    bank G  student = GDEP LOCAL-MLP n960 seed2 (the current student)
    bank L  student = LIN_L4 (S2-C2's student) -- the base where S2-C2 already
            counted real rescue bundles with k* <= 32, so the structure
            questions have objects to be asked about.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s1_audit import OUT                                        # noqa: E402,F401
import s3a_common as A                                          # noqa: E402,F401

DS_ALL = A.DS_ALL
BUDGET = A.BUDGET
N_VIS = A.N_VIS
GRID_HW = A.GRID_HW
THR = A.THR

TEACHER_NPZ = A.TEACHER_NPZ                                     # P1-G2 maps
STUDENT_N_ARM, STUDENT_SEED = A.STUDENT_N_ARM, A.STUDENT_SEED   # GDEP scorer
LIN_SCORES = "s2c1_scores.npz"                                  # LIN_L4 scores
LIN_PILOT = "s2c1_pilot.json"                                   # LIN_L4 sets
PROPS_NPZ = "s2c2_token_props.npz"                              # per-token props
FEAT_L2 = "s2c1_feats_L2.npy"
FEAT_L4 = "s2c1_feats_L4.npy"
VIS_CACHE = A.VIS_CACHE                                         # merged embeds
CELLSTATS = "s3b_cellstats.npz"                                 # image-side stats
CASES_JSON = "s3b_cases.json"

SEED_ARMS = 4101                                                # null-draw root
KS_SWEEP = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)                   # Part A grid
KS_EXT = (48, 64)                                               # Part A tail
KS_MAX = 32                                                     # bundle cap
N_RANDWIN, N_RANDT, N_RANDO = 6, 4, 4                           # Part B nulls
SHIFTS = (1, 2, 4)
N_BLOCKPERM, N_REMRAND = 4, 3


# ---------------------------------------------------------------------------
# grid geometry
# ---------------------------------------------------------------------------
def rc(t):
    return divmod(int(t), GRID_HW)


def _pair_xy(ts):
    xy = np.array([rc(t) for t in ts], dtype=float)
    d = np.abs(xy[:, None, :] - xy[None, :, :])
    return np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(ts), 1)], \
        np.maximum(d[..., 0], d[..., 1])[np.triu_indices(len(ts), 1)]


def pairwise_dists(tokens, metric="cheb"):
    ts = [int(t) for t in tokens]
    if len(ts) < 2:
        return np.zeros(0)
    euc, cheb = _pair_xy(ts)
    return cheb if metric == "cheb" else euc


def bbox(tokens):
    """row/col extents, inclusive: (r0, r1, c0, c1)."""
    xy = np.array([rc(t) for t in tokens], dtype=int)
    return (int(xy[:, 0].min()), int(xy[:, 0].max()),
            int(xy[:, 1].min()), int(xy[:, 1].max()))


def bbox_area(tokens):
    r0, r1, c0, c1 = bbox(tokens)
    return float((r1 - r0 + 1) * (c1 - c0 + 1))


def n_components(tokens, conn=8):
    s = set(int(t) for t in tokens)
    nbrs = ([(-1, 0), (1, 0), (0, -1), (0, 1)] if conn == 4 else
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr, dc) != (0, 0)])
    seen, n = set(), 0
    for t in s:
        if t in seen:
            continue
        n += 1
        stack = [t]
        seen.add(t)
        while stack:
            r, c = rc(stack.pop())
            for dr, dc in nbrs:
                v = (r + dr) * GRID_HW + (c + dc)
                if v in s and v not in seen:
                    seen.add(v)
                    stack.append(v)
    return n


def _frac_equal(tokens, axis):
    ts = [int(t) for t in tokens]
    if len(ts) < 2:
        return 0.0
    v = [rc(t)[axis] for t in ts]
    pairs = [(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.mean([a == b for a, b in pairs]))


def centroid(tokens):
    xy = np.array([rc(t) for t in tokens], dtype=float)
    return xy.mean(0)


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------
def rank_sets(g, s):
    gorder = np.argsort(-g, kind="stable")
    sorder = np.argsort(-s, kind="stable")
    T = [int(t) for t in gorder[:BUDGET]]
    S = [int(t) for t in sorder[:BUDGET]]
    sT, sS = set(T), set(S)
    return (T, S,
            [t for t in T if t not in sS],          # T_only, teacher-rank order
            [t for t in S if t not in sT],          # S_only, student-rank order
            [t for t in T if t in sS])              # C


def load_bank(bank, keys=None):
    """per-instance records for one base, from frozen caches (CPU).

    `keys` restricts the (expensive) bank-G scorer pass to the instances a
    stage actually needs; `None` means all 150.
    """
    meta, plan, keys_all, rows_of = A.load_m1_plan()
    test_rows = rows_of["test"]
    if keys is not None:
        want = set(keys)
        test_rows = [r for r in test_rows if keys_all[r] in want]
    Z = np.load(os.path.join(OUT, TEACHER_NPZ))
    if bank == "G":
        import torch
        from m2_gdep import build_scorer
        scorer, mu, sd = build_scorer(STUDENT_N_ARM, STUDENT_SEED, "cpu")
        H = np.load(os.path.join(OUT, FEAT_L4), mmap_mode="r")
        scores = {}
        with torch.no_grad():
            for r in test_rows:
                h = torch.from_numpy(np.asarray(H[r], dtype=np.float32))
                scores[keys_all[r]] = scorer(((h - mu) / sd).unsqueeze(0),
                                             None)[0].float().numpy()
    elif bank == "L":
        Zs = np.load(os.path.join(OUT, LIN_SCORES))
        pilot = json.load(open(os.path.join(OUT, LIN_PILOT)))["runs"]
        order_of = {}
        for ds in DS_ALL:
            r = pilot[f"b{BUDGET}|topk|LIN_L4|{ds}"]
            for n, i in enumerate(r["idx"]):
                order_of[f"{ds}_{i}"] = [int(t) for t in r["select_idx"][n]]
        scores = {k: Zs[f"LIN_L4__{k}"].astype(np.float64) for k in order_of}
    else:
        raise KeyError(bank)

    out = []
    for r in test_rows:
        key = keys_all[r]
        g = Z[key].astype(np.float64)
        s = np.asarray(scores[key], dtype=np.float64)
        T, S, T_only, S_only, C = rank_sets(g, s)
        if bank == "L":
            # the S2-C2 convention was "list order == rank order"; verified here
            # (0/150 mismatch against the pilot select_idx lists).
            assert S == order_of[key], key
        ds, idx = key.rsplit("_", 1)
        out.append(dict(key=key, bank=bank, ds=ds, idx=int(idx), row=int(r),
                        g=g, s=s, T=T, S=S, T_only=T_only, S_only=S_only,
                        C=C, m=len(T_only)))
    return out


def by_key(bank):
    return {r["key"]: r for r in load_bank(bank)}


def frozen_outcomes():
    """per-instance hits of the frozen arms the classes are defined on."""
    oc = A.load_outcomes()
    ident = json.load(open(os.path.join(OUT, "s2c2_identity.json")))["runs"]
    pil_t = json.load(open(os.path.join(OUT, "s2c0_pilot.json")))["runs"]
    pil_l = json.load(open(os.path.join(OUT, LIN_PILOT)))["runs"]
    out = {}
    for ds in DS_ALL:
        for name, rec in (("identity_T", ident[f"identity_T|{ds}"]),
                          ("pilot_T", pil_t[f"b{BUDGET}|topk|P1G2|{ds}"]),
                          ("pilot_L", pil_l[f"b{BUDGET}|topk|LIN_L4|{ds}"])):
            for n, i in enumerate(rec["idx"]):
                out.setdefault(f"{ds}_{i}", {})[name] = float(rec["hits"][n])
    for key, d in out.items():
        d["B0"] = oc["B0"][key]
        d["B1"] = oc["B1"][key]
        d["G_engine"] = oc["C1-P|s2"][key]
    return out


def s2c2_bundles():
    """bank-L rescue inventory, recomputed on the S2-C2 FINE grid (free).

    S2-C2's plan file put first_rescue_k on the coarse grid; its refinement run
    recorded the whole `teacher:k` family for k = 1..96, so the true minimal k
    under the same protocol is available without any GPU work.
    """
    runs = json.load(open(os.path.join(OUT, "s2c2_rescue.json")))["runs"]
    plan = json.load(open(os.path.join(OUT, "s2c2_rescue_plan.json")))
    curves = {}
    for arm, r in runs.items():
        if not arm.startswith("teacher:"):
            continue
        k = int(arm.split(":")[1].split("|")[0])
        for key, h in zip(r["keys"], r["hits"]):
            curves.setdefault(key, {})[k] = float(h)
    out = {}
    for rec in plan["rescuable"]:
        key = rec["key"]
        cur = curves.get(key, {})
        ks_hit = sorted(k for k, h in cur.items() if h >= THR)
        out[key] = dict(key=key, ds=rec["ds"], m=rec["m"],
                        hit_student=rec["hit_student"],
                        hit_teacher=rec["hit_teacher"],
                        kstar=(ks_hit[0] if ks_hit else None),
                        curve={str(k): v for k, v in sorted(cur.items())},
                        block_s2c2=rec["block"],
                        first_coarse=rec["first_rescue_k"])
    return out


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def prefix_arms(rec, k):
    k = min(int(k), rec["m"])
    return list(rec["T_only"][:k]), list(rec["S_only"][rec["m"] - k:])


def apply_arms(rec, adds, drops):
    """the delivered 256-token set of one arm; refuses malformed arms."""
    adds = [int(t) for t in adds]
    drops = [int(t) for t in drops]
    S = set(int(t) for t in rec["S"])
    assert len(adds) == len(drops), (len(adds), len(drops))
    assert len(set(adds)) == len(adds) and len(set(drops)) == len(drops)
    assert not (set(adds) & S), f"add already retained: {set(adds) & S}"
    assert set(drops) <= S, "drop not retained"
    sel = sorted((S - set(drops)) | set(adds))
    assert len(sel) == BUDGET, len(sel)
    return sel


def _rng(rec, tag):
    h = sum((i + 1) * ord(ch) for i, ch in enumerate(tag)) % 100003
    return np.random.default_rng(SEED_ARMS + rec["idx"] * 977 + h)


def window(rec, k):
    """the teacher-rank window a bundle of size k sits in."""
    return min(rec["m"], max(2 * k, k + 16))


def partb_plan(rec, kstar):
    """frozen Part-B arm list for one bundle (docs section 4)."""
    k = int(kstar)
    adds0, drops0 = prefix_arms(rec, k)
    arms = [dict(name="full", adds=adds0, drops=drops0, kind="bundle")]
    for j in range(min(k, 8)):
        arms.append(dict(name=f"single{j}", adds=[rec["T_only"][j]],
                         drops=[rec["S_only"][rec["m"] - 1]], kind="single"))
    for j in range(k):
        arms.append(dict(name=f"loo{j}",
                         adds=adds0[:j] + adds0[j + 1:],
                         drops=drops0[:j] + drops0[j + 1:], kind="loo"))
    W = window(rec, k)
    G = set(adds0)
    win_pool = [t for t in rec["T_only"][:W] if t not in G]
    uni_pool = [t for t in rec["T_only"] if t not in G]
    sT = set(rec["T"])
    out_pool = [t for t in range(N_VIS)
                if t not in set(rec["S"]) and t not in sT]
    for r in range(N_RANDWIN):
        pick = _rng(rec, f"randwin{r}").choice(win_pool, size=k, replace=False)
        arms.append(dict(name=f"randwin{r}", adds=[int(t) for t in pick],
                         drops=drops0, kind="null_window", window=W))
    for r in range(N_RANDT):
        pick = _rng(rec, f"randT{r}").choice(uni_pool, size=k, replace=False)
        arms.append(dict(name=f"randT{r}", adds=[int(t) for t in pick],
                         drops=drops0, kind="null_uni"))
    for r in range(N_RANDO):
        pick = _rng(rec, f"randO{r}").choice(out_pool, size=k, replace=False)
        arms.append(dict(name=f"randO{r}", adds=[int(t) for t in pick],
                         drops=drops0, kind="null_out"))
    for a in SHIFTS:
        if a + k <= rec["m"]:
            arms.append(dict(name=f"shift{a}", adds=list(rec["T_only"][a:a + k]),
                             drops=drops0, kind="shift"))
    b = max(1, k // 2)
    tail_pool = list(rec["T_only"][k:k + 2 * b])
    if len(tail_pool) >= b and k - b >= 1:
        keep = adds0[:k - b]
        for r in range(N_BLOCKPERM):
            pick = _rng(rec, f"blockperm{r}").choice(tail_pool, size=b,
                                                     replace=False)
            arms.append(dict(name=f"blockperm{r}",
                             adds=list(keep) + [int(t) for t in pick],
                             drops=drops0, kind="blockperm"))
    for r in range(N_REMRAND):
        pick = _rng(rec, f"remrand{r}").choice(rec["S"], size=k, replace=False)
        arms.append(dict(name=f"remrand{r}", adds=adds0,
                         drops=[int(t) for t in pick], kind="removal_control"))
    for a in arms:
        apply_arms(rec, a["adds"], a["drops"])
    return arms


# ---------------------------------------------------------------------------
# image-side per-token statistics (content proxies; no OCR engine available)
# ---------------------------------------------------------------------------
def load_cellstats():
    p = os.path.join(OUT, CELLSTATS)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z["stats"][n] for n, k in enumerate(list(z["keys"]))}


def row_pil(ds_obj, idx):
    """the dataset's own image for row `idx`, as an RGB PIL image (dump_image
    is what every stage of this project has used to materialise the frames)."""
    from PIL import Image
    path = ds_obj.dump_image(ds_obj.data.iloc[int(idx)])
    if isinstance(path, (list, tuple)):
        path = path[0]
    return Image.open(path).convert("RGB")


def canonical_frame(ds_obj, idx, side=1024):
    """the `side`x`side` RGB image the vision tower actually receives: the
    wrapper's own scale-longest-side + expand2square + resize."""
    from vlmeval.vlm.qwen3_vl.model_fixed_res import expand2square
    im = row_pil(ds_obj, idx)
    w, h = im.size
    sc = side / max(w, h)
    im = im.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))))
    return expand2square(im, (125, 125, 125)).resize((side, side))


def build_cellstats():
    """(150, 1024, 3): luminance mean / std / edge energy of the 32x32-pixel
    block each merged token covers in the 1024x1024 frame the model actually
    sees (same expand2square + resize the wrapper applies)."""
    import common

    path = os.path.join(OUT, CELLSTATS)
    have = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        have = {str(k): n for n, k in enumerate(list(z["keys"]))}
        arr = z["stats"]
    meta, plan, keys, rows_of = A.load_m1_plan()
    todo = [k for k in (keys[r] for r in rows_of["test"]) if k not in have]
    if not todo:
        print("[cellstats] complete")
        return
    ds_objs = {ds: common.build_dataset(ds) for ds in DS_ALL}
    cell, side = GRID_HW, 1024
    new = np.zeros((len(todo), N_VIS, 3), dtype=np.float32)
    for n, key in enumerate(todo):
        ds, idx = key.rsplit("_", 1)
        im = canonical_frame(ds_objs[ds], int(idx), side=side)
        g = np.asarray(im, dtype=np.float32).mean(-1)
        blk = g.reshape(side // cell, cell, side // cell, cell) \
            .transpose(0, 2, 1, 3).reshape(-1, cell, cell)
        new[n, :, 0] = blk.mean((1, 2))
        new[n, :, 1] = blk.std((1, 2))
        new[n, :, 2] = (np.abs(np.diff(blk, axis=2)).mean((1, 2))
                        + np.abs(np.diff(blk, axis=1)).mean((1, 2)))
        if (n + 1) % 25 == 0:
            print(f"  [cellstats] {n + 1}/{len(todo)}")
    all_keys = [keys[r] for r in rows_of["test"]]
    merged = np.stack([arr[have[k]] if k in have else new[todo.index(k)]
                       for k in all_keys])
    np.savez_compressed(path, keys=np.array(all_keys),
                        stats=merged.astype(np.float32))
    print(f"[cellstats] {len(all_keys)} instances -> {path}")


# ---------------------------------------------------------------------------
# Part-C feature vectors
# ---------------------------------------------------------------------------
_PROPS = None
_F4 = None
_PROPS_IDX = None


def _lazy():
    """s2c2_token_props.npz is indexed by ITS OWN key order (0..149 over the
    held-out bank), while the feature caches are indexed by m1-plan row -- the
    two index spaces differ, so the props table needs its own lookup."""
    global _PROPS, _F4, _PROPS_IDX
    if _PROPS is None:
        _PROPS = np.load(os.path.join(OUT, PROPS_NPZ), allow_pickle=True)
        _F4 = np.load(os.path.join(OUT, FEAT_L4), mmap_mode="r")
        _PROPS_IDX = {str(k): n for n, k in enumerate(list(_PROPS["keys"]))}
    return _PROPS, _F4


_GR, _UNIT, _KEEP, _BACK = {}, {}, {}, {}


def _grank(rec):
    """teacher rank map, cached per instance (a 1024-log sort per arm is the
    single most expensive thing in set_features otherwise)."""
    key = rec["key"]
    if key not in _GR:
        _GR[key] = {int(t): int(r) for r, t in
                    enumerate(np.argsort(-rec["g"], kind="stable"))}
    return _GR[key]


def _backbone(rec):
    key = rec["key"]
    if key not in _BACK:
        g = rec["g"]
        _BACK[key] = sorted(rec["C"], key=lambda t: -g[t])[:32]
    return _BACK[key]


def _kept(rec, exclude):
    key = rec["key"]
    if key not in _KEEP:
        _KEEP[key] = np.array([rc(t) for t in rec["S"]], dtype=float)
    kk = _KEEP[key]
    if exclude:
        mask = np.array([t not in exclude for t in rec["S"]])
        kk = kk[mask]
    return kk


def _mean_unit_row(row):
    """mean over ALL 1024 tokens of their unit L4 vectors. Only this 4096-vector
    is cached: keeping the normalised 1024x4096 matrix per instance (>16 MB
    each, ~110 instances) is what OOM-killed the first analysis run."""
    if row not in _UNIT:
        H = np.asarray(_F4[row], dtype=np.float32)
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
        _UNIT[row] = H.mean(0)
    return _UNIT[row]


FEATURE_KEYS = [
    "sp_mean_cheb", "sp_mean_euc", "sp_min_cheb", "sp_frac_adjacent",
    "sp_bbox_area", "sp_bbox_over_k", "sp_bbox_aspect", "sp_ncomp8",
    "sp_ncomp4", "sp_comp_over_k", "sp_frac_same_row", "sp_frac_same_col",
    "sp_centroid_r", "sp_dist_backbone", "sp_nn_retained",
    "rk_teacher_rank_mean", "rk_teacher_rank_span", "rk_tonly_pos_mean",
    "rk_tonly_contig", "sc_g_mean", "sc_g_min", "sc_g_span", "sc_s_mean",
    "sc_s_vs_S", "sc_mass_frac", "ft_l4_norm", "ft_l2_norm", "ft_dnorm", "ft_cos_l2_l4",
    "ft_sim_sel_S", "ft_sim_sel_T", "ft_nb_sim", "ft_uniq", "ft_g2pct",
    "ft_linpct", "ft_mean_pair_cos_l4", "ft_cos_to_all_mean", "ct_lum_mean",
    "ct_lum_std", "ct_edge_mean", "ct_edge_hi_frac", "ct_same_line"]


def set_features(rec, tokens, cells=None, all_stats=None):
    """every Part-C property of one added set (bundle or null). An empty set
    (the k=1 bundle with its only member ablated == the base set) has no
    geometry: every property is NaN, and the tests filter on finiteness."""
    ts = [int(t) for t in tokens]
    if not ts:
        return {k: float("nan") for k in FEATURE_KEYS}
    props, F4 = _lazy()
    ts = [int(t) for t in tokens]
    k = len(ts)
    g, s, row = rec["g"], rec["s"], rec["row"]
    prow = _PROPS_IDX[rec["key"]]
    f = {}
    # ---- spatial
    cheb = pairwise_dists(ts, "cheb")
    euc = pairwise_dists(ts, "euc")
    f["sp_mean_cheb"] = float(cheb.mean()) if k > 1 else 0.0
    f["sp_mean_euc"] = float(euc.mean()) if k > 1 else 0.0
    f["sp_min_cheb"] = float(cheb.min()) if k > 1 else 0.0
    f["sp_frac_adjacent"] = float((cheb <= 1).mean()) if k > 1 else 0.0
    if k:
        r0, r1, c0, c1 = bbox(ts)
        f["sp_bbox_area"] = float((r1 - r0 + 1) * (c1 - c0 + 1))
        f["sp_bbox_over_k"] = f["sp_bbox_area"] / k
        f["sp_bbox_aspect"] = float((r1 - r0 + 1) / (c1 - c0 + 1))
        f["sp_ncomp8"] = float(n_components(ts, 8))
        f["sp_ncomp4"] = float(n_components(ts, 4))
        f["sp_comp_over_k"] = f["sp_ncomp8"] / k
    else:
        for kk in ("sp_bbox_area", "sp_bbox_over_k", "sp_bbox_aspect",
                   "sp_ncomp8", "sp_ncomp4", "sp_comp_over_k"):
            f[kk] = float("nan")
    f["sp_frac_same_row"] = _frac_equal(ts, 0)
    f["sp_frac_same_col"] = _frac_equal(ts, 1)
    cen = centroid(ts) if k else np.array([(GRID_HW - 1) / 2] * 2)
    f["sp_centroid_r"] = float(np.hypot(*(cen - (GRID_HW - 1) / 2)))
    backbone = _backbone(rec)
    f["sp_dist_backbone"] = float(np.hypot(*(cen - centroid(backbone)))) \
        if backbone else float("nan")
    kk = _kept(rec, set(ts))            # (n_kept, 2) grid coordinates
    if len(kk) and k:
        dd = [float(np.min(np.maximum(np.abs(kk[:, 0] - rc(t)[0]),
                                      np.abs(kk[:, 1] - rc(t)[1]))))
              for t in ts]
        f["sp_nn_retained"] = float(np.mean(dd))
    else:
        f["sp_nn_retained"] = float("nan")
    # ---- rank / score
    gr = _grank(rec)
    ranks = np.array([gr[t] for t in ts], dtype=float)
    f["rk_teacher_rank_mean"] = float(ranks.mean())
    f["rk_teacher_rank_span"] = float(ranks.max() - ranks.min()) if k else 0.0
    tonly = rec["T_only"]
    pos = {t: i for i, t in enumerate(tonly)}
    ins = np.array([pos[t] for t in ts if t in pos], dtype=float)
    f["rk_tonly_pos_mean"] = float(ins.mean()) if ins.size else float("nan")
    f["rk_tonly_contig"] = float(ins.size == k and k > 0 and
                                 (np.sort(ins) == np.arange(k)).all())
    gv = np.array([g[t] for t in ts])
    sv = np.array([s[t] for t in ts])
    f["sc_g_mean"] = float(gv.mean())
    f["sc_g_min"] = float(gv.min())
    f["sc_g_span"] = float(gv.max() - gv.min())
    f["sc_s_mean"] = float(sv.mean())
    f["sc_s_vs_S"] = float(sv.mean() - np.mean([s[t] for t in rec["S"]]))
    # how much of THIS IMAGE's total teacher mass the set carries: the
    # scale-free version of "is the evidence strong here", and the competing
    # explanation for any spatial signal (concentration, not geometry).
    f["sc_mass_frac"] = float(gv.sum() / (g.sum() + 1e-12))
    # ---- per-token cached properties (all pre-LLM)
    for name, key in (("ft_l4_norm", "n_l4"), ("ft_l2_norm", "n_l2"),
                      ("ft_dnorm", "d_norm"), ("ft_cos_l2_l4", "cos_l2_l4"),
                      ("ft_sim_sel_S", "sim_sel_S"),
                      ("ft_sim_sel_T", "sim_sel_T"), ("ft_nb_sim", "nb_sim"),
                      ("ft_uniq", "uniq"), ("ft_g2pct", "g2_pct"),
                      ("ft_linpct", "lin_pct")):
        f[name] = float(np.mean(props[key][prow, ts]))
    H4 = np.asarray(F4[row][ts], dtype=np.float32)
    Hn = H4 / (np.linalg.norm(H4, axis=1, keepdims=True) + 1e-6)
    f["ft_mean_pair_cos_l4"] = float(
        (Hn @ Hn.T)[np.triu_indices(k, 1)].mean()) if k > 1 else 0.0
    mu = _mean_unit_row(row)
    f["ft_cos_to_all_mean"] = float((Hn @ mu).mean())
    # ---- content proxies (NaN-filled when the cell-statistics cache is
    # missing an instance, so every row keeps the same key set)
    if cells is None:
        for kk in ("ct_lum_mean", "ct_lum_std", "ct_edge_mean",
                   "ct_edge_hi_frac", "ct_same_line"):
            f[kk] = float("nan")
    if cells is not None:
        c = cells[ts]
        f["ct_lum_mean"] = float(c[:, 0].mean())
        f["ct_lum_std"] = float(c[:, 1].mean())
        f["ct_edge_mean"] = float(c[:, 2].mean())
        hi = float(np.percentile(cells[:, 2], 90))
        f["ct_edge_hi_frac"] = float((c[:, 2] > hi).mean())
        f["ct_same_line"] = float(k > 1 and
                                  len({t // GRID_HW for t in ts}) == 1)
    return f


def bundle_and_null_features(rec, arms, cells):
    """features for every Part-B arm of one instance (bundle + nulls)."""
    out = {}
    pool = set()
    for a in arms:
        for t in a["adds"]:
            pool.add(int(t))
    stats = None
    for a in arms:
        out[a["name"]] = set_features(rec, a["adds"], cells=cells,
                                      all_stats=stats)
        out[a["name"]]["kind"] = a["kind"]
    return out
