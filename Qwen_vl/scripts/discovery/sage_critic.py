"""
SAGE step 2 -- critic features and training (decision 7).

For each g: build phi (E, 16388) for every labeled fit/val edge from the bank's
bf16 vision cache + the captured q̄ (the SAME phi_batch the deployed pruner
runs, so there is no train/serve skew), then fit the 2-hidden-layer MLP by
squared error against Delta. Capacity (width in {256, 1024}) is chosen by RMSE
on the val edges. The UNARY control is trained on the same edges' 8 unary
features with the identical protocol -- it is the "same-label unary exchange
scorer" of the falsification design.

Outputs
    sage_phi_{split}_g{g}.npz          X, y, img, minus_flat, plus_flat, ptr
    sage_critic_g{g}_w{w}.pt           state dict + standardisation stats
    sage_unary_g{g}_w{w}.pt            the control, same protocol
    sage_critic_summary.json           val RMSE per (g, family, width)

Usage
    python scripts/discovery/sage_critic.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                        # noqa: E402
from m5_common import load_m5_bank                                   # noqa: E402
from m6_common import load_bank                                      # noqa: E402
import sage_common as S                                              # noqa: E402


def load_labels(split: str) -> dict:
    return S.load_json(S.F_LABELS.format(split=split))


def load_q(split: str) -> dict:
    z = np.load(os.path.join(OUTPUT_DIR, S.F_QBANK.replace(".npz", f"_{split}.npz")))
    return {str(k): q for k, q in zip(z["key"], z["q"])}


def build_split_features(bank, vis, qbank, labels, split: str, g: int):
    """(X, y, img, edges_flat) for every labeled edge of one (split, g)."""
    index = {k: i for i, k in enumerate(bank["key"])}
    keys = [k for k in labels["s0_hits"] if k in index]
    Xs, ys, imgs = [], [], []
    minus_flat, plus_flat, ptr = [], [], [0]
    for key in sorted(keys):
        i = index[key]
        s0 = torch.tensor(np.sort(bank["s0"][i]), dtype=torch.long)
        edges = labels["edges"][str(g)].get(key, [])
        if not edges:
            continue
        vis_i = torch.from_numpy(np.array(vis[i])).view(torch.bfloat16).float()
        q = torch.tensor(qbank[key], dtype=torch.float32)
        X = S.phi_batch(vis_i, q, s0, edges)
        Xs.append(X)
        ys.extend(e["hit"] - labels["s0_hits"][key] for e in edges)
        imgs.extend([i] * len(edges))
        for e in edges:
            minus_flat.extend(e["minus"])
            plus_flat.extend(e["plus"])
            ptr.append(len(minus_flat))
        assert len(vis_i) == 1024
    X = torch.cat(Xs).numpy().astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    img = np.array(imgs, dtype=np.int64)
    return X, y, img, (np.array(minus_flat), np.array(plus_flat), np.array(ptr))


def train_critic(Xtr, ytr, Xva, yva, d_in: int, width: int, family: str):
    torch.manual_seed(0)
    mu = Xtr.mean(0)
    sd = np.clip(Xtr.std(0), 1e-6, None)
    model = S.SageCritic(d_in, width)
    opt = torch.optim.Adam(model.parameters(), lr=S.TRAIN_LR, weight_decay=S.TRAIN_WD)
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    n = Xtr_t.shape[0]
    g = torch.Generator().manual_seed(0)
    for epoch in range(S.TRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        for lo in range(0, n, S.TRAIN_BATCH):
            idx = perm[lo:lo + S.TRAIN_BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.nn.functional.mse_loss(
            model(Xva_t), torch.tensor(yva))))
        rmse_tr = float(torch.sqrt(torch.nn.functional.mse_loss(
            model(Xtr_t), ytr_t)))
    return model, dict(val_rmse=rmse, fit_rmse=rmse_tr,
                       mu=np.asarray(mu), sd=np.asarray(sd),
                       width=width, family=family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs=2, default=("fit", "val"),
                    help="fit split name, val split name")
    args = ap.parse_args()
    fit_split, val_split = args.split

    bank = load_bank()
    vis = np.load(os.path.join(OUTPUT_DIR, "m5_vis.npy"), mmap_mode="r")
    q_fit, q_val = load_q(fit_split), load_q(val_split)
    lab_fit, lab_val = load_labels(fit_split), load_labels(val_split)
    print(f"[labels] fit {len(lab_fit['s0_hits'])} imgs / "
          f"val {len(lab_val['s0_hits'])} imgs")

    summary = dict(widths=list(S.WIDTHS), g_grid=list(S.G_GRID), families={})
    for g in S.G_GRID:
        Xtr, ytr, _, _ = build_split_features(bank, vis, q_fit, lab_fit, fit_split, g)
        Xva, yva, _, _ = build_split_features(bank, vis, q_val, lab_val, val_split, g)
        np.savez(os.path.join(OUTPUT_DIR, S.F_PHI.format(split=fit_split, g=g)),
                 X=Xtr, y=ytr)
        np.savez(os.path.join(OUTPUT_DIR, S.F_PHI.format(split=val_split, g=g)),
                 X=Xva, y=yva)
        print(f"[g={g}] fit {Xtr.shape}, val {Xva.shape}; "
              f"y fit mean {ytr.mean():+.4f} nonzero {(ytr != 0).mean():.3f}")

        fam = {}
        for family, d_in in (("set", S.D_PHI), ("unary", S.D_UNARY)):
            if family == "unary":
                # unary features are rebuilt from the bank columns + the edge
                # lists -- no vision cache needed
                Utr = _unary_matrix(bank, lab_fit, fit_split, g)
                Uva = _unary_matrix(bank, lab_val, val_split, g)
                for w in S.WIDTHS:
                    model, meta = train_critic(Utr, ytr, Uva, yva, d_in, w, family)
                    torch.save(dict(state_dict=model.state_dict(), meta=meta),
                               os.path.join(OUTPUT_DIR, f"sage_unary_g{g}_w{w}.pt"))
                    fam[f"unary_w{w}"] = meta["val_rmse"]
                    print(f"  [g={g} unary w={w}] val RMSE {meta['val_rmse']:.4f} "
                          f"(fit {meta['fit_rmse']:.4f})")
            else:
                for w in S.WIDTHS:
                    model, meta = train_critic(Xtr, ytr, Xva, yva, d_in, w, family)
                    torch.save(dict(state_dict=model.state_dict(), meta=meta),
                               os.path.join(OUTPUT_DIR, S.F_CRITIC.format(g=g, w=w)))
                    fam[f"set_w{w}"] = meta["val_rmse"]
                    print(f"  [g={g} set w={w}] val RMSE {meta['val_rmse']:.4f} "
                          f"(fit {meta['fit_rmse']:.4f})")
        summary["families"][f"g{g}"] = fam

    S.dump_json(S.F_CRITIC_SUMMARY, summary)


def _unary_matrix(bank, labels, split: str, g: int) -> np.ndarray:
    index = {k: i for i, k in enumerate(bank["key"])}
    rows = []
    for key in sorted(labels["s0_hits"]):
        i = index[key]
        s0 = torch.tensor(np.sort(bank["s0"][i]), dtype=torch.long)
        imp = torch.tensor(bank["X"][i][:, bank["fi"]["imp"]])
        cos = torch.tensor(bank["X"][i][:, bank["fi"]["cos_s0c"]])
        edges = labels["edges"][str(g)].get(key, [])
        if edges:
            rows.append(S.unary_batch(imp, cos, edges))
    return torch.cat(rows).numpy().astype(np.float32)


if __name__ == "__main__":
    main()
