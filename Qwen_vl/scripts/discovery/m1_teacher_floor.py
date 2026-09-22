"""
M1 gate G2 follow-up: what is the P1-G2 backward's reproducibility floor?

Gate G2 (prereg §3.3) required the extension teacher generator to reproduce
``s2b_gradient_scores.npz`` to a relative 1e-4. It failed at 3.7e-2 -- but with
``top1_match=True`` on every checked instance, i.e. the forward, the token
construction and the argmax are identical and only the *backward* moves. Before
that is called a different computation, the backward's own reproducibility has to
be measured: if this exact code path does not reproduce *itself* to 1e-4, then
1e-4 was never an achievable criterion and the gate needs a measured floor rather
than an assumed one.

Three numbers per instance:

    self     the same instance twice, same process   -> CUDA reduction noise
    repeat   the same instance twice, fresh process  -> + allocator/kernel-choice
                                                        noise
    published the published map                      -> + whatever the original
                                                        run did differently

They are reported together because the ordering is what carries the information:
if ``self`` and ``repeat`` sit at the same magnitude as ``published``, the gate
was comparing a noisy quantity against a noise-level threshold; if the first two
are ~0 and ``published`` is 1e-2, something in the two code paths really differs.

Usage
    python m1_teacher_floor.py [--n 4]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                         # noqa: E402
from common import eadp_model_name                                    # noqa: E402
from s1_audit import OUT                                              # noqa: E402
from m1_common import TEACHER                                         # noqa: E402
from m1_teacher import dataset_objects, p1g2                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="m1_teacher_floor.json")
    ap.add_argument("--g2-keys", action="store_true",
                    help="use exactly the instances gate G2 checked")
    args = ap.parse_args()

    G = np.load(os.path.join(OUT, TEACHER))
    rng = np.random.default_rng(20260924)
    if args.g2_keys:
        # the same draw m1_teacher.py's G2 used, so the two are comparable
        keys = [G.files[i] for i in sorted(rng.choice(len(G.files),
                                                      size=args.n, replace=False))]
    else:
        keys = [G.files[i] for i in sorted(rng.choice(len(G.files),
                                                      size=args.n, replace=False))]

    datasets = dataset_objects()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=2048)
    model.model.eval()
    torch_ok = True
    import torch
    torch.set_grad_enabled(True)
    for p in model.model.parameters():
        p.requires_grad_(False)
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device
    emb_layer = model.model.get_input_embeddings()

    rows = []
    for k in keys:
        ds, idx = k.rsplit("_", 1)
        maps = [p1g2(model, datasets, ds, int(idx), emb_layer, image_token_id,
                     dev)[0].astype(np.float64) for _ in range(args.repeats)]
        ref = np.asarray(G[k], dtype=np.float64)
        den = float(np.abs(ref).max())
        self_max = max(float(np.abs(maps[i] - maps[j]).max())
                       for i in range(len(maps)) for j in range(i + 1, len(maps)))
        pub_max = max(float(np.abs(m - ref).max()) for m in maps)
        row = dict(key=k, ref_max=den, n_tokens=len(ref),
                   self_max_abs=self_max, self_rel=self_max / den,
                   published_max_abs=pub_max, published_rel=pub_max / den,
                   # how many tokens have a rank-relevant disagreement
                   top32_self=int(len(set(np.argsort(-maps[0])[:32])
                                      ^ set(np.argsort(-maps[1])[:32]))),
                   top32_published=int(len(set(np.argsort(-maps[0])[:32])
                                           ^ set(np.argsort(-ref)[:32]))))
        rows.append(row)
        print(f"[floor] {k}: self rel={row['self_rel']:.3e}  "
              f"published rel={row['published_rel']:.3e}  "
              f"top32 |symdiff| self={row['top32_self']} "
              f"published={row['top32_published']}", flush=True)

    rep = dict(n=len(keys), repeats=args.repeats, rows=rows,
               worst_self_rel=max(r["self_rel"] for r in rows),
               worst_published_rel=max(r["published_rel"] for r in rows),
               worst_self_abs=max(r["self_max_abs"] for r in rows),
               worst_published_abs=max(r["published_max_abs"] for r in rows))
    with open(os.path.join(OUT, args.out), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\n[floor] worst self rel      = {rep['worst_self_rel']:.3e}")
    print(f"[floor] worst published rel = {rep['worst_published_rel']:.3e}")


if __name__ == "__main__":
    main()
