#!/usr/bin/env python3
"""AnchorZip Figure-1 round 2026-10-07 — Stage D (packaging + verification).

Takes the finalist sample ids (Stage C PASS + recorded manual ranking),
re-opens every final npz, re-renders a thumbnail from the real arrays,
builds the delivery tree:

    round_20261007/delivery/
        README.md                      (screening scope, rules, rationale,
                                        top-2 recommendation, simple table)
        inventory.csv
        screening_round20261007.csv    (real-array metrics, Stage C)
        stageA_pool.csv / stageA_batch1.csv  (CPU pre-screen)
        finalists.csv
        samples/<sid>/  (source_original.*, sample_aligned.png, sample.npz,
                         anchor_update.npz, sample.json, checks.json,
                         token_ranking.csv, fig1_metrics.json, preview.png)
        contact_sheet.png
        scripts/        (all round scripts, byte-exact)
        SHA256SUMS

zips to outputs/figures/anchorzip/anchorzip_fig1_candidates_20261007.zip,
then unpacks the zip into a temp dir and re-verifies members, arrays and
hashes.  The zip's own SHA256 is printed (outside the zip, no circularity).

Usage:
    python fig1_round20261007_stageD.py SID1 SID2 ...
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/figures/anchorzip"
SV3 = os.path.join(FIG_DIR, "samples_v3")
ROUND_DIR = os.path.join(FIG_DIR, "round_20261007")
DELIV = os.path.join(ROUND_DIR, "delivery")
ZIP = os.path.join(FIG_DIR, "anchorzip_fig1_candidates_20261007.zip")
ROUND_SCRIPTS = [
    "fig1_round20261007_stageA.py",
    "export_anchorzip_figure_sample_v3.py",
    "fig1_round20261007_stageC.py",
    "fig1_round20261007_stageD.py",
]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_sample(d: str, sid: str) -> dict:
    """Re-open the final npz files and verify the desktop contract keys."""
    z = np.load(os.path.join(d, "sample.npz"), allow_pickle=False)
    au = np.load(os.path.join(d, "anchor_update.npz"), allow_pickle=False)
    meta = json.load(open(os.path.join(d, "sample.json")))
    checks = json.load(open(os.path.join(d, "checks.json")))
    req_npz = ["response", "weights", "global_response", "local_response",
               "fused_response", "importance", "keep", "dropped_idx", "gid",
               "group", "group_size"]
    req_au = ["highlight_anchor_slot", "highlight_member_indices",
              "highlight_v_a", "highlight_mu_a", "highlight_y_a", "lam"]
    missing = [k for k in req_npz if k not in z] + \
              [k for k in req_au if k not in au]
    assert not missing, f"{sid}: missing npz keys {missing}"
    H, W, L = z["response"].shape
    N, K = int(z["n_vis"]), int(z["keep"].shape[0])
    assert (H, W) == (32, 32) and z["weights"].shape == (L,)
    assert z["keep"].shape == (K,) and z["group"].shape == (H, W)
    assert z["group_size"].sum() == N
    assert checks["all_pass"], f"{sid}: checks not all_pass"
    # w still sums to 1, group/keep identities hold on the packaged bytes
    assert abs(float(z["weights"].sum()) - 1.0) < 1e-5
    assert all(int(z["group"].ravel()[z["keep"][j]]) == j for j in range(K))
    # render a small verification thumbnail from the packaged arrays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(z["importance"], cmap="magma")
    ax.axis("off")
    fig.savefig(os.path.join(d, "_verify_thumb.png"), dpi=40)
    plt.close(fig)
    return {"sample_id": sid, "N": N, "K": K, "L": L,
            "question": meta["question"], "dataset": meta["dataset"],
            "w_top1": meta["display_token_indices"][0],
            "highlight_slot": int(au["highlight_anchor_slot"]),
            "npz_keys": sorted(z.files), "verified": True}


def main() -> None:
    args = sys.argv[1:]
    top2 = []
    if "--top2" in args:
        i = args.index("--top2")
        top2 = args[i + 1:i + 3]
        args = args[:i] + args[i + 3:]
    finalists = args
    assert finalists, "pass the finalist sample ids"
    if os.path.exists(DELIV):
        shutil.rmtree(DELIV)
    os.makedirs(os.path.join(DELIV, "samples"))
    os.makedirs(os.path.join(DELIV, "scripts"))

    # ---- copy finalists + verify ------------------------------------------
    ver = []
    for sid in finalists:
        src = os.path.join(SV3, sid)
        dst = os.path.join(DELIV, "samples", sid)
        shutil.copytree(src, dst)
        thumb = os.path.join(dst, "_verify_thumb.png")
        if os.path.exists(thumb):
            os.remove(thumb)
        ver.append(verify_sample(dst, sid))
        # remove the verification artefact again (it was written into dst)
        if os.path.exists(thumb):
            os.remove(thumb)

    # ---- CSVs ---------------------------------------------------------------
    for fn in ("screening_round20261007.csv", "stageA_pool.csv",
               "stageA_batch1.csv", "stageA_batch2.csv"):
        p = os.path.join(ROUND_DIR, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(DELIV, fn))
    with open(os.path.join(DELIV, "finalists.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ver[0].keys()))
        w.writeheader()
        w.writerows(ver)

    # ---- inventory.csv -------------------------------------------------------
    OUT = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs"
    inv = []
    for root, _, files in os.walk(DELIV):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            inv.append({"artifact": os.path.relpath(p, DELIV),
                        "bytes": os.path.getsize(p),
                        "sha256": sha256_file(p), "note": ""})
    for ds in ("TextVQA_VAL", "OCRBench", "DocVQA_VAL", "ChartQA_TEST",
               "AI2D_TEST", "InfoVQA_VAL"):
        for kind, p in [
            ("bank_full_rtg", os.path.join(
                OUT, "stage1_roundtrip_pilot", "full",
                f"bank_full_rtg_{ds}.json.gz")),
            ("L_R_MAIN025_pred", os.path.join(
                OUT, "stage1_roundtrip_pilot", "legacy_full", "acc",
                "L_R_MAIN025", f"{ds}.json")),
            ("L_R_MAIN025_score", os.path.join(
                OUT, "stage1_roundtrip_pilot", "legacy_full", "acc",
                "L_R_MAIN025", f"{ds}_score.json")),
        ]:
            note = ("pack-external provenance" if not os.path.exists(p)
                    else "pack-external provenance (not in zip)")
            if os.path.exists(p):
                inv.append({"artifact": f"[external] {kind}/{ds}",
                            "bytes": os.path.getsize(p),
                            "sha256": sha256_file(p), "note": note})
            else:
                inv.append({"artifact": f"[external] {kind}/{ds}",
                            "bytes": None, "sha256": "MISSING",
                            "note": note})
    with open(os.path.join(DELIV, "inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["artifact", "bytes", "sha256",
                                          "note"])
        w.writeheader()
        w.writerows(inv)

    # ---- scripts -------------------------------------------------------------
    for fn in ROUND_SCRIPTS:
        shutil.copy(os.path.join(HERE, fn), os.path.join(DELIV, "scripts",
                                                         fn))

    # ---- contact sheet --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n = len(finalists)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, sid in enumerate(finalists):
        ax = axes[i // cols][i % cols]
        p = os.path.join(DELIV, "samples", sid, "preview.png")
        ax.imshow(mpimg.imread(p))
        ax.set_title(sid, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DELIV, "contact_sheet.png"), dpi=110)
    plt.close(fig)

    # ---- README ---------------------------------------------------------------
    scr = {}
    p = os.path.join(DELIV, "screening_round20261007.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            scr[r["sample_id"]] = r
    readme = []
    readme.append("# AnchorZip 图 1 真实可视化样例包（round 2026-10-07）\n")
    readme.append(
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M%z')} · "
        "git 63ad8b9（EADP_amp worktree, codex/anchor-completion-validation）· "
        "GPU: NVIDIA A40, qwen3vl_clean env\n")
    readme.append("## 1. 冻结方法与协议\n")
    readme.append("""
Qwen3-VL-8B-Instruct-1024，N=1024/K=256，官方 `official_legacy` 场景关联
（deepstack=False, pos='1d'）：**RTG → 原 Facility → main-only
Completion（λ=0.25, uniform）**。全部数组来自冻结 CUDA 重放
（`export_anchorzip_figure_sample_v3.py`）：prepare/encode/
instruction_embeds/rtg_importance/facility/compute_assignment，逐位与
`bank_full_rtg_*` 一致（rtg_legacy_checks C1：视觉路径在两种引擎旗标下
逐位相同，故数组可与三条预测链配对）。未删停用词、未缩短 prompt、未翻转
A 符号、未调温度/λ、未改算法。
""")
    readme.append("## 2. 筛查范围与数量\n")
    readme.append("""
- CPU 预筛（Stage A）：6 个数据集全量池（TextVQA_VAL 5000 / OCRBench 1000 /
  DocVQA_VAL 5349 / ChartQA_TEST 2500 / AI2D_TEST 3088 / InfoVQA_VAL 2801）
  按官方 legacy 全量逐题分（==1.0）、组结构（gmax≤32）、图像可读性
  （min_side≥400、fill≥0.55、同图去重）→ 485 行池、197 PASS。
- 第一批真实数组：23 个候选（含指定的 5 个旧候选 TextVQA_VAL_787/1536、
  OCRBench_403/517、DocVQA_VAL_430）。
- 第二批（+10）：bank 摘要 w_diag 的 eff_tokens/L（权重集中度）作为 CPU
  预筛信号，取 ≤0.90 的集中样本（0.62–0.86）。
- 实际取得真实数组：33 个（全部冻结 CUDA 重放；bank 只缓存 keep/gid/gsize
  与 w_diag 摘要，无 w/A，故权重指标必须重放；keep/gid 与 full bank 逐位
  核对一致）。
- 真实数组指标后 PASS 15 个（门槛：checks 全过 + top-12 内容词 ≥3 + 前 6
  有内容词 + bank keep 一致）；最终交付 8 个（覆盖 6 个数据集）。
""")
    readme.append("## 3. 筛选规则（预注册）\n")
    readme.append("""
- 分类规则：固定指令后缀（"Please answer ..."）起的 token = template；
  问题内停用词表/纯标点 = function；其余 = content（词表见
  fig1_round20261007_stageC.py，token 级、随包交付）。
- 排名规则：原始 RTG 权重 w 降序、稳定 argsort（并列取较小 token 下标）。
- 不硬设 w>0.1；w_top1 以 L·w_top1（相对均匀权重倍数）报告。
- highlight 组规则：最大非单点组、并列取最小 slot（预注册、确定性）。
- 门槛（全部须过）：checks all_pass；top-12 内容词 ≥3；前 6 有内容词；
  bank keep 逐位一致。图像可读性/预测正确性已在 Stage A 过滤。
- P(i|t)=softmax_i(100·A_it)；熵按 ln(1024) 归一；top-10% 位置质量 =
  前 103 个最高概率位置的质量和。A/g 是原负 cosine 响应，不称 attention。
""")
    readme.append("## 4. 最终候选简表（全部为真实数组核验通过）\n")
    readme.append("| 样本 | 数据集 | 问题 | Top-3 原始 token (w) | "
                  "L·w_top1 | 中位 H | bank | 判定 |")
    readme.append("|---|---|---|---|---|---|---|---|")
    for sid in finalists:
        r = scr.get(sid)
        if not r:
            readme.append(f"| {sid} | — | — | — | — | — | — | (无 screening "
                          "行，见 finalists.csv) |")
            continue
        t12 = json.loads(r["top12"])
        t3 = " ".join(f"{t['tok']}({t['w']:.3f})"
                      for t in t12[:3])
        readme.append(
            f"| {sid} | {r['dataset']} | {r['question'][:60]} | {t3} | "
            f"{r['L_w_top1_vs_uniform']} | {r['H_norm_median_top12']} | "
            f"{'一致' if json.loads(r['bank_keep_match']).get('full') else '?'}"
            f" | {r['decision']} |")
    readme.append("\n## 5. 首推 2 个\n")
    if top2:
        for sid in top2:
            r = scr.get(sid, {})
            t12 = json.loads(r["top12"]) if r.get("top12") else []
            t3 = " ".join(f"{t['tok']}({t['w']:.3f})" for t in t12[:3])
            readme.append(
                f"- **{sid}**（{r.get('dataset','')}）：Top-3 原始 token "
                f"{t3}；L·w_top1={r.get('L_w_top1_vs_uniform')}；"
                f"中位 H={r.get('H_norm_median_top12')}；"
                f"top-12 内容词 {r.get('top12_content_n')} 个；"
                f"响应两两相关 {r.get('pairwise_corr_mean')}±"
                f"{r.get('pairwise_corr_std')}。推荐理由见 "
                "finalists.csv 与 screening_round20261007.csv。\n")
    else:
        readme.append("(未指定 --top2)\n")
    readme.append("## 6. 每样本目录内容\n")
    readme.append("""
- `source_original.*`：数据集原图（原字节；SHA256 见 sample.json 与
  SHA256SUMS）。
- `sample_aligned.png`：本次实际预处理（smart_resize→方形 pad→1024²）的
  patch 级重建图（与 pixel_values 相关性记录在 checks.json）。
- `sample.npz`（`np.load(..., allow_pickle=False)`）：response[H,W,L]=原始
  A；weights[L]=原始 w（sum=1）；global/local/fused_response[H,W]；
  importance[H,W]；keep[K]（升序）；dropped_idx[N-K]、gid[N-K]（gid 为
  anchor 槽位，对应升序 dropped_idx）；group[H,W]（cell=keep 槽位；
  group[keep[j]]=j）；group_size[K]（含 anchor）；另附 logP[N,L]、
  weights_unnorm_c[L]、grid、n_vis。token i ↔ (h,w)=(i//W, i%W)。
- `anchor_update.npz`：highlight anchor slot/成员（anchor 在首位）、真实
  v_a、含 anchor 的 μ_a、主路 y_a、λ=0.25；另附全部 K 槽的 v/μ/y 及模型
  dtype cast（bf16→fp32 无损保存）；FP32 计算、cast 记录在案。
- `sample.json`：样本键/问题/instruction_text/input_ids/token_labels/
  question span/N/K/grid/模型与 checkpoint/执行路径/arm/λ/git SHA/脚本
  SHA/预处理参数/bank 与预测引用/数组来源/展示索引/渲染命令。
- `checks.json`：逐项核验（形状/有限值/w 非负且 sum≈1/从 A 的 FP32 RTG
  重建 w·local·fused·importance/keep 唯一升序/组不变量/FP64 cosine 重分配/
  μ 含 anchor/y 恒等式/bank keep 逐位一致）。
- `token_ranking.csv`：完整原始排名 + question/suffix 归属。
- `fig1_metrics.json`：Top-12 的熵/A 范围/argmax 位置等（由真实数组计算）。
- `preview.png`：三栏真实数组预览（左图+importance；中 4×3 原始 Top-12、
  共享 A 色标；右全部 K anchor + highlight 组，y_a=v_a+0.25(μ_a−v_a)）。
""")
    readme.append("## 7. 局限与如实记录\n")
    readme.append("""
- Top-12 为完整提示词的原始权重排名；若干样本前 12 含模板词（'conc'
  ='concisely' 等），按真实数值如实展示，未做重归一化或删词。RTG w 对短
  prompt（L≈17–23）天然较散：w_top1≈1.2–1.6×均匀是常态，本次以 bank 摘要
  eff/L 预筛把第二批候选推向 3–6×。
- AI2D_TEST_703 的参考链 corr=0.783（交付 8 样本中最低）：patch 重组按
  构造逐位正确（重组即引擎 pixel_values 本身；逐样本布局由 gthw 确定性
  决定，2026-10-04 已整体验证），低 corr 来自 PIL 参考链对小图的插值差
  （文本亚 patch 级 ghosting），已人工比对确认结构一致。样品照常交付。
- 预测/评分仅作背景记录，不以"答对/答错"为入选条件。
- gid 为 keep 槽位（anchor rank），不是视觉 flat index；组分配按主视觉
  特征 cosine 最近 anchor，非空间距离。
- 渲染脚本：scripts/fig1_round20261007_stageD.py（contact sheet）与
  export_anchorzip_figure_sample_v3.py::render_preview（每样本 preview）。
""")
    with open(os.path.join(DELIV, "README.md"), "w") as f:
        f.write("\n".join(readme))

    # ---- SHA256SUMS (inside tree, excluding itself) ------------------------
    sums = os.path.join(DELIV, "SHA256SUMS")
    with open(sums, "w") as f:
        for root, _, files in os.walk(DELIV):
            for fn in sorted(files):
                if fn == "SHA256SUMS":
                    continue
                p = os.path.join(root, fn)
                f.write(f"{sha256_file(p)}  {os.path.relpath(p, DELIV)}\n")

    # ---- zip ------------------------------------------------------------------
    if os.path.exists(ZIP):
        os.remove(ZIP)
    subprocess.run(["zip", "-q", "-r", ZIP, "."], cwd=DELIV, check=True)
    zip_sha = sha256_file(ZIP)
    zip_bytes = os.path.getsize(ZIP)

    # ---- post-pack verification: unpack and re-check ---------------------------
    tmp = tempfile.mkdtemp(prefix="azfig_verify_")
    subprocess.run(["unzip", "-q", ZIP, "-d", tmp], check=True)
    n_members = sum(len(fs) for _, _, fs in os.walk(tmp))
    errors = []
    for sid in finalists:
        zd = os.path.join(tmp, "samples", sid)
        try:
            z = np.load(os.path.join(zd, "sample.npz"), allow_pickle=False)
            assert z["response"].ndim == 3 and z["keep"].shape[0] == 256
            assert abs(float(z["weights"].sum()) - 1.0) < 1e-5
        except Exception as e:
            errors.append(f"{sid}: {e}")
    # hash consistency inside the unpacked tree
    bad_hashes = 0
    with open(os.path.join(tmp, "SHA256SUMS")) as f:
        for line in f:
            h, rel = line.strip().split("  ", 1)
            p = os.path.join(tmp, rel)
            if not os.path.exists(p) or sha256_file(p) != h:
                bad_hashes += 1
    shutil.rmtree(tmp)
    print(f"ZIP {ZIP}\nBYTES {zip_bytes}\nSHA256 {zip_sha}")
    print(f"MEMBERS {n_members}  BAD_HASHES {bad_hashes}  "
          f"UNPACK_ERRORS {errors if errors else 'none'}")
    print(f"FINALISTS_OK {[v['sample_id'] for v in ver]}")


if __name__ == "__main__":
    main()
