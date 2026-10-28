#!/usr/bin/env python3
"""Assemble the v2 (new-candidates-only) AnchorZip Figure-1 delivery zip.

Reads outputs/figures/anchorzip/samples_v2/{sample}/  (exporter v2 output
plus fig1_metrics.json) and produces, under outputs/figures/anchorzip/:

    anchorzip_fig1_new_candidates_<date>.zip
        samples_v2/<sample>/   source_original.*, sample_aligned.png,
                               sample.npz, sample.json, checks.json,
                               fig1_metrics.json
        update_report_v2.md    protocol audit + screening correction +
                               per-sample verification table (data-driven)
        screening_v2.csv
        SHA256SUMS

Prints the zip's byte count and sha256.  Run AFTER
export_anchorzip_figure_sample_v2.py and fig1_metrics.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

FIG = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/figures/anchorzip"
SV2 = os.path.join(FIG, "samples_v2")
DELIV1 = os.path.join(FIG, "delivery_20261004")
DATE = "20261005"
ZIP = os.path.join(FIG, f"anchorzip_fig1_new_candidates_{DATE}.zip")

SUFFIX = "Please answer concisely"
FUNCTION_WORDS = {
    "what", "which", "where", "who", "whom", "whose", "when", "is", "are",
    "was", "were", "the", "a", "an", "of", "on", "in", "at", "to", "for",
    "with", "by", "from", "and", "or", "this", "that", "these", "those",
    "it", "its", "as", "be", "been", "was", "does", "do", "did", "can",
    "please", "answer", "concisely", "short", "words", "phrases", "when",
    "possible", ".", "?", "?\n", ",", "'", "'s", "given",
}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def content_words(top12, labels, q_end):
    out = []
    for r in top12:
        t = int(r["token_index"])
        tok = labels[t].strip().lower().strip(".,?!\n")
        if t < q_end and tok and tok not in FUNCTION_WORDS:
            out.append(r)
    return out


def median_h(top12):
    hs = sorted(r["spatial_entropy_normalized"] for r in top12)
    return hs[len(hs) // 2]


def main() -> None:
    run = json.load(open(os.path.join(SV2, "run_summary.json")))
    rows, ok_samples = [], []
    for sid, sr in run["samples"].items():
        if not sr.get("all_pass"):
            rows.append((sid, None))
            continue
        d = os.path.join(SV2, sid)
        meta = json.load(open(os.path.join(d, "sample.json")))
        checks = json.load(open(os.path.join(d, "checks.json")))
        try:
            fig = json.load(open(os.path.join(d, "fig1_metrics.json")))
        except FileNotFoundError:
            rows.append((sid, None))
            continue
        labels = meta["token_labels"]
        q_end = meta["question_span"]["end_exclusive"]
        cw = content_words(fig["top12"], labels, q_end)
        mh = median_h(fig["top12"])
        preds = meta["predictions"]
        p_leg = preds["L_R_MAIN025_official_legacy"].get("prediction")
        p_dev = preds["R_MAIN025_dev100_native"].get("prediction")
        rows.append((sid, dict(
            question=meta["question"], N=meta["N"], K=meta["K"],
            L=meta["L"], all_pass=checks["all_pass"],
            bank_keep=checks["bank_keep_match"],
            pred_legacy=p_leg, pred_dev100=p_dev,
            pred_full=preds["R_MAIN025_full_native"].get("prediction"),
            pred_consistent=(p_leg == p_dev == p_full) if False else None,
            score_legacy=meta["scores_per_question"].get(
                "L_R_MAIN025_official_legacy"),
            arrays_complete=checks["all_pass"] and all(
                os.path.exists(os.path.join(d, f)) for f in (
                    "sample.npz", "sample.json", "checks.json",
                    "fig1_metrics.json")),
            top12_content=len(cw),
            top12_content_tokens=[r["token"].strip() for r in cw],
            median_H=round(mh, 4), max_w=fig["top12"][0]["weight"],
            weights_sum=fig["weights_sum"])))
        ok_samples.append(sid)

    # ---- report ------------------------------------------------------------
    L = []
    L.append("# AnchorZip 图 1 新候选真实数组补交报告（v2）\n")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M%z')} · "
             f"git {run['git_sha']} · 计算设备：{run['device']}（CUDA）\n")
    L.append("## 1. 协议逐文件审计（回应 2026-10-04 报告的 legacy 标注错误）\n")
    L.append("""
| 链 | 协议 | 位置 |
|---|---|---|
| Stage-1 RTG dev-100 bank + 旧 3 样本数组 | NativeEngine（默认旗标 deepstack=True, pos=mrope3d），prefill 前视觉路径 | `stage1_roundtrip_pilot/bank_dev_rtg_*.gz`、`outputs/figures/anchorzip/samples/` |
| dev-100 各臂预测（含 acc/R_MAIN025） | NativeEngine 默认旗标（**非 legacy**，此前报告误标，特此更正） | `stage1_roundtrip_pilot/acc/{arm}/` |
| full 面板 R_MAIN025 预测 | NativeEngine 默认旗标，全量 split | `stage1_roundtrip_pilot/full/acc/R_MAIN025/` |
| **官方 legacy 场景 L_R_MAIN025** | deepstack=False, pos='1d'（= 论文场景；e0 N5/s1_dualpath 已核旗标位等价） | `stage1_roundtrip_pilot/legacy_full/acc/L_R_MAIN025/` |

图 1 数组全部产生于 prepare/encode/instruction_embeds/select_keep——该路径
不调用 prefill/decode，rtg_legacy_checks C1 已逐位验证 bank 记录在两种旗标
下一致。因此同一套数组与三条链的预测都可配对；本报告以官方 legacy 场景
（L_R_MAIN025）预测为主配对，native 链仅作旁证。
""")
    L.append("## 2. screening.csv 两列恒定值的核实与撤回\n")
    L.append("""
- `keep_spatial_entropy`（300 行全部 0.8）：真实计算值，但 facility 把 256 个
  anchor 近均匀铺满 32×32 网格，熵 ≈ ln(256)/ln(1024)=0.80 是结构恒等式，
  无分辨力 → 作为门槛**撤回**。
- `group_mean`（300 行全部 3.0）：= (N−K)/K = 768/256 = 3.0，算术恒等式 →
  作为门槛**撤回**。
- 用原始评分文件重算的 `screening_v2.csv`（不经 CSV 舍入）只保留 4 道真实
  门槛（low_res / padding_heavy / wrong_prediction / mega_blob）；
  **PASS 集与 v1 完全一致（139/300，0 行差异）**——撤回不改变任何决策。
""")
    L.append("## 3. CPU 重放尝试（负结果，记录在案）\n")
    L.append("""
数组只依赖 prefill 前视觉路径，曾在 CPU（bf16, sdpa）上复算 TextVQA_787 以
避开 GPU 队列：encode 22s 可行，但 facility 的近平局使 keep 仅
**194/256** 与 CUDA bank 一致（62 个翻转）——CPU bf16 与 CUDA bf16 累加顺序
差异超出平局容差。按"同协议逐位对齐"标准，CPU 数组不可交付，已弃置；
以下全部数组为 CUDA 权威重放。
""")
    L.append("## 4. 新候选逐样本核验简表\n")
    hdr = ("| 样本 | 协议 | legacy/dev100 预测一致 | bank keep 一致 (dev/full) | "
           "数组齐全 | Top-12 内容词 | 中位 H | 判定 |")
    L.append(hdr)
    L.append("|---|---|---|---|---|---|---|---|")
    for sid, r in rows:
        if r is None:
            L.append(f"| {sid} | — | — | — | 否（run 失败） | — | — | 淘汰 |")
            continue
        pcons = (r["pred_legacy"] == r["pred_dev100"] == r["pred_full"])
        bkm = r["bank_keep"]
        fok = all(v is True for v in bkm.values()) if bkm else False
        if mh := r["median_H"]:
            verdict = ("不支持强聚焦叙事（中位 H≥0.80，响应空间分散）"
                       if mh >= 0.80 else
                       "存在空间集中信号（中位 H<0.75）"
                       if mh < 0.75 else
                       "中等（0.75≤H<0.80）")
        L.append(f"| {sid} | L_R_MAIN025(legacy)+native 旁证 | "
                 f"{'是' if pcons else '否'} | "
                 f"{'是/是' if fok else str(bkm)} | "
                 f"{'是' if r['arrays_complete'] else '否'} | "
                 f"{r['top12_content']} ({','.join(r['top12_content_tokens'][:6])}) | "
                 f"{r['median_H']:.3f} | {verdict} |")
    L.append("""
Top-12 权重为冻结 text_weight 在**完整提示词**上的原始输出（未删停用词、
未重归一化，不得称"注意力占比"）；熵为 P(i|t)=softmax_i(100·A_it) 的空间熵
（归一化 ln(1024)）。若某样本 Top-12 仍以指令后缀模板词为主且中位 H≥0.80，
按预注册规则如实记为"数值不支持图 1 聚焦叙事"——是否启用该样本由桌面端
依据真实数组决定。
""")
    L.append("## 5. 交付内容\n\n本 zip 只含新候选真实数组与更新报告；"
             "旧 3 样本未重复打包（见 delivery_20261004）。\n")
    rep = os.path.join(FIG, "update_report_v2.md")
    with open(rep, "w") as f:
        f.write("\n".join(L))

    # ---- zip ----------------------------------------------------------------
    if os.path.exists(ZIP):
        os.remove(ZIP)
    stage = os.path.join(FIG, f"_stage_v2_{DATE}")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(os.path.join(stage, "samples_v2"), exist_ok=True)
    for sid in ok_samples:
        shutil.copytree(os.path.join(SV2, sid),
                        os.path.join(stage, "samples_v2", sid))
    shutil.copy(rep, stage)
    shutil.copy(os.path.join(DELIV1, "screening_v2.csv"), stage)
    subprocess.run(["zip", "-q", "-r", ZIP, "."], cwd=stage, check=True)
    sums = os.path.join(stage, "SHA256SUMS")
    with open(sums, "w") as f:
        for root, _, files in os.walk(stage):
            for fn in sorted(files):
                if fn == "SHA256SUMS":
                    continue
                p = os.path.join(root, fn)
                f.write(f"{sha256_file(p)}  {os.path.relpath(p, stage)}\n")
        f.write(f"{sha256_file(ZIP)}  ../{os.path.basename(ZIP)}\n")
    subprocess.run(["zip", "-q", ZIP, "SHA256SUMS"], cwd=stage, check=True)
    shutil.rmtree(stage)
    n = os.path.getsize(ZIP)
    print(f"ZIP {ZIP}\nBYTES {n}\nSHA256 {sha256_file(ZIP)}")
    print(f"SAMPLES_OK {ok_samples}")


if __name__ == "__main__":
    main()
