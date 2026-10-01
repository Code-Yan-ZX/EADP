# Anchor-Merge Pilot Round 2 — 流范围 × 选择器 × budget 报告

**Branch**: `codex/anchor-merge-pilot` · 2026-10-01
**Protocol**: `docs/anchor_merge_pilot_round2_protocol.md`(冻结于任何本轮数字之前);
合并数学、评分、统计与 round 1 完全一致。Round 1 报告见
`docs/anchor_merge_pilot_report.md`。

## 0. 一句话结论

**λ=0.25 的聚合收益来自主特征流**;DS 流单独聚合无独立收益、是否聚合 DS
不可分辨。**block8 选择器被确认显著差于 official facility(−3.08,CI 排除
零),合并不足以弥补选择器差距**。一个新信号:**在 block8@128 这个破坏最大
的组合上,合并效应首次被确认(+1.80,CI [+0.19, +3.38])**——选择器越差、
budget 越紧,聚合修复的份额越大。

---

## 1. 正确性(round-2 追加 gates,`correctness_round2.json`)

- G-A:λ=0 下 main/ds 两个 scope 与纯 gather 的 prefill logits **bitwise
  相等**、预测一致(3 数据集 × 2 scope),特征级逐元素相等。
- G-B:三个新 bank(b2@256、b1@128、b2@128)与现场重算 S+gid 全等(9/9)。
- G-C:B2BASE/B2U025 n_vis_kept=256、K128 arms n_vis_kept=128、
  ds_lengths/layer_calls/cache 不变量全过(4/4)。
- 评分自检:本轮全部 24(DEV)+ 15(CONFIRM)个评分单元逐题均值复现官方
  headline。

## 2. 轴 1:流范围(K=256,official facility,λ=0.25)

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs BASE,95% CI |
|---|---:|---:|---:|---:|---|
| BASE(facility gather) | 83.30 | 76.53 | 72.50 | 77.442 | — |
| MAIN025(仅主特征) | 83.60 | 76.14 | 76.00 | 78.578 | +1.136 [+0.005, +2.344] |
| BOTH=U025(主+DS) | 83.60 | 76.91 | 76.00 | 78.838 | +1.396 [+0.324, +2.569] |
| DS025(仅 DS;仅 DEV) | 80.00 | 76.97 | 88.00 | 81.657 | +0.500 [−0.820, +1.925](DEV) |

- MAIN025 在 CONFIRM 上同样为正(+1.14,CI 勉强不跨零)。
- **U025 − MAIN025 = +0.259,CI [−0.109, +0.720]** → 跨零,不可分辨。
  DEV 上方向相反(MAIN025 − U025 = +0.615,CI [−0.623, +1.861],也跨零)。
- **结论:聚合收益来自主特征;是否同时聚合 DS 流不可分辨(不确定)**。
  DS025(只聚 DS 不聚主路)在 DEV 上明显弱于两者 → 没有 DS 单独的增益。
  工程含义:若要最省,MAIN-ONLY 的 pooling 量是 BOTH 的 1/4(主路一路),
  收益统计上无损失;但两者都未证明优于对方,本轮维持 BOTH 为参考实现。

## 3. 轴 2:选择器(block8 vs official facility,K=256)

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs BASE(facility),95% CI |
|---|---:|---:|---:|---:|---|
| B2BASE(block8 gather) | 81.75 | 70.34 | 71.00 | 74.364 | **−3.078 [−5.162, −0.976]** |
| B2U025(block8+merge) | 81.95 | 71.24 | 72.00 | 75.063 | −2.379 [−4.509, −0.231] |

- **block8 被确认显著差于 facility**(CI 排除零;主要掉在 DocVQA −6.2)。
  与项目历史结论一致(block8 在旧 panel 上 −2.0)。
- 合并对 block8 的改善:B2U025 − B2BASE = +0.699,CI [−0.503, +1.910]
  → **跨零,未被确认**(DEV 上 +1.548 [CI +0.160, +3.170] 为正,CONFIRM
  未复现)。
- 选择器差距无法用合并弥补:U025(facility+merge)− B2U025 = **+3.775
  [CI +1.701, +5.885]**。

## 4. 轴 3:budget K=128

| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs BASE(facility@256),95% CI |
|---|---:|---:|---:|---:|---|
| K128BASE(facility@128) | 76.60 | 60.37 | 82.00 | 72.992 | −8.166 [−12.089, −4.125](DEV) |
| K128U025(facility@128+merge) | 73.90 | 61.43 | 83.00 | 72.777 | −8.381 [−12.399, −4.180](DEV) |
| B2K128BASE(block8@128) | 71.30 | 45.59 | 54.50 | 57.129 | **−20.313 [−23.758, −16.993]** |
| B2K128U025(block8@128+merge) | 72.20 | 48.57 | 56.00 | 58.925 | −18.517 [−21.616, −15.454] |

- b1@128:合并 −0.215(CI [−1.792, +1.360],DEV)→ 按冻结规则未触发
  CONFIRM;facility@128 的聚合收益不可分辨。
- b2@128:**B2K128U025 − B2K128BASE = +1.795,CI [+0.187, +3.383]** →
  **本轮唯一 CI 排除零的轴内合并效应**,发生在 baseline 最差的组合上
  (rescue 21 / break 8)。
- @128 相对 @256 的大幅掉点(K128BASE −8.2;DocVQA −18.4)说明 1024×1024
  文档场景在 128 token 下信息损失严重,聚合只能修复其中一小部分。

## 5. 综合解读

三个 selector×budget 组合的合并效应(CONFIRM 或 DEV 内的轴内配对):

| 组合 | baseline 水平 | merge Δ | 95% CI | 判定 |
|---|---:|---:|---|---|
| facility@256 | 77.44 | +1.396 | [+0.324, +2.569] | 确认(round 1) |
| block8@256 | 74.36 | +0.699 | [−0.503, +1.910] | 不确定(CONFIRM) |
| block8@128 | 57.13 | +1.795 | [+0.187, +3.383] | 确认 |
| facility@128 | 72.99(DEV) | −0.215 | [−1.792, +1.360] | 不确定(仅 DEV) |

一致的图景:**聚合是对"选择+删减造成的信息损失"的部分修复**——基线被
破坏得越重,merge 的可测增益越大;基线已经很强时(facility@256),增益
小而稳定(+1.4),在 block8@256 上不足以达到可分辨。这支持 round 1 的
机制结论(内容修正有效、非范数效应),并给出边界:merge 不是选择器的
替代品,而是其质量短板的缓冲。

## 6. 边界

- 流范围结论(主路是收益来源)建立在两个跨零 CI 上,是"方向性证据",
  不是分辨性结论;300/600 题分辨不了 <0.5 点的流间差。
- b1@128 的 CONFIRM 未运行(冻结规则:DEV 点估计为负不触发);若要确认
  facility@128 的合并效应,需追加运行。
- block8 系的结论绑定本实现的 block8(E0 b2 原样);不外推到其他
  近似选择器。

## 7. 运行台账

- 新增生成:DEV 8 arms × 300 = 2400;CONFIRM 5 arms × 600 = 3000;合计
  5400 次贪心生成(另 round 1 累计 ~4100)。
- 总 GPU 时间(本轮,含 bank 构建 6 组 × 3 数据集、gates、DEV、CONFIRM):
  约 4–4.5 小时,单卡 A40,无争用。
- 失败/未执行:b1@128 的 CONFIRM(规则未触发,见 §6);无其他失败。
- 偏离:无新实验定义偏离(仅有一次监控脚本自身 pgrep 自匹配导致的等待
  循环,与实验无关)。

## 8. 产物

- 代码:`amp2_correctness.py` 新增;`amp_common`(scope 支持)、
  `amp_bank`/`amp_accuracy`(selector×K 参数化)、`amp_analyze`/`amp_report`
  (K 无关加载)更新。
- 数据:`bank_{dev,confirm}_{b1,b2}_K{128,256}_*.json.gz`(6 组新 bank)、
  `acc/{dev,confirm}/<arm>/K<K>/`、`correctness_round2.json`、
  `analysis_{dev,confirm}.json`(已含全部 arms)。
- 本报告:`docs/anchor_merge_pilot_round2_report.md`。
