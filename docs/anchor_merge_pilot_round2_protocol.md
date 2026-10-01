# Anchor-Merge Pilot Round 2 — 流范围 × 选择器 × budget(协议修订)

**Branch**: `codex/anchor-merge-pilot` · 本文件在任何本轮新数字产生之前冻结。
Round 1 协议(`anchor_merge_pilot_protocol.md`)继续有效;本文件只**追加**
三个轴,不改变 round 1 的合并数学、评分、统计与偏离记录规则。

## 冻结的公共设置

- λ = 0.25 固定(均匀聚合,round 1 winner);不搜索新 λ/温度。
- 合并数学、FP32 确定性归约、bank 冻结机制、逐题评分(含 D-4 修正后的
  OCRBench 规则)、cluster bootstrap(5000 次,seed 20261001)全部沿用。
- 分配与权重**永远来自主特征空间**(round 1 §3),无论聚合哪几路流。
- 数据:DEV = round 1 的 DEV-100 × 3(同一 manifest);CONFIRM = round 1 的
  CONFIRM-200 × 3(同一 manifest,含 OCRBench 36 行部分复用)。不新增数据。
- 评分路径差异声明:block8 与 K=128 使用各自的冻结 bank
  (`bank_{split}_{sel}_K{K}_{ds}.json.gz`),arm 只读 bank。

## 轴 1:流范围(K=256,official facility,λ=0.25)

| arm | 定义 |
|---|---|
| MAIN025 | 只聚合主特征 V;三路 DS 走 anchor gather(原值) |
| DS025 | 只聚合三路 DS;主特征走 anchor gather(原值) |
| BOTH | = round 1 的 U025(不重跑,直接引用其 DEV/CONFIRM 数字) |

**判定规则(冻结)**:DEV 上 MAIN025 / DS025 与 BOTH(U025)比较;若最优
新候选的 DEV macro 严格高于 BOTH → 在 CONFIRM 上跑该候选(BASE 已有)做
确认;否则该轴记"无继续信号",不消耗 CONFIRM。

## 轴 2:选择器(block8 = b2,K=256,λ=0.25)

| arm | 定义 |
|---|---|
| B2BASE | block8 选择,恒等 gather(即 E0 的 b2,本轮原生重跑) |
| B2U025 | block8 anchors + 均匀聚合 λ=0.25(主+DS) |

**判定规则(冻结)**:DEV 上 B2U025 > B2BASE → CONFIRM 两 arm(600)确认;
否则不消耗 CONFIRM。附注:同时记录 B2U025 相对 facility BASE 的位置(用户
关心的是"block8 + 合并"能否追平/超过 facility 基线),但该比较跨选择器,
只作参考不作判定。

## 轴 3:budget(K=128,selector × budget 完整 2×2,λ=0.25)

| arm | 定义 |
|---|---|
| K128BASE | official facility @128,恒等 gather |
| K128U025 | facility @128 + 均匀聚合 |
| B2K128BASE | block8 @128,恒等 gather |
| B2K128U025 | block8 @128 + 均匀聚合 |

选 K=128(而非 512)的理由,冻结于此:组均成员 7 vs 1.5,合并机制在更紧
budget 下应有更大空间;且 E0 网格已含 128,便于日后横向阅读。

**判定规则(冻结)**:每个 selector 内部比较 U025 vs BASE;DEV 上为正 →
该 (selector, K) 组合进 CONFIRM 确认;为负 → 不消耗 CONFIRM。

## 执行顺序与总量

1. 正确性(round 2 追加 gates):λ=0 × 三个 scope 与 gather 全等;三个新
   bank(b2@256, b1@128, b2@128)与现场重算一致;K128/B2 的不变量。
2. DEV:8 个新 arm × 300 = 2400 次生成。
3. 冻结判定 → CONFIRM:最多 6 个新 arm × 600(B2 两 arm、K128 两 selector
   各两 arm、流范围最多 1 arm)= ≤3600 次。
4. 效率不重测(合并成本与 budget/选择器无关的部分已在 round 1 测定;
   K=128 的 TTFT 差异属于 budget 本身,不并入本轮结论)。

## Round 1 结果引用(本轮比较基线,来自 analysis_dev/confirm.json)

DEV: BASE 81.157,U025(BOTH) 82.652。CONFIRM: BASE 77.442,U025 78.838,
NORM 77.697,SHUF 76.790。

## 偏离记录

(运行中追加;当前为空。)
