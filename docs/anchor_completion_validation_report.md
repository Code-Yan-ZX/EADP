# Anchor Completion Validation — 报告

**Branch**: `codex/anchor-completion-validation`（base = `cedf05f`）· 2026-10-02
**Protocol**: `docs/anchor_completion_validation_protocol.md`（冻结于任何全量
数字之前，commit `7638b54`）+ 既有 `docs/anchor_merge_full_eval_prereg.md`。
**Model**: Qwen3-VL-8B-Instruct-1024，native E0 engine（DeepStack 三路、
3D mRoPE、SDPA、greedy），K=256，pre-LLM，forward-only，无训练。

> 状态：**进行中**。本文数字占位处一律标 `NOT_RUN`，未跑就是不跑。

---

## 0. 一句话结论

（待主面板与 fresh 面板完成后填写；将明确回答：MAIN025 的新数据收益是否
稳定？收益是否仅能由普通阻尼合并解释？与最近已有方法的差异和实测证据
是什么？是否值得进入多预算/第二模型？）

## 1. 公式与近邻文献审查（先于任何全量得分完成）

### 1.1 已取得的原始来源

| 来源 | URL / 版本 | 取得方式 |
|---|---|---|
| LLaVA-PruMerge, arXiv:2403.15388 (ICCV 2025) | https://arxiv.org/abs/2403.15388 （HTML v6） | 2026-10-02 WebFetch 全文方法节 + Algorithm 1 |
| Libra-Merging, CVPR 2025 (Yang et al., pp.9402-9412) | CVF open-access PDF | 2026-10-02 下载全文（本地 `libra_cvpr2025.pdf`） |
| Libra 官方代码 | https://github.com/longrongyang/Libra-Merging | 2026-10-02 访问：**仅 README，代码未发布**；第三方忠实实现检索亦无 |

### 1.2 逐项公式对照

见协议 §4.2 表（选择器/分配/组均值含 anchor/权重/更新强度/特征流/负例
处理/位置/token 数）。要点：

- **MAIN025 就是固定分组下的阻尼组均值**：y_a = v_a + 0.25·(mean(G_a) − v_a)。
  本轮不将其换名包装为新方法或独立对照。
- PruMerge 的合并 = 类注意力加权的**无归一化和**、不含 anchor、无 λ、
  kNN 分组、IQR 自适应选择 → 与 MAIN025 在权重/归一化/阻尼/分组/选择器
  五个维度不同。
- Libra 的合并 = 匹配矩阵上的 softmax(α/η) 加权 + cosine 阈值 τ=0.7 的
  正负集 + 追加补偿 token → 与 MAIN025 在权重/集合划分/补偿机制/选择器
  （LLM 层 output attention + 空间区间）四个维度不同。

### 1.3 完整方法对照：判定为不可信实现（2026-10-02，先于任何全量得分）

1. **PruMerge**：选择器需要 CLIP-ViT 倒数第二层 [CLS]→patch attention；
   本环境 transformers 4.57.6 `Qwen3VLVisionModel` 源码核实 Qwen3-VL ViT
   **无 CLS token / class embedding**，该信号结构性不存在。按规则不得以
   embedding norm 等替代物冒充。
2. **Libra**：(a) 重要性 α 是 LLM 层 output attention（层级合并于层
   3/15/23），忠实移植需把冻结的 pre-LLM 压缩点改为 in-LLM 并新建
   eager-attention 路径，改变全部计时语义与成本结构；(b) 论文规范缺口：
   α 多头/多 output token 聚合未说明、Eq.(6)(7) 的 W 记号维度矛盾
   （W∈R^{N×N′} 而 V′=W×V 要求 W∈R^{N′×N}）、区间 tie-breaking 未说明；
   (c) 官方代码未发布、无第三方忠实实现；(d) 单流设计未定义 DS 四流适配。
3. **处置**：按预授权规则，"最近完整方法的数值对照"记为**未完成**；
   不自创弱算法凑基线。论文定位如实为"公式级对照成立、数值对照缺失、
   创新证据不完整"。

## 2. 样本暴露审计（exposure_manifest）

来源全部可枚举（9 类，含 m1/sage/s2c2/s2a/bank150/e0 plan/amp manifest/
全部实际生成 shard 兜底），**unknown = 0**。fresh 仅指未用于本项目方案
选择，不声称排除预训练污染。

| 数据集 | rows | exposed | fresh | unknown |
|---|---:|---:|---:|---:|
| TextVQA_VAL | 5000 | 3291 | 1709 | 0 |
| DocVQA_VAL | 5349 | 4244 | 1105 | 0 |
| OCRBench | 1000 | 1000 | **0** | 0 |
| ChartQA_TEST | 2500 | 1546 | 954 | 0 |
| MMBench_DEV_EN_V11 | 4876 | 2815 | 2061 | 0 |
| MMStar | 1500 | 1049 | 451 | 0 |
| RealWorldQA | 765 | 681 | 84 | 0 |
| POPE | 5127 | 2863 | 2264 | 0 |

**OCRBench fresh = 0：无法独立比较**，正式方法在 OCRBench 上不存在独立
确认，全量标签不能掩盖这一重用。fresh 对照面板已冻结（seed=20261002，
整 cluster 纳入，≥200 题）：TextVQA 200 题/128 图、DocVQA 200 题/58 图
（manifest sha256 `f8f69921…`）。

## 3. 正确性门

（G1 已过；G2–G5 于 bank 完成后自动运行。占位待填。）

- **G1 公式**（合成）：uniform λ∈{0.25,1.0}、sim τ=0.1 vs 独立逐组 FP32
  参考实现：rel ≤ 1.07e-7（<1e-5），bf16 cast bitwise 全过；空 dropped
  恒等；N=K 恒等。**PASS**
  （偏离 D-x：参考实现初版把 anchor rank 误当全局索引，修正后全过；
  该修复只改测试侧参考实现，不改生产代码。）
- **G2 路径**：NOT_RUN
- **G3 复现**：NOT_RUN（bank vs 现场重算 bitwise；门贪心输出 vs 评测
  shard 逐字比对）
- **G4 不变量**：NOT_RUN
- **G5 λ=0 恒等**：NOT_RUN

## 4. 主面板（BASE vs MAIN025 全量配对）

（NOT_RUN。预注册判定：pooled cluster bootstrap 5000、seed 20261002；
改善成立 = pooled CI>0 且三任务方向一致。将报告每任务与等权 macro 的
Δ/CI、20000 次嵌套敏感性、rescue/break、full/exposed/fresh 分解、
coverage 门。）

## 5. 非回归面板

（NOT_RUN。ChartQA/MMBench/MMStar/RealWorldQA/POPE 全量，逐 benchmark
Δ 与 CI；POPE headline 用官方 'acc'（'Overall' 为 F1），逐题分为行级
score、官方聚合经 category-explode 精确复算 diff=0。）

## 6. fresh 对照面板（算子消融）

（NOT_RUN。对比：MAIN025−MAIN100（更新幅度）、MAIN025−MAIN_SIM025
（权重形式），每任务与 macro Δ/CI；均为补充比较，CI 跨零 ≠ 相等/非劣。
MAIN025−完整方法：**未运行**（§1.3），醒目标明。）

## 7. 效率

（NOT_RUN。30 题配对、现场全路径、64 token、ignore_eos、交错+预热、
每轮释放 state/KV 后 reset peak。完整方法对照无计时——见 §1.3。）

## 8. 偏离记录

- **D-1（2026-10-02，任何全量得分前）**：G1 参考实现初版 bug（anchor
  rank 当作全局索引），导致首次运行误报不一致；修复后 G1 全过。仅测试
  侧改动。
- **D-2（2026-10-02，任何全量得分前）**：评分预演发现各数据集官方
  results 文件列名/索引语义不同（MMBench/TextVQA/DocVQA 的 `index` 是
  数据集自身题目号而非行号；MMStar/RWQA/MMBench headline 为 0-1 量纲；
  POPE 官方 acc 为 category-explode 均值）。抽取器据此实现并经 8 数据集
  预演验证（diff=0）。仅测试/评分实现侧，不改实验定义。

## 9. 运行台账

（待补：生成总数、GPU 用时、失败/未执行项、运行命令。）
