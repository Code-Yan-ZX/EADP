# 给服务器 Claude Code 的 E0 执行提示词

将下面分隔线之间的内容整体粘贴给服务器上的 Claude Code。

---

你在 EADP 视觉 token 剪枝项目的服务器工作区工作。任务：执行预注册实验 **E0**，完成 native Qwen3-VL 修复、基线复现和编码器轴余量审计。

## 开始前必读（按顺序完整读完）

1. `docs/e0_native_baselines_prereg.md`：本任务唯一的规范。其中的门、arm、测量和决策规则都已冻结。
2. `docs/research_reset_20260929.md` 和 `docs/scoring_search_m2_system_amendment.md` 的 §1.4 与 §7：说明为什么需要修复，以及 paired-block 计时协议。
3. `AGENTS.md` 和 `docs/project_handoff.md`：背景。handoff 中 M1 时期的建议已经过时，以更新的证据为准。
4. 先执行 `git fetch origin`，然后 `git checkout e0-native-baselines`（本地没有就从 `origin/e0-native-baselines` 建跟踪分支），再看 `git status` 和 `git log -5`，确认以上文档都在。如果不在，停下来报告，不要自己重写规范。工作区有未提交改动时，先报告，不要 stash 或丢弃。

## 工作纪律

- 所有工作都在已有分支 `e0-native-baselines` 上进行，不要直接提交到 main。每个里程碑完成后提交一次，提交时只 add 具体文件，不用 `git add .`。
- **不得修改冻结规范**。任何偏离，例如某个基线无法移植、数据集不可用或门需要调整，都要在看到受影响的数字**之前**写入 `docs/e0_native_baselines_prereg_amendment.md`，写明原因和替代方案。
- 不要调任何方法的超参。CONFIRM 集只生成索引和 sha256，不读答案，也不跑生成。
- 不得把任何数据或模型输出发往外部 API。需要 LLM judge 的数据集，一律用 VLMEvalKit 的规则抽取路径。
- 环境里已有 `HF_HUB_OFFLINE=1` 等约定（见 `Qwen_vl/scripts/discovery/common.py`），沿用。PACE 使用独立 conda 环境，不要污染主环境。
- 如实报告：门没通过就是没通过；某个 arm 没跑就写没跑。不要用 legacy 数字填空，也不要引用论文数字和我们的 Qwen3 数字直接比较。
- 如果同一个修复方法连续失败两次，停下来分析根因并写进报告，再换思路，不要反复微调。

## 里程碑（按顺序；每一步有停止点）

### M0 环境与计划
- 记录 torch、transformers、CUDA、驱动和 GPU 型号。
- 实现 `Qwen_vl/scripts/e0/e0_plan.py`，按 §2 生成 `Qwen_vl/outputs/e0/e0_plan.json`：按图像剔除、DEV/CONFIRM 划分、每个数据集的计数和 sha256。
- 提交代码，并把计数表写进一个进度笔记。

### M1 Native 修复与正确性门
- 实现 `Qwen_vl/model/native_qwen3.py`，严格按 §3.2：
  - 统一选择器接口，只输出 keep_idx；
  - 所有特征流（主特征 + 三层 DeepStack）用同一索引收缩；
  - 3D 位置从完整序列的 `get_rope_index` 取子集；
  - prefill 显式传 `position_ids`；decode 使用 `rope_deltas`；
  - 为 LLM 内剪枝提供同步收缩接口。
- 用 stock `Qwen3VLModel.forward` 完成注入，不要复制 decoder 实现。先读服务器上已安装的 transformers 版本中的 `modeling_qwen3_vl.py`，确认 `get_rope_index`、`_deepstack_process` 和 decode 分支的行为与 prereg 描述一致。不一致时以源码为准，并写进 amendment。
- 实现 `Qwen_vl/scripts/e0/e0_gates.py`，运行 N1–N5，输出 `e0_gates.json`。
- **停止点**：任一门失败，修复后重跑，不得产出任何准确率数字。所有门通过后提交代码，并在进度笔记中贴出门的结果表。

### M2 基线移植
- 已有的 DivPrune、CDPruner、HiPrune、EADP（B1）、B2 只需改成输出 keep_idx，接入 native 路径。
- FastV、PDrop、SparseVLM、VisionZip：克隆官方仓库，阅读实现后移植到 `Qwen_vl/model/baselines/`，保持官方默认超参。在代码注释里注明对应的官方文件和提交哈希。
- 每个 arm 在 12 个样本上通过 N4 不变量检查，并做一次 30 题 smoke test，检查截断率和是否出现明显异常。
- PACE 按 §4.1 处理：先在 Qwen2.5-VL-7B 上复现官方分数，再移植到 Qwen3-VL。
- CIVIC 按 §4.2 处理：检索代码，记录查询和结果。有代码时，只写训练预算估计，**不开始训练**。
- 提交。

### M3 Pilot 与预算重估
- 每个 arm 在 K=256 下跑 30 题 DEV pilot，测出每题耗时。
- 按实测速度重估全量 GPU 小时数，写进进度笔记。
- **停止点**：如果重估结果超过 prereg §8 上限的 1.5 倍，或者 CIVIC 需要训练，先停下，把估计交给人决定。否则继续。

### M4 全量准确率
- 实现 `e0_accuracy.py`，完成 §5.1 规定的全部 arm × K × 数据集，外加 A1、A2。支持断点续跑，可按数据集并行到多卡。
- 输出 `e0_acc_*.json`，保存逐题预测、逐题得分和截断标志。

### M5 效率与分辨率扫描
- 实现 `e0_perf_paired.py`，按 §5.2 的 paired-block 协议执行，可以复用 `Qwen_vl/scripts/discovery/m2_perf_paired.py` 的结构。计时期间 GPU 上不能有其他作业。
- 实现 `e0_res_sweep.py`，按 §5.3 执行。

### M6 分析与报告
- 实现 `e0_analyze.py`：统计 paired cluster bootstrap、rescue/break，绘制 Pareto 图，并**机械地**按 §6 判定 D1–D4，结果写入 `e0_verdict.json`。
- 撰写 `docs/e0_native_baselines.md`，内容包括：
  - 门结果；
  - K=256 主表，外加 K=64 和 128 附表；
  - Pareto 图；
  - 修复效应（A1/A2 与 native B2 的比较）；
  - `ViTshare`；
  - D1–D4 判定；
  - 移植存疑的清单；
  - 偏差与修正记录；
  - 计算用量。
- 提交并推送分支 `e0-native-baselines`，**不要创建 PR，也不要合并**。

## 完成时交付给人的摘要（不超过一屏）

1. 门是否全部通过；
2. native B2 与 legacy B2 的差值及 CI；
3. K=256 时每个 arm 的 OCR macro、通用 macro 和 TTFT 中位数（一张表）；
4. `ViTshare`，以及 PACE 相对 B2 的 TTFT 和 OCR 差值；
5. D1–D4 的判定结果；
6. 所有偏差、没跑的项目和移植存疑的项目。

---
