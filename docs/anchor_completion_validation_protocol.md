# Anchor Completion Validation — 本轮补充协议

**冻结日期**: 2026-10-02（写于本轮任何全量评测数字存在之前；工作分支
`codex/anchor-completion-validation`，base = `cedf05f`，现场核对时工作区干净、
与 origin 一致、无运行中的评测作业）。
**性质**: 本文件**补充**而非改写既有冻结文档。主判定、统计量、预案
（BOTH025）与评测范围沿用 `docs/anchor_merge_full_eval_prereg.md`（2026-10-02
冻结）；合并数学、确定性归约、bank 冻结、评分规则（D-4 修正后 OCRBench、
DocVQA ANLS、S0 硬失败）沿用 round 1/2 与 cross_stream 协议既有定义。

---

## 1. 现场核对结论（2026-10-02）

- 分支 `codex/stage1-cross-stream-pilot` @ `cedf05f` 干净且与远端一致；
  已知纠错 `60a40c3`、外部复核入库 `cedf05f` 均为最新，无更新遗漏。
- 未发现任何已启动或已完成的本轮全量评测（`Qwen_vl/outputs/` 下无
  anchor_completion_validation 产物；`discovery/s2b_full*.json` 为旧 S2-B
  梯度 teacher 产物，与本轮无关）。**本轮全部产物新建，无历史结果可复用。**
- Python: `/home/dell/miniconda3/envs/qwen3vl_clean/bin/python`
  （3.10.20 / torch 2.10.0+cu128 / transformers 4.57.6），单卡 A40 46GB。
- Stage-1 三个方向（grounding `d50da38`、visual-calibration `ad08187`、
  cross-stream `1a9fb44`/`60a40c3`）均 KILL；DCC 维持
  **official EADP Stage 1 + official Facility + MAIN025**。禁止重试 m、floor、
  三流权重、grounding、novelty scorer 或归档求和版结果。

## 2. 冻结的正式方法与执行路径（不变）

native E0 engine（`model/native_qwen3.py`），Qwen3-VL-8B-Instruct-1024
（fixed-res wrapper，1024×1024，逐图记录实际 N），DeepStack 三路注入、
3D mRoPE、SDPA、greedy（`do_sample=False` 等价）、max_new_tokens=2048。
K=256，pre-LLM，forward-only，不训练。

- **BASE** = EADP importance（alpha=0.5, beta=2.0, entropy T=100,
  keep_ratio=0.2, M_temp=0.01, smoothing k=3 σ=1.0）+ official facility
  纯 gather（selector `b1`）。
- **MAIN025** = 同 Stage 1、同 Facility、同 anchors（同一冻结 bank）；
  仅主特征流：dropped token 按原始主路特征 FP32 cosine argmax 分配到
  最相似 anchor（并列取更小 anchor 索引）；G_a 含 anchor 自身；
  y_a = v_a + 0.25·(mean(G_a) − v_a)。DS 三路保持 DS[s][keep] 原值。
- 多图不跨图分组；N≤K 整图保留。

## 3. 两个算子消融（预冻结；均为 arm 配置，不改合并代码）

与 BASE/MAIN025 **同一冻结 bank（同 keep/gid）**，只修改主路：

| arm | 定义 | 角色 |
|---|---|---|
| MAIN100 | 同分组，y_a = mean(G_a)（λ=1.0，均匀权重） | 更新幅度消融 |
| MAIN_SIM025 | 同分组，成员权重 softmax(cos(v_i,v_a)/0.1)，anchor logit=1/0.1，加权均值后再 y_a = v_a + 0.25·(m_a − v_a) | 权重形式消融（复用 round-1 S025 算子，τ=0.1 固定，不调参） |

MAIN025 在给定分组下**就是阻尼均值聚合**——本协议承认这一点，不换名。
MAIN025 与 MAIN100 的差异只能支持"更新幅度"的作用，不能支持权重形式的
作用；MAIN025 与 MAIN_SIM025 的差异同理。两项均为算子消融，**不是**
PruMerge/Libra 的官方实现，不得如此声称。

（条件预案臂 BOTH025 = scope=both λ=0.25，仅当 prereg §4 机械条件触发时
运行一次，标为条件触发的次要分析。）

## 4. 公式审查与最近完整方法对照的判定（先于任何全量得分）

### 4.1 已核对的原始来源

| 来源 | URL / 版本 | 取得内容 |
|---|---|---|
| LLaVA-PruMerge (ICCV 2025) | https://arxiv.org/abs/2403.15388 （HTML v6, 2026-10-02 取得） | 全文方法节 + Algorithm 1 |
| Libra-Merging (CVPR 2025) | CVF open-access PDF `Yang_Libra-Merging_…_CVPR_2025_paper.pdf`（2026-10-02 下载，sha256 见报告） | 全文 §3.2–3.4、Eq.(5)–(7)、Algorithm 与全部实验设置 |
| Libra 官方代码 | https://github.com/longrongyang/Libra-Merging （2026-10-02 访问） | **仅 README，代码未发布**（"Release code" 待办） |

### 4.2 逐项对照（选择器 / 分配 / 组均值 / 权重 / 强度 / 流 / 位置 / token 数）

| 维度 | MAIN025（本轮） | LLaVA-PruMerge | Libra-Merging |
|---|---|---|---|
| 选择器 | EADP importance + official facility @K=256 | CLIP-ViT 倒数第二层 [CLS]→patch attention + IQR 上栅栏（自适应 ~32/576；PruMerge+ 再加空间均匀采样） | LLM 层 output attention α，视觉序列等分 N·R 个区间、每区间取 α 最高 1 个（层级合并于 LLM 层 {3,15,23}） |
| 被删 token 分配 | 主特征 cosine argmax → 唯一 anchor | 每 token 的 k 近邻 cluster center（Key 点积） | 匹配矩阵 M_ij=𝕀(S_ij=max_k S_ik)，最相似 target |
| 组均值含 anchor | 是（G_a 含 anchor） | 否（center 更新 y′_p=Σ_q a[j_q]·y_{j_q}，未显式加回 y_p，未归一化） | 权重行 softmax 归一 |
| 权重 | 均匀（或 sim softmax τ=0.1） | 类注意力 a[j_q]，无归一化 | W_i = softmax(M′_ij/η)，M′=α_j·M_ij，η=mean(α) |
| 更新强度 | λ=0.25 阻尼 | 无 λ（全量替换） | 无 λ（全量替换） |
| 特征流 | 仅主路（DS 原值 gather） | 单流（LLaVA） | 单流 |
| 负例处理 | 全部并入组 | 全部并入 | cosine s^max≤τ=0.7 者不并入，聚合为追加的 information compensation token |
| 位置 | anchor 原生 3D mRoPE | （LLaVA 1D） | （LLaVA 1D） |
| token 数 | K=256 固定 | 自适应 m（~32 或 2m） | N·R 固定 |

**结论**：MAIN025 = 固定分组下的阻尼组均值，与 PruMerge（注意力加权和、
无归一化、无 anchor、无 λ）和 Libra（softmax 重要性加权 + 正负集 + 补偿
token）公式均不同；同时 MAIN025 的机制新颖性主张必须限定为"更新幅度/权重
形式的算子消融所能支持的范围"，不预设存在足够创新。

### 4.3 完整方法对照的可实现性判定（2026-10-02，任何全量准确率存在之前作出）

1. **PruMerge：不可忠实实现。** 其选择器需要 CLIP-ViT 倒数第二层
   [CLS]→patch attention；已核对本环境 transformers 4.57.6 的
   `Qwen3VLVisionModel` 全部源码，Qwen3-VL ViT **无 CLS token / class
   embedding**，该注意力结构性不存在。按任务规则不得以 embedding norm 等
   替代物冒充。其 ViT Key 向量虽可得，但选择器缺失使完整方法无从谈起。
2. **Libra：不可在当前范围可信实现。** (a) 重要性 α 是 **LLM 层** output
   attention（层级合并于 LLM 层 3/15/23），忠实移植需要把压缩点从 frozen
   pre-LLM 改为 in-LLM 并新建 eager-attention 执行路径，改变成本结构与
   "现场 prepare/encode/scoring/facility/completion/prefill/decode" 的全部
   计时语义；(b) 论文存在规范缺口：α 对多头/多个 output token 的聚合方式
   未说明，Eq.(6)(7) 的 W 记号维度（W∈R^{N×N′} 而 V′=W×V 要求 W∈R^{N′×N}）
   相互矛盾，区间划分的 tie-breaking 未说明；(c) 官方代码未发布（§4.1），
   亦无第三方忠实实现（2026-10-02 检索确认）；(d) 单流设计未定义 DS 四流
   适配。这些缺口任何一项都无法在不"自创算法"的前提下闭合。
3. **处置**：按预授权规则，"最近完整方法的数值对照"记为**未完成**；
   不为凑基线自创弱算法；继续完成主实验与两项算子消融。论文定位如实写为
   "与最近已有方法只有公式级对照、无数值对照，创新证据不完整"。

## 5. 样本暴露审计（exposure_manifest）

**输入源（全部枚举，脚本落盘并记录每源 SHA256/样本数）**：

1. `outputs/discovery/`：`m1_plan.json`（720 instances）、`sage_plan.json`
   （720）、`sage_labels_fit/val.json`、`s2c2_rescue_plan.json`（28 cases）、
   `s2a_gradient_viability.json`（15 cases）；
2. bank150：`discovery/common.sample_indices(n,150)` 在
   TextVQA_VAL/DocVQA_VAL/OCRBench 的历史重算；
3. `outputs/e0/e0_plan.json`：8 数据集全部 dev_rows + confirm_rows
   （E0 各 arm、M12/M13 使用）；
4. **实际生成记录兜底**：`outputs/**/acc/**/*.json` 全部 shard 的
   records 键并集（e0、m12、m13、anchor_merge、cross_stream、
   grounding、visual_calibration 等实际跑过的每一题）；
5. `outputs/anchor_merge_pilot/manifest.json`（DEV/CONFIRM，含 OCRBench
   36 行部分复用标注）。

**规则**：image identity 沿用 `e0_plan.image_key`；同一图像下的全部问题属
一个 cluster，cluster 内任一 row 暴露则整 cluster 暴露（跨 DEV/CONFIRM 重复
图像一并排除出 fresh）。每条来源的匹配按 row index 与 image_key 双轨。
输出 `exposure_manifest.json`：每数据集每 row ∈
{**exposed**, **fresh**, **unknown**}；unknown = 无法核实历史使用且不得计入
fresh（本轮若全部来源可枚举则 unknown 应为 0，出现即如实报告原因）。
fresh 仅表示"未用于本项目方案选择"，**不声称排除模型预训练污染**。
不以答案、难度或本轮得分选择 fresh。

## 6. 评测范围

1. **主面板**（BASE 与 MAIN025 全量配对）：TextVQA_VAL（5000）、
   DocVQA_VAL（5349）、OCRBench（1000）官方完整 split，全部 row，
   不挑子集。bank（EADP+facility）先全量冻结落盘。
2. **非回归面板**（BASE 与 MAIN025）：ChartQA_TEST、MMBench_DEV_EN_V11、
   MMStar、RealWorldQA、POPE 全量（prereg §2）。
3. **fresh 对照面板**（仅消融臂生成）：每任务按图像 cluster 用
   seed=20261002 独立打乱，整组依次纳入，≥200 题停止；不足 200 用全部
   fresh；为零则标"无法独立比较"。不裁同图问题凑整。manifest 与 SHA256
   在任何消融生成前落盘。BASE/MAIN025 在该面板的预测**取自全量结果的
   同题配对**（配置完全一致，不重新生成）；MAIN100/MAIN_SIM025 只生成
   这些题。用途：有界机制比较；不保证分辨 0.5 点差异；不因 CI 跨零扩样。
4. **正式方法另报**：每任务 full/exposed/fresh/unknown 的题数、图像数与
   配对效果；fresh 全部可用问题均分析（不限于 200 题面板）。无 fresh 时
   如实承认缺独立确认。

## 7. 正确性门（全过才允许全量准确率）

- **G1 公式**：MAIN025/MAIN100/MAIN_SIM025 与独立逐组 FP32 参考实现
  （显式循环，明确均值/求和、分母、anchor 自身、权重）rel<1e-5；bf16
  输出 cast 与生产路径 bitwise。
- **G2 路径**：真实 N>K 剪枝分支（N≈1024 样本）；多图隔离（拼接=独立
  结果）；重复/零向量、空 dropped、N≤K 退化确定。防 keep-all 冒充。
- **G3 复现**：每任务 ≥3 道已有样本，BASE 与 MAIN025 的 anchors、分组、
  主路/DS 特征、prefill logits、贪心输出 bitwise 一致（同一 backend）。
  λ=0 恒等门重跑；**λ=0 不能替代 λ=0.25/1.0 的正式复现检查**。
- **G4 不变量**：三臂共享同 keep/gid；DS==DS[keep]；输出数量/顺序/3D
  位置/cache 不变量。
- **G5 smoke**：各臂每任务少量（≤10 题/任务）先跑，查空答/截断/重复/
  覆盖/评分；不用于调参；provenance 全吻合的 smoke 预测可续用。
- **评分**：完整生成 + 官方评分；S0 硬失败（空 official / evaluate 异常 =
  score_failures.jsonl + 非零退出）；逐题键完全对齐；逐题聚合在官方舍入
  精度内复现 headline（不得 ±0.5 宽松）；OCRBench Final Score 仅全量集
  上报，子集用实际题数归一化逐题分。

## 8. 统计与决策（不事后换口径）

- **主判定**：完全沿用 prereg §3——主面板三任务逐题配对差 pooled（逐题
  加权合并，非等权 macro），image-cluster paired bootstrap 5000 次、
  seed=20261002、双侧 α=0.05；改善成立 = pooled CI>0 且三任务方向一致。
- **敏感性**：同索引 20000 次 bootstrap，不覆盖 5000 次判定；边界结论
  不稳时如实报告。
- **补充报告**：每任务与等权 macro 的 Δ/CI；不同估计量不混用。
- **fresh 面板对比**（各自独立分母）：MAIN025−MAIN100、
  MAIN025−MAIN_SIM025，各报每任务与等权 macro Δ 及 CI（cluster bootstrap，
  同 seed 家族）。均为补充比较，不得挑显著者证创新；CI 跨零 ≠ 相等/非劣。
  （MAIN025−完整方法：按 §4.3 未运行，报告醒目标明。）
- **预案**：prereg §4 机械条件（主面板 pooled CI 跨零且点估计 <+0.5）触发
  时运行一次 BOTH025 全量，标条件触发次要分析；未触发不运行。
- **两个问题分开回答**：相对硬剪枝是否有效 ≠ 是否区别于普通阻尼合并/
  已有方法；前者成立不推出后者。新数据无稳定收益即停；有收益但等价或缺
  对照则如实定位。

## 9. 效率

三任务各 10 题共 30 题配对，BASE / MAIN025（完整方法对照未实现，见 §4.3，
不虚构其计时）；固定 64 token、ignore_eos、断言实际长度、交错顺序、
充分预热；现场 prepare/encode/scoring/facility/assignment/completion/
prefill/decode 全路径计时（bank lookup 不替代）。CUDA 同步；每轮测量前
释放上一轮 out/state/KV 及一切 GPU tensor 引用，再同步并 reset peak；
记录只存 CPU 标量（修复旧 perf 脚本遗留 out 的问题）；各臂同清理策略。
报告 TTFT 边界（消息起至首 token，含预处理与全部现场阶段）、总时间、
各阶段、decode64、峰值 VRAM。

## 10. 运行计划与预计 GPU 用时（单卡 A40，无争用）

| 阶段 | 规模 | 预计 |
|---|---|---|
| bank 冻结（8 数据集全量 EADP+facility） | ~20k 图 | ~1.5 h |
| 正确性门 G1–G5 + smoke | 短 | ~1 h |
| 主面板生成+评分（BASE, MAIN025） | 11349×2 | ~19 h |
| 非回归面板生成+评分 | 14768×2 | ~25 h |
| fresh 面板消融（MAIN100, MAIN_SIM025） | ~600×2 | ~1 h |
| 条件预案 BOTH025（仅触发时） | 11349 | +9.5 h |
| 效率 | 30×2 | ~0.5 h |
| **合计（不含预案）** | | **≈ 48 h** |

驱动脚本 `Qwen_vl/scripts/anchor_completion_validation/`，分片可续跑，
以 nohup 后台提交并记录 PID/log。生成配置、bank、shard 元数据含
manifest/bank SHA、code commit、模型与环境哈希；只有全部 provenance
吻合才复用缓存。

## 11. 交付

`Qwen_vl/outputs/anchor_completion_validation/`：exposure_manifest、
fresh/对照 manifest、correctness、analysis_full/fresh/controls、perf、
coverage、配置/版本/哈希与偏离记录；完整逐题预测与评分分片（CI 可独立
复算）；小型统计产物入库，权重/图片/大 bank 留服务器路径+SHA256。
报告 `docs/anchor_completion_validation_report.md`（每任务 full/fresh 的
n、图像数、准确率、Δ、CI；缺失对照醒目标明；不省略失败/unknown/退化
任务/未执行项）；project_handoff 增量更新（保留旧历史）。提交并推送
本分支，不合并 main，无 AI 署名。

## 12. 偏离记录

（运行中追加；当前为空。）
