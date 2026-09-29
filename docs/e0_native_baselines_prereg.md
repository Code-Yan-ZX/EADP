# E0 预注册：Native Qwen3-VL 修复、基线复现与编码器轴余量审计

状态：**预注册，尚未运行**。日期 2026-09-29，基于 `main@cb5821a`。本地会话撰写，服务器 Claude Code 执行。

## 0. 目的与非目的

E0 只回答三个问题，不提出新方法：

- **Q1 修复效应**：恢复 DeepStack 与 3D mRoPE 之后，历史结论（B2 等）在 native 模型上变化多少？
- **Q2 基线地形**：在同一 native 执行图、同一数据、同一计时协议下，24/25 年强基线、EADP、2026 年 PACE/CIVIC 和降分辨率对照的准确率–TTFT 曲线分别在哪里？
- **Q3 编码器轴余量**：native TTFT 中，预处理加视觉编码占多少？这个份额随输入分辨率怎样变化？它决定后续 E1（编码器端稀疏编码）值不值得做，以及要打败谁。

非目的：

- 不调任何方法的超参；不读取 CONFIRM 集；
- 修复本身不是研究贡献；
- E0 结果不能用来宣称新方法。

背景证据见 `docs/research_reset_20260929.md`、`docs/scoring_search_m2_system_amendment.md` §7、`refine-logs/EXPERIMENT_RESULTS_20260929.md`。

## 1. 固定设置

- 模型 Qwen3-VL-8B-Instruct，bf16，SDPA，batch 1，greedy（沿用 VLMEvalKit 配置 `top_k=1`），与 M2 同一台 A40 主机。
- 主输入协议沿用项目历史的 fixed 1024×1024（`expand2square` + resize），得到 64×64 patch，合并为 1024 个视觉 token。这是 harness 约定，不是 Qwen3-VL 默认的动态分辨率，报告中如实标注。
- 预算 K 指进入 LLM 的视觉 token 数。主预算 K=256，曲线预算 K∈{64,128,256}。
- 记录 torch / transformers / CUDA / 驱动版本。transformers 版本一旦变化，必须重跑 N1。

## 2. 数据划分（新划分，按图像隔离）

1. 候选池为每个数据集的全部行。剔除任何历史阶段用过的行，来源包括：
   - `Qwen_vl/outputs/discovery/` 下所有 `*_plan.json`；
   - SAGE fit/val 标签文件；
   - `common.sample_indices(n, 150)` 生成的 bank；
   - S2-A causal cases。

   剔除按图像进行：同一图像的任何一题被用过，整张图的所有题都剔除。
2. 在剩余池上，用固定种子 `E0_SEED=20260929` 按图像分组，50/50 划分为 DEV 和 CONFIRM。DEV 每个数据集上限 300 题，超出时按打乱后的图像顺序截断。
3. CONFIRM 在 E0 中完全不读答案、不运行生成，只写出索引并存档 sha256。
4. 数据集：
   - OCR 面板：TextVQA_VAL、DocVQA_VAL、OCRBench、ChartQA_TEST；
   - 通用面板：MMBench_DEV_EN_V11、MMStar、RealWorldQA、POPE。
5. 可用量不足时（预计 OCRBench 会出现），按实际数量报告，不从 CONFIRM 借题。
6. 评分沿用 VLMEvalKit 官方规则。需要外部 LLM judge 的数据集，一律改用内置的规则抽取路径，并在报告中声明。**禁止把任何数据发往外部 API。**
7. 输出 `Qwen_vl/outputs/e0/e0_plan.json`，包含每个数据集的 DEV/CONFIRM 数量、剔除数量和 sha256。

## 3. Native 修复规范

### 3.1 现状

- B0（`Qwen3VLChatFixedRes`）走 `model.generate(**inputs)`，本身已经是 native 路径。
- 所有剪枝 wrapper（`model_fixed_res.py` 中的 CDPruner、EADP、DivPrune、HiPrune，以及 discovery 各阶段的引擎）只传 `inputs_embeds`。结果是 DeepStack 的三层注入缺失，位置退化为 1D running index（见 M2 §1.4）。因此历史剪枝结论只适用于这个改动过的模型。

### 3.2 原则：用 stock forward 注入，不复制实现

新建 `Qwen_vl/model/native_qwen3.py`。修复只在一个地方实现，所有 arm 共用；每个方法只提供一个"选择器"：输入视觉特征（及文本），输出 `keep_idx`（按光栅顺序升序、互不重复）。

修复不重写 decoder。做法是给 stock `Qwen3VLModel.forward` 喂一组一致的输入，按以下顺序：

1. **视觉编码**：调用 `model.visual(pixel_values, grid_thw)`，得到主特征 `V` 形状 `[1024, D]`，以及 `deepstack_feature_lists`。后者包含 3 个张量，每个形状 `[1024, D]`，都是 merge 之后的特征。所有 N0 以外的 arm 都从这一次视觉输出做选择。
2. **按同一索引收缩所有流**：

   ```text
   V_k = V[keep_idx]
   DS_k[i] = DS[i][keep_idx]                      # 三层 DeepStack 必须用同一个 keep_idx
   pos3d_full = get_rope_index(input_ids_full, image_grid_thw, attention_mask_full)[0]
                                                   # [3, 1, L_full]，用未剪枝的完整序列计算
   keep_seq = text positions ∪ (image span start + keep_idx)
   position_ids = pos3d_full[:, :, keep_seq]       # 保留 token 继承原始 (t,h,w) 坐标
   rope_deltas = pos3d_full.max() + 1 - len(keep_seq)
   ```

3. **构造 prefill**：`inputs_embeds` 用文本 embedding 加上 `V_k`，按原顺序排列。`visual_pos_masks` 是长度 `len(keep_seq)` 的布尔向量，只在 K 个视觉位置为真。调用 language model 时显式传入 `position_ids`、`visual_pos_masks` 和 `deepstack_visual_embeds=DS_k`。
4. **decode**：把 `model.model.rope_deltas` 设为上面的 `rope_deltas`，后续每步位置 = `cache_position + rope_deltas`，与 stock 的 decode 分支一致。实现方式二选一：自写 greedy 循环，或者 hook `prepare_inputs_for_generation`。选定后所有 arm 使用同一种。
   - 注意 stock `forward` 的判断：当 `position_ids is None` 且 `cache_position[0]==0` 时，它会重新调用 `get_rope_index`。所以 prefill 必须显式传 `position_ids`；decode 阶段传 `None`，依赖已缓存的 `rope_deltas`。
5. **LLM 内剪枝（FastV / PDrop / SparseVLM）**：视觉 token 在第 L 层被删时，以下对象必须一起按同一索引收缩：
   - hidden state；
   - 3D position 与 `position_embeddings`（cos/sin）；
   - attention mask；
   - 每层的 KV cache。

   DeepStack 只注入第 0–2 层；删除发生在第 2 层之前时，`visual_pos_masks` 和 `DS_k` 也要同步收缩。decode 继续沿用第 3 步的 `rope_deltas`，这个值由剪枝前的完整序列决定，不因层内剪枝改变。
6. **可选 flag**：`deepstack={on,off}` 与 `pos={mrope3d,1d}` 两个 flag，只用于 N-ablation（§5 的 A1/A2），默认值为 `on`/`mrope3d`。

### 3.3 正确性门（任一失败即停，不产出准确率数字）

| 门 | 检验 | 通过标准 |
|---|---|---|
| N1 identity | K=1024、`keep_idx`=全部时，与 stock `model.generate(**inputs)` 在 16 个 DEV 样本上比较：prefill logits、每一步 decode logits（32 步） | max\|Δlogits\| ≤ 1e-3；32 步 greedy token 全部相同。若非零，必须解释来源（参考 M2 G-C 的 mask sentinel 教训） |
| N2 DeepStack 生效 | K=1024 时关掉 deepstack，与 N1 比较 | logits 必须明显不同（max\|Δ\| > 0.1），证明注入确实发生，而不是被静默跳过 |
| N3 位置 | K=256，任取 keep_idx | 每个保留视觉 token 的 (t,h,w) 等于它在完整序列中的坐标；文本位置单调；decode 第 n 步位置 = prefill 最大位置 + n |
| N4 不变量 | 12 个样本 × 每个 arm | 视觉 token 恰为 K 个，无重复，无越界；DS_k 三层长度都是 K；mask、position、cache 长度一致；`layer_calls = 36·(1+decode_steps)`，没有隐式重算 |
| N5 历史一致 | 新代码在 legacy 路径（deepstack off、1D 位置）下运行 B2，与 SAGE 确认集中已存档的 B2 预测比较。SAGE 720 已从 DEV 剔除，所以不会污染 DEV | 在 32 个旧样本上预测一致率 ≥ 95%。用来确认新代码没有顺带改变选择器 |

门的结果写入 `Qwen_vl/outputs/e0/e0_gates.json`。

## 4. Arms

所有 arm 都在 native 路径上运行（DeepStack on、3D mRoPE）。除非另有说明，均为免训练方法，超参固定为原论文或官方代码的默认值，**E0 不调参**。

| ID | 方法 | 出处 | 剪枝位置 | 实现来源与移植要点 |
|---|---|---|---|---|
| B0 | 不剪枝，1024 token | — | — | stock `generate`，作为上限参考 |
| R-res | 降分辨率，不剪枝 | 对照 | 输入端 | fixed 分辨率改为 32·√K 像素：K=256 用 512×512，K=64 用 256×256；K=128 时，每个合并 token 对应 32 像素，所以边长必须是 32 的倍数，取 352（121 token）和 384（144 token）两档都跑，并记录实际 token 数。直接走 stock `generate`。这是最关键的"平凡对照" |
| FastV | 文本对视觉 attention | ECCV 2024 | LLM 第 2 层后 | 官方代码 [pkunlp-icler/FastV](https://github.com/pkunlp-icler/FastV)。按官方 R、K=2 设定移植。需要显式 attention 权重：只对第 K 层单独计算 attention 分数，其他层保持 SDPA |
| PDrop | 分阶段递减 | CVPR 2025 | LLM 多层 | 官方代码 [Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)。按官方的层与比例表移植；以"平均视觉 token 数 = K"对齐预算。报告里同时写出真实的每层 token 数 |
| SparseVLM | 文本引导 + 回收 | ICML 2025 | LLM 多层 | 移植官方实现；token recycling 保留。若 Qwen3 上无法完整移植，降级为不含 recycling 的版本，并明确标注 |
| VisionZip | dominant + contextual 合并 | CVPR 2025 | ViT 输出后 | 官方代码 [dvlab-research/VisionZip](https://github.com/dvlab-research/VisionZip)。用 ViT 倒数第二层的 CLS 或平均 attention 选 dominant，其余 token 合并为 contextual。Qwen 没有 CLS，按官方 Qwen2.5-VL 适配方式处理。合并 token 的 3D 坐标取簇内被选中者（dominant 最近邻）的坐标，DS 流按相同分配求均值 |
| DivPrune | 多样性选择 | CVPR 2025 | ViT 输出后 | 用 repo 已有的 `model/divpruner.py` 输出 keep_idx |
| CDPruner | 条件 DPP | 2025 预印本（会议待核实） | ViT 输出后 | 用 repo 已有的 `model/cdpruner.py`。确认发表会议前，不计入 `Best24/25` |
| HiPrune | ViT 层级 attention | ACL 2026 Findings | ViT 输出后 | 用 repo 已有的 `model/hipruner.py`。属于 2026 方法，不计入 `Best24/25` |
| B1 | EADP，facility 选择 | ECCV 2026 | ViT 输出后 | `model/pruner.py`，alpha 0.5，beta 2.0 |
| B2 | EADP 分数 + block8 | 本项目参考 | ViT 输出后 | `instrumented.select_block_greedy(block=8)` |
| PACE | APC 自适应降采样 + DDAE | EMNLP 2026 Findings | 编码前 + LLM | 基于官方代码 [jjL357/PACE](https://github.com/jjL357/PACE)，原本为 Qwen2.5-VL。见 §4.1 |
| CIVIC | 编码前 learned anchor 聚合 + 压缩 KV 的 ViT | arXiv 2605.28115 | 编码前 | 需要训练，且未找到公开代码。见 §4.2 |

### 4.1 PACE 移植

1. **先在 Qwen2.5-VL-7B 上复现官方结果**：用官方 `reproduce_10_percent.sh`，跑 RealWorldQA、MMStar、ChartQA。repo 中给出的参考分数为 68.37、59.08、73.52。偏差超过 2 分时先排查环境；不排查清楚就不移植。官方依赖 transformers 4.57.6 和 lmms-eval，使用独立 conda 环境，不要污染主环境。
2. **移植到 Qwen3-VL**：
   - patch 由 28 改为 32（16×2 merge）；
   - Qwen3 ViT 没有 window attention，`ShallowFeaturePreview` 中的 `get_window_index` 分支删除，所有层都用 full attention；
   - `fast_pos_embed_interpolate` 必须按降采样后的网格重新计算；
   - DDAE 所需的 LLM 端信号接到 §3.2 第 5 步的 in-LLM 剪枝接口；
   - DeepStack 流按 DDAE 的最终保留索引同步收缩。
3. **预算对齐**：PACE 的 APC 是逐图自适应的，每张图的最终 token 数可能不同。报告每个 arm 的平均视觉 token 数和分布，按平均 K 对齐。同时记录 ViT 实际处理的 patch 数，因为这正是 PACE 省下时间的来源。
4. 若 Qwen3 移植失败，可以采用降级方案：在 Qwen2.5-VL-7B 副面板上，同时运行 PACE、R-res、B2 和 VisionZip，保证它们可比，并在报告中写明原因。**不得用 PACE 论文中的数字直接和我们的 Qwen3 数字比较。**

### 4.2 CIVIC 的处理

- 实施前先在 GitHub、项目页和作者主页检索代码；运行时再检索一次，把查询和结果写进报告。
- **找到代码**：按官方配置在 Qwen3-VL-8B 上训练。训练需要 GPU 小时，**先把预算估计报告给人，等人批准后再开始**。E0 的其他部分不等 CIVIC。
- **没有代码**：E0 不重实现 CIVIC。
  - 原论文只在 Qwen3-VL-2B 和 RTX 4090 上评估了 MMMU、MathVision、ODinW-13、RealWorldQA、VideoMME，与本项目设置不重合。
  - 报告中把它列为"未复现的近邻"，并写清两点：它需要训练；它的 attention 使用压缩 KV anchor。
  - 若后续方法进入论文阶段，再决定是否实现一个忠实的简化版。简化版必须标注为"复现版"，不得冒充原方法。

### 4.3 历史对照 arm（只跑 K=256）

- **A1**：B2 在 legacy 路径（deepstack off、1D 位置）下运行，即历史设定。
- **A2**：B2 在 deepstack on、1D 位置下运行。

A1、A2 与 native B2 三者比较，用来量化修复效应（Q1），同时拆分 DeepStack 和位置编码各自贡献多少。

## 5. 测量

### 5.1 准确率

- 所有 arm × K∈{64,128,256} 都在 DEV 全部 8 个数据集上评测。B0 只跑一次。A1、A2 只跑 K=256 的 OCR 面板。
- 生成配置与 VLMEvalKit 中 `Qwen3-VL-8B-Instruct-1024` 一致，`max_new_tokens` 保持原值。另外记录被截断（达到上限）的比例：任一 arm 截断率超过 2% 时，要在报告中调查原因（M2 的 PRESERVE 缺陷就是先通过截断率暴露出来的）。
- 报告以下内容：
  - 每个数据集的官方分数；
  - OCR macro、通用 macro、总 macro；
  - 对 B2 和对最强 24/25 基线的 paired bootstrap 95% CI（按图像做 cluster bootstrap，10,000 次）；
  - rescue/break 计数。

### 5.2 效率（沿用 M2 amendment §7 的 paired-block 协议）

- K=256 的所有 arm，加上 R-res、PACE 的全部 K，在同一进程内测量：
  - 15 个 warm-up block，至少 100 个测量 block；
  - 每个 block 内随机打乱 arm 顺序；
  - 每个 block 用同一个样本。样本从 DEV 中按数据集分层随机抽取，**要包括高、低不同原始分辨率的图**。
- 分段计时：
  1. 预处理（CPU wall，包含 PACE 的 APC）；
  2. ViT（CUDA event）；
  3. 选择或合并机制；
  4. LLM prefill；
  5. 首 token 的 TTFT（wall）；
  6. decode 32 步（`ignore_eos`）；
  7. 峰值显存。
- 同时记录每个 arm 的 ViT 输入 patch 数和进入 LLM 的视觉 token 数，以及解析 FLOPs，分开写 ViT 和 LLM 两部分。
- paired contrast 与 M2 相同：block 内配对的中位数差，加上 block-bootstrap CI 和胜负计数。

### 5.3 编码器轴余量扫描（Q3）

- 仅 B0 和 R-res。fixed 分辨率取 {256, 384, 512, 768, 1024, 1280, 1536} 的方形；1280 和 1536 要调大 `max_pixels`，超出显存则停止扩大并记录。
- 每个分辨率测 ViT 时间、LLM prefill 时间和 TTFT，并跑 DEV 的 OCR 面板准确率。
- 输出两条曲线：ViT 占 TTFT 的比例随分辨率的变化；准确率随分辨率的变化。

## 6. 决策规则（预先写定，E0 结束后机械判定）

记号：
- `Best24/25(K)` 是 FastV、PDrop、SparseVLM、VisionZip、DivPrune 在预算 K 下 OCR macro 最高的那个；
- `ViTshare` 是 native B2 在 K=256 下 (预处理 + ViT) / TTFT 的比例。

- **D1 修复效应**：若 |native B2 − A1| ≥ 2 macro（CI 不含 0），则所有历史结论降级为"只适用于 legacy 模型"，在 handoff 中标注。
- **D2 平凡对照是否致命**：若 R-res 在 K=256 的 OCR macro ≥ `Best24/25(256)` − 1，且 TTFT 更低，说明"在 ViT 输出后剪枝"在 OCR 上已经被降分辨率支配。此时 E1 的比较对象改为 R-res 和 PACE，不再以 24/25 方法为主。
- **D3 编码器轴是否值得做**：若 `ViTshare` ≥ 0.5，且 PACE 相对 B2 的 TTFT 下降 ≥ 15%，同时 OCR macro 下降 ≥ 2，那么"在不掉 OCR 精度的前提下减少编码器计算"就是有空位的具体目标，**进入 E1**。
  - 若 PACE 在 OCR 上不掉点，而且 TTFT 更快，说明这个空位已被 PACE 占据。停止 E1 立项，回到方向选择。
  - 若 `ViTshare` < 0.35，编码器轴的余量不足，同样停止。
- **D4 基线可信度**：任何移植基线在 native B0 基础上出现明显异常（例如 K=256 时 OCR macro 比 R-res 低 10 分以上），先对照官方代码排查，才能作为基线报告。如果无法排查清楚，就标注"移植存疑"，并从 `Best24/25` 中剔除。

## 7. 产物

- 代码：
  - `Qwen_vl/model/native_qwen3.py`：修复与统一注入；
  - `Qwen_vl/model/baselines/`：fastv.py、pdrop.py、sparsevlm.py、visionzip.py、pace_qwen3.py；
  - `Qwen_vl/scripts/e0/`：plan、gates、accuracy、perf、res_sweep、analyze。
- 数据：
  - `Qwen_vl/outputs/e0/` 下的 `e0_plan.json`、`e0_gates.json`、`e0_acc_*.json`、`e0_perf_paired.json`、`e0_res_sweep.json`、`e0_verdict.json`。
- 报告：`docs/e0_native_baselines.md`，内容包括：
  - 门结果；
  - 准确率表，K=256 为主，另附 K=64 和 128；
  - 准确率–TTFT Pareto 图；
  - 修复效应；
  - D1–D4 判定；
  - 偏差与修正记录。
- 所有偏离本预注册的改动都写入 `docs/e0_native_baselines_prereg_amendment.md`，**必须在看到受影响的数字之前写入**。

## 8. 算力估计（粗算，待服务器实测校准）

- 准确率部分：约 13 个 arm × 3 个 K × 最多 2400 题 ≈ 9 万次生成，外加 A1、A2 和分辨率扫描。按 A40 每题约 0.6–1.0 s 计，需 15–25 GPU 小时，可按数据集并行到多卡。
- 性能部分：约 2 GPU 小时。
- PACE 在 Qwen2.5-VL 上复现：约 2 GPU 小时。
- CIVIC 训练：**不在默认预算内**，需人工批准。

先跑门和一个 30 题的小 pilot，按实测速度重估总耗时，并报告给人，再启动全量运行。
