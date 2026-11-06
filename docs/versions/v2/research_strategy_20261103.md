# EADP 后续研究：公平基线与创新路线

日期：2026-11-03。目标是在视觉 token 剪枝/压缩中形成可部署、可重复的更强方法；最终架构和创新点由实验证据决定，不预设模块组合。

## 已知结果和边界

本机 Qwen3-VL-8B、DocVQA 验证集 5,349 题、1024→256，EADP 的 DeepStack 关闭/开启 ANLS 为 61.1660/67.9031，DivPrune 为 55.7750/63.6164，CDPruner 为 53.0448/59.5965。每个方法的两臂选中位置相同，三种方法各有自己的选择器。三方法的 DS 收益差中差区间均包含 0，尚无证据证明某方法更依赖 DeepStack。完整评分与证明见 `reproduction/20261103/qwen_other_methods_ds_pair/background_completion_20261103/comparison/RESULT.md`，差中差见 `reproduction/20261103/qwen_other_methods_ds_pair/gain_interaction_20261103/RESULT.md`。

该对照沿用旧 1D 位置路径，不是 Qwen3 全原生 mRoPE；开启 DS 是公开实现的修复，不代表论文实际运行配方，也不是新剪枝贡献。EADP 作者当前公开 Qwen3 剪枝路径对三种方法均不传 DS；论文未披露足够细节来证明主表运行时的 DS 状态。DivPrune 和 CDPruner 原论文使用的骨干也不等于 Qwen3。DS 已是 Qwen3 原生功能；已有 [NVIDIA Qwen3 剪枝实现](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html)同步裁剪 DS 和 mRoPE，[Hi-Lo Qwen3 源码](https://github.com/sealost/Hi-Lo_Prune/blob/main/hilo_prune/models/qwen3_vl/modeling_hilo_qwen3_vl.py)也包含 DS 路径。不可概括其他论文均未使用。

## 评价协议与基线

1. 保留公开实现原设置作为复现锚点；建立统一的 DS-on、原始视觉坐标及 mRoPE、同模型权重/图像预处理/提示/解码/指标的强基线。先在 K=1024 不剪枝条件下核对原生生成路径的 prefill 位置、RoPE 增量、解码 cache 和 logits，再检查 K=256 逐索引选择、DS 对齐。上述改动需要在 EADP、DivPrune、CDPruner 上一致应用，当前分数不能无条件沿用到新位置协议。
2. 当前三个已运行的对照是基础，而不是整个领域排名。选取公开实现可运行且有关联的近期强方法并核查版本、原生 DS/mRoPE、题目和数据许可。优先审计 [Hi-Lo Prune](https://github.com/sealost/Hi-Lo_Prune) 的 Qwen3 图片实现与 [PriorTR](https://github.com/CodeChildCZJ/PriorTR)；其剪枝层、合并/前向成本与 EADP 不同，应同时比较准确率和真实成本。Hi-Lo 的仓库目前未打包完整 benchmark 流程。保留随机选择和不剪枝原模型作为参照；有训练的方案单独交代训练数据/计算。
3. 在同一 benchmark 的题目/评分/模型输入下，至少对 K=128/256/512 和不剪枝建立精度—真实成本曲线；对于分层或动态预算方法，报告每层 token 数和预算分布，不能把最终 K 视为等算力。覆盖 DocVQA、TextVQA 及通用视觉理解任务，在可行时加另一骨干；测冷启动、prefill/TTFT、端到端延迟、吞吐、峰值显存和 selector 开销，控制 batch、生成长度及硬件。竞争方法可保留原生官方运行环境，公开差异，不能为了同代码路径破坏其原始方法。
4. 模型开发和调参按图像划分训练/验证/独立测试，预注册对比和主要指标；报告逐题差、image-cluster 置信区间、多个 seed 及失败案例。少量预探索不能作为正式验证集，置信区间含 0 不等于非劣。

## 优先研究假设：多层视觉信息能否帮助确定保留集合

问题：当前 DS 已在保留位置注入额外特征，依然有大量视觉位置在送入语言模型前被丢弃；在真实回答任务下，是否存在仅靠主视觉流难以发现、但可利用多层视觉证据识别的高价值信息？这是候选假设，不是现有结果已证明的事实。

先做低成本 oracle/消融，而不是训练模型：在冻结基线、固定 K256、图像隔离的探索集上，记录主流与 DS 的实际选点与输出；比较仅主流与多流的信息、同预算随机/重要度替换、简单均值或已有选择/合并基线，分析逐题真实生成与答案损失。严格区分 selection、特征补偿和 merging，确保预算和计算开销可比；把额外高维缓存和 oracle 成本计入分析。若离线 oracle 在独立验证集上的多流额外贡献无法稳定超过简单基线，或配对增益区间已能排除预先指定的最小有意义收益，应终止或改变假设。置信区间仍覆盖目标值则为证据不足，不等于无效。

此前纠错后的 cross-stream scorer 未支持相对 EADP 的非劣，不能简单重跑多层平均/求和并声称创新：`docs/stage1_cross_stream_pilot_report.md`。若 oracle 证实明确、可泛化的多流增益，再研究 forward-only 的轻量 selector 或压缩映射，离线 teacher 的训练成本单独报告；随后多预算多任务真实生成及同机速度验证。机制设计需与近作逐项比较，例如 [Hi-Lo Prune](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html) 在早层保留/融合信息、[ET-Prune](https://arxiv.org/abs/2608.01979) 关注文本证据和动态预算；后者当前仓库仍标记代码待发布。不能只以“文本引导、保护重要 token、动态预算”宣称新颖。

## 备选路线

- 同图多问共享视觉特征与问题条件选择，以是否真实降低摊销 TTFT、缓存内存和质量损失为判断依据；与缓存全部视觉特征后逐题运行 EADP 的强对照比较。同图共享 token 本身已有[先行工作](https://arxiv.org/abs/2610.23974)，需要证明不同的选择机制和可复现的系统收益，不能把仅有单图单问的数据当成多轮收益。
- 精确保留选点的 EADP facility 加速是系统方向候选：逐步索引、ties、生成输出须一致，并测真实端到端收益。当前 M14 为草稿；单纯采用既有 lazy-greedy 不能宣称新的算法贡献。

公平复现、修复原生模型能力以及方法创新分别报告。只有在预定模型、预算、数据集和计算约束下，对已核验的强方法在准确率—实际效率曲线上取得可重复优势，才能限定条件声称相应设置下的 SOTA。
