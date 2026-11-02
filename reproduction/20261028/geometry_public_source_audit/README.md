CDPruner 一手材料支持当前 post-projector 实现，不能用它证明需要 pre-projector 修复。

官方仓库已固定到 `9541616c40fcd5625de1cdb8ea6c33c129eb7864`。其 `llava_arch.py:148` 先调用 LLM 的 mm_projector，151–153 才计算 DPP 视觉余弦相似度；论文 §3.1 定义 Hv=g(fv(X))，§3.2 的核使用这个 Hv。跨模态相关性走另一条 CLIP 路径：`clip_encoder.py:39–45,94–95` 加载预训练 visual_projection/text tower，并应用 post_layernorm 和 CLIP visual_projection。它与 LLM connector 是两个投影。论文 §3.3 对没有配对文本编码器的 Qwen2.5 另用 projector 输出和 LLM instruction embeddings mean；这不是 Qwen3 主表配方。

EADP 本文65–79行把 V 定义为 vision encoder token，再单独描述 projected LLM tokens；73–74行的简记只明确适用于跨模态相似度。facility/附录仍使用 V 和符号 d_v，图省略两个 projector，没有实际1024/768/4096维度。因此它允许提出一个字面 pre-connector 几何假设，但不能认定论文唯一采用它，更不能认定源码出错。

可证伪的下一项只能明确为 LLaVA 原生 mm_projector 输入1024 vs 输出4096的 facility 相似度；相同576索引、原重要度/指导/参数/归一化/greedy及最终投影特征全部保持。先做有限 source-anchor 机制比较，再由真实选择变化决定后续问题。本次没有准备运行器或启动任何实验。现有缓存不含实际 raw1024；较新 NeXT 缓存有 CLIP768，也不能冒充它。

原始来源：

- [CDPruner 官方代码](https://github.com/Theia-4869/CDPruner/tree/9541616c40fcd5625de1cdb8ea6c33c129eb7864)
- [CDPruner 论文 v1](https://arxiv.org/html/2506.10967v1)
- EADP 官方 TeX archive 已有本地归档；本目录保存 exact methodology、appendix 和 framework PDF。

完整路径、SHA、行号及事实/假设边界在 `geometry_source_evidence_20261029.json`。只进行了只读源码/静态配置和官方来源下载，未加载 torch/模型、执行 selector、读取 GT/答案或评分。
