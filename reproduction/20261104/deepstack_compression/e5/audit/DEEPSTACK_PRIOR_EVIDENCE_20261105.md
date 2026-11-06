# DeepStack 相关来源与证据边界（2026-11-05）

本备忘仅整理远端静态阅读的一手资料，未下载或运行外部代码、模型与数据，也不把外部数值当成本项目结果。

## 未验证的三槽消融线索

[第三方 Qwen3-VL fork](https://github.com/adityakothuri/qwen3-vl-deepstack-optimization) 中有 [论文草稿](https://raw.githubusercontent.com/adityakothuri/qwen3-vl-deepstack-optimization/main/paper_draft.md)、[三槽干预代码](https://raw.githubusercontent.com/adityakothuri/qwen3-vl-deepstack-optimization/main/src/deepstack/ablation.py) 和 [汇总 JSON](https://raw.githubusercontent.com/adityakothuri/qwen3-vl-deepstack-optimization/main/results/20260603_184741/sensitivity.json)。草稿有作者和场所占位，作者的[项目页面](https://adikothuri.com/work/qwen3-vl-deepstack)出现“Generated stand-in”标记；未核实其发表、运行真实性或源码与结果的绑定。不能因这些迹象断言造假，也不能把草稿的结果和机制解释视为已证实。仓库是 Qwen 官方仓库的第三方 fork，顶层 README 延续原项目说明，并非 Qwen 官方对该研究的背书；远端检查未确认具体提交 SHA。

代码定义 full、逐槽 drop、逐槽 keep、全部 drop 八种条件，通过语言模型前向钩子将指定槽替换成同形零张量，保留槽和主视觉序列。与本项目 E2 的三槽完整因子干预有明确的设计重叠，不宜宣称该设计首次出现。已查看的汇总文件记录模型侧来源路径、dtype、设备和任务汇总，但未提供能在该文件内独立逐题重建的预测、权重哈希和完整进程退出证据；不能排除其他未查看文件存在进一步材料。

[预算实验源码](https://raw.githubusercontent.com/adityakothuri/qwen3-vl-deepstack-optimization/main/src/experiments/exp_budgeting.py) 使用顺序跳过可用样本行的 sweep/validation，所查材料未证明按图像或文档组隔离，亦无法确认与本项目样本不重叠。等预算单位是保留的非零 DeepStack 特征行数，主序列和注入张量未缩短，作者源码明确不据此声称延迟改善。其置信区间计算各条件的样本均值，而非图像组配对差值区间；异常发生前已追加的条件记录可能留下不同分母，这是代码路径风险，未证实实际发生。草稿的跨槽冗余或效果等价属于待验证解释，不由置信区间重叠自动证明。

本项目 E2 使用固定 8B 权重、压缩后的 K256、固定输入及位置、图组划分和配对检查；这些条件与该线索不同。E3 的 PERM/MEAN 干预是否已有完全相同先例，以上材料不能证明；不应据此声称首创。

## Qwen 报告与相关方法

- [Qwen3-VL 技术报告](https://arxiv.org/html/2511.21631) 包含 DeepStack 结构有无的训练消融；这不是同一冻结 8B 权重上的三槽独立开关，也不能替代 E2/E3 的因果对照。
- [PixelPrune 论文](https://arxiv.org/html/2604.00886) 研究 Qwen3-VL 视觉编码器不同位置的 token 剪枝及与中间特征路径相关的计算，与 E4/M2 的剪枝位置和计算问题有交叉。论文设置与本项目固定 K 的逐槽干预和同后端计时不同，不可将其性能数字当作本地结果。
- [TensorRT-Edge-LLM 的 DART 文档](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html) 描述推理时联合处理主视觉特征、DeepStack 特征和位置的压缩路径，可作为实现与跨后端对照线索；它不能替代本项目对三槽各自作用的比较，跨后端时间也不能直接解释为同实现的净收益。

E4 的置零、真正跳过注入、视觉主序列缩短和端到端延迟是不同操作；效率判断需要实际执行路径、相同计时条件和真实成本证据。整体新颖性应具体落在可复核的干预、设定与结果上，而非泛称首次发现跨层视觉注入或首次进行三槽消融。
