# EADP 论文 Qwen3 对照方法与 DeepStack 公开代码核查

核查日期：2026-11-02。本文只报告一手公开论文和固定提交的源码可证明的事实，不把公开源码行为当成论文实际运行日志。

[EADP 论文表 4](https://arxiv.org/html/2607.02484v1#S5.T4)的 Qwen3-VL-8B 剪枝对照为 DivPrune、CDPruner 和 EADP；另列不剪枝的 Qwen3。HiPrune 和 FastV 不在该表。K256 的 DocVQA 分别为 55.8、53.0 和 62.8；这些均为论文值。

EADP 作者固定提交 [e1a088 的 Qwen3 代码](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py)：CDPruner、DivPrune 和 EADP 的实际剪枝入口共用 generate_inner_transformers_fixed（477–506行），视觉结果只取主特征并传剪枝后的 inputs_embeds 和 attention_mask，不向语言模型传 DeepStack 及其视觉位置 mask。另有 Qwen3 HiPrune 代码，但它不在论文该表。Qwen3 不剪枝基线走 295–305 行原生多模态生成路径。在先前固定16题实测中，无剪枝原生路径语言模型接收三组 DeepStack，而 EADP 剪枝路径未接收。

[DivPrune 原论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.pdf)和 [CDPruner 原论文](https://arxiv.org/html/2506.10967)未报告 Qwen3-VL 结果；EADP 作者提供了 [Qwen3 适配评测代码](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/README.md)。因此 Qwen3 比较数不是直接摘自这两篇原论文，合理推断需有后续适配评测，但现有公开材料未证明每一数值的运行人、代码版本、完整参数与实际 DeepStack 状态。论文表1部分 VizWiz 星号结果明确注明验证集复现，不能据此断定其所有表格均由作者重跑或全部引用。

公平性：公开剪枝路径中三个方法均未向语言模型传 DeepStack，不能说只有 EADP 关闭；不剪枝 baseline 与剪枝路径还存在其他区别。我们目前只有 EADP 同环境 DeepStack 开/关完整分数（DocVQA 61.166→67.903），不能将它直接与其他方法论文结果排名。对比方法需在同一模型路径、数据、预算和 DeepStack 设置下重新评测。
