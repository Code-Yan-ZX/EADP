# Scatter Graph：三锚完整模型首 token 试验

该候选将作者选择器的一处高级索引赋值改为 scatter，其余贪心张量运算沿用作者实现，并对固定 K256 捕获 CUDA Graph。此前 32 张真实图、三个预算合计 96 个案例的完整有序选点和配额与作者一致；这只覆盖该组件样本。此处接回同一 Qwen3-VL-8B、P3 和三路 DeepStack，三个 DEV 锚点各预热一对、再做五对随机相邻的固定 32 token 生成。

| 锚点 | 作者中位 TTFT | scatter Graph 中位 TTFT | 中位数变化 |
|---|---:|---:|---:|
| DocVQA | 396.43 ms | 381.63 ms | −3.73% |
| TextVQA | 349.93 ms | 329.04 ms | −5.97% |
| AI2D | 357.42 ms | 333.03 ms | −6.82% |

三锚各自中位数的中位为 357.42→333.03 ms，下降约 6.82%。36 次调用、18 对的完整选择顺序、quota 和 32 个输出 token 均一致。独立结果和汇总审查分别见 `../../e1_independent_review/scatter_graph_ttft_v2_result_review.json`、`../../e1_independent_review/scatter_graph_ttft_v2_summary_review.json`。

首次 Graph 捕获约 329.58 ms、PyTorch allocated 增约 4 MiB，另列且不包含在预热重放 TTFT；该分配器变化不是设备峰值显存。首 token 后持久化原始输出，故全生成耗时不作性能指标。正式验收器在 GPU 执行完毕后、验收者读计时前冻结；首次独立审查脚本语法错误在读数据前退出并保留，修正重跑通过。

三题、warm-cache、固定 32 token、带计时插桩，且外部 CPU 活动的逐样本重叠未知；这些数字不能证明自然生成、普遍精确或完整 M1 成功，更不能从组件身份推出全模型泛化。
