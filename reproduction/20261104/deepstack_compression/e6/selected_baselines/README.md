# 本地 DivPrune/CDPruner 对照准备

这里使用 EADP 项目中固定的 Qwen 适配代码，并不把它称为两个原论文的原生骨干。`recover_indices.py` 仅在实际剪枝输出能与原始视觉特征唯一按位匹配时恢复索引；若存在重复/歧义、输出不匹配或非空间顺序则失败，不能自行猜测。`adapter.py` 不更改原 pruner，而是在其真实输出上按恢复索引构建 P3 位置及三路 DeepStack 输入；`baseline_runner.py` 是从已验证 E4 诊断核心改写的预备代码，无独立启动入口或准入权限。

`cpu_check_003` 的四项纯 CPU 测试覆盖原始 Div/CD 模块的小随机输入、BF16 位级索引恢复、歧义拒绝和合成 P3/三路对齐。这个证明冻结的是 `adapter.py` 与恢复辅助代码；新加的 `baseline_runner.py` 尚不在该证明中。以上不构成真实 8B 模型正确性或 benchmark 结果；还需独立审核、固定少量 DEV 身份小样、真实模型与原 wrapper 选择一致性、KV/位置/注入验证、可终止容器和完整独立验收。
