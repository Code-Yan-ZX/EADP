# E6 固定随机基线：CPU 准备

`plan.json` 固定三个 seed、基于图像身份的种子序列化、无偏 Fisher–Yates、K256、P3 与三路 DS 对齐。seed 在已有 DEV 实验结果之后、随机基线产生任何预测之前冻结；这不应称为在已有 DEV 结果之前预注册。

`RandomAdapter` 复用现有 arbitrary-selection 诊断 adapter，先核同题 E1 的实际贪心索引，再把选中主特征替换为随机索引，沿用实际 P3 和三路 DS 注入观察。它仍运行 EADP 贪心选择器，不能用于性能比较。`test_selector.py` 与 `test_adapter.py` 是合成 CPU 验证，不是模型结果。

下一步须固定全部运行/依赖哈希及 seed manifest，使用真实容器无 GPU 完整预检；GPU 小样覆盖三个 seed、多个任务、原生/复用端点、32 步 forced prefix、逐层 cache/位置/实际 DS 与独立数组验收。完整面板和 GT 评分需独立门控，当前无 GPU、模型运行或评分许可。

首次未监督合成测试有一项失败：测试把 fake 输入长度误写成 1030，实际为 1029；已把预期文本尾部改为取真实输入长度。修正后的 `cpu_check_001` 六项通过，`monitor_cpu001/terminal.json` 记录真实退出；未曾调用真实模型。
