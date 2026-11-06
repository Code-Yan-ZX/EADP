# E0：原生路径身份与 K256 DeepStack 槽位检查

本结果只验证本机 Qwen3-VL-8B、固定 1024×1024 输入、BF16、FlashAttention 2、单 A40 的入口身份，不是下游精度或速度结论。固定 DocVQA 16 题没有读取参考答案或评分。

独立 CPU verifier 对完整 16 题通过技术身份门（`anchor16_verification.json`）：

- 原生 FULL 与适配 K1024/P3/DS111 的实际主视觉流、三路 DeepStack、prefill embedding、mask、三轴位置、首步 logits 和自然生成 token 在本面板逐字节一致。
- K256 中每题观察原作者 facility greedy 有序 ID 与空间排序 ID，重建 256 个主视觉行及三路 DeepStack 行；未丢失文本行，三轴坐标取未剪枝原位置。K256 DS 关闭和三槽置零的输出与 logits 一致。
- 每题以同一强制前缀检查 prefill 加 32 个 decode，核对实际每步三轴位置、rope delta、cache position 和 36 层 KV cache 前后长度；两项身份比较的原始 logits、argmax 和 token 一致。强制前缀用于可比 decode，不能将自然生成分叉后的不同上下文当作同条件比较。
- 数组补跑包含 80 次自然生成及 80 次强制前缀生成，额外的 K256 DS-on 臂用于实际索引和注入检查；自然生成 80/80 遇到 EOS，0/80 达到 2048 token 上限。所有高维主流、各 DS、prefill 和 logits 存在 160 个 NPZ；输入、索引及 cache 轨迹另有 JSON。`anchor16_evidence_manifest.json` 绑定 160 NPZ、160 arm JSON 和 32 个比较 JSON。

监督器 `monitor_array16/terminal.json` 记录 GPU 运行实际结束，launcher `array16_exit.json` 退出码 0、无超时且结束时计算 PID 为空；容器的原始宿主进程身份与运行中 argv/GPU PID 记录在 `array16_worker_midrun.json`。独立 verifier 的 `monitor_verify_array16/terminal.json` 也记录退出 0、进程组空。数组捕获及 CPU 拷贝有额外开销，本轮耗时不能用于性能比较。四片模型权重在本轮运行中及运行后 stat 匹配此前校验，运行后完整 SHA256 与固定 LFS 身份一致；没有在本次启动前重新完整哈希，不能声称有启动时即时哈希。

本轮暴露了预先定义 ANCHOR 与原冻结分区的重叠：16 个不同图像组中 DEV 1、CONFIRM 2、FINAL 13。锚点生成没有读 GT 或评分，但不能将相应图像组继续作为独立的本轮确认或正式终验数据。原 split 保留；`../data_manifest/anchor_exclusion/` 另存生成后制定的保守剔除清单和独立验证，CONFIRM 排除 2 组、FINAL 排除 13 组。技术身份通过并不代替 E1/E4 的实现、数据及统计门。

历史尝试 `smoke_001` 真实退出 1：旧 Div/CD 观察器拦截 `torch.sort`，而 EADP 使用 `Tensor.sort`。后续实例改为观察作者 `_greed_select_impl` 原返回、验证原选点重建真实视觉张量；原失败、日志和退出均保留。`smoke002`–`smoke004` 是逐步验证，不能当 16 题正式门。先前 `anchor16` 完整摘要运行没有保存高维数组，也单独保留；本报告的完整张量门以随后独立 `array16` 为准。
