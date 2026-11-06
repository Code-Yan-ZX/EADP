# E4-C 组件微基准准备

这里只包含独立组件 profiler 的执行原语和 CPU 测试，不加载 GPU、不声称性能结果。正式计划由 `../e4_c_timing/microbench_contract.py` 固定：32 个 DEV 图像、每图每组件 20 次，selector、三路 DS merger 与注入各计 640 次，共 1920 次；warmup 与测量分开。

GPU event 时长反映设备事件之间的区间，不包含全部主机开销，也不能替代端到端 TTFT；另记包含 profiler 边界同步的主机墙钟时间。selector 应覆盖作者完整 pruner（相似度、重要性和 greedy），不能仅测 greedy 内核。输入缓存限制为当前图像，`capture.py` 准备了捕获完整 pruner、三路 merger 输入和实际注入并在测量前重放比对的诊断 hook；已用 CPU 假模型测试，尚未验证真实模型。正式运行仍须绑定真实模型组件与输入身份、提供独立的固定 GPU launcher/终止证明，并通过轻量路径身份门，禁止据此启动 GPU。

`monitor_cpu004/terminal.json` 是包含组件重放、错误注入反例和 1920 次合成调度的 CPU 测试退出证明，无 GPU/真实性能数据。
