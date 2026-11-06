# PriorTR 单题 DEV 实机 smoke：准入已审查，等待 GPU

固定 `smoke_single_plan002.json`：一题 DEV，无评分；依次 stock 自然/强制、patched-disabled 自然、作者默认自然、显式 K256 自然、方法开启 keep-all 自然、K256 强制。最初的 `smoke_single_plan.json` 因没有区分 stock/禁用补丁而保留但作废。

`processor_check002.json` 已在独立环境对预先固定的8题逐项比较 E1 实际送模 bfloat16 processor 张量，全部一致；`processor_check001.json` 保存先前 float32 与 bfloat16 比较口径错误导致的失败。两者监督器分别保留。

`smoke_worker.py`、`trace.py`、`smoke_launch.py` 和 `smoke_verify.py` 是已做 CPU 合同检查、待独立源码审查的草稿；launcher/worker 均需要外部独立 review proof 才能进入 GPU。`cpu_validation004.json` 绑定精确源码、计划、输入端证明、隔离环境指纹与 3 个 CPU 测试；`monitor_cpu_validation004/{started,terminal}.json` 记录实际监督器启动和退出。有效 proof 必须绑定这些实际 SHA 与环境指纹及来源，不能仅填 passed。未获真实 review proof 时严禁运行实机 smoke。

trace 在随机微型真实多模态模型上与无插桩路径逐步 logits 相同；验收记录层输入长度、三路实际 DS、剪枝前后位置/hidden 及选择数组、每步自动 mRoPE / 每层 KV、token/原始 logits。stock 和 patched-disabled 分开对照；keep-all 如有差异记录而不强行修补。logits 和完整高维数组只用于诊断，不作性能结果。每次生成结束后先 fsync 生成 journal，才清理 hook 或写数组。

未来正式执行必须由监督器监控真实 journal，附 Docker 内外硬时限、容器身份与终止证据；运行后独立验收及其监督器都真实退出后才能考虑后续评分。禁止 CONFIRM/FINAL 预测或答案；固定环境 `pip check` 的 decord wheel 标签问题和实际 CPU 导入通过均已原样记录，不能将其当作所有依赖无缺陷。

容器实查补充：`container_preflight001` 因 UID 1000 无 passwd 条目而在 PyTorch 导入失败，完整日志保留；launcher 现显式配置 `USER/LOGNAME=eadp` 以及数据盘上的 `TORCHINDUCTOR_CACHE_DIR` 等缓存。`container_preflight002` CPU 导入通过，但容器使用 `--rm`，无法事后 inspect，仅保留为部分证据。`container_preflight003` 使用完全相同的环境构建函数、保留容器并记录实际 image/argv、宿主 PID/start_ticks、退出 inspect，未请求 GPU，外层监督 worker exit0。当前源冻结在 `sources007/`；`cpu_validation007.json` 及对应监督终端记录新版 CPU 测试，`monitor_preflight_admit004` 验证真实 003 工件通过。`independent_review007.json` 与独立 reviewer 监督终端已通过；新的完整 `admit_review` CPU 联调正在进行。B 占用 GPU 的期间不得启动 PriorTR。
