# R-res/EADP 计时独立审查交接

截至本记录，正式32图计时尚未启动或被我授权；GPU排程属于主控。三锚两臂真实GPU身份、正式验收与独审已经完成：`e1_independent_review/rres_eadp_identity3_gpu_result_review.json` 和对应 `monitor_rres_eadp_identity3_gpu_result` completed/0。6/6与各自参考的processor、原始输出、视觉/三DS、P3、36层KV、配置6标量/EOS通过。EADP先验无首prefill logits可比；高维张量未归档；诊断插桩不可用于时延。

`e1_independent_review/rres_eadp_timing_statistics_v2_prelaunch_review.json` 及对应监督 completed/0；`rres_eadp_timing_statistics_v2_supplement.json` 补绑独立测试辅助源码并重跑。仅覆盖纯统计/归一化窗口，未批准读取实际计时或运行GPU。独立合成检验覆盖三个任务×三个端点、图内5次中位、配对差/比、1万次相同图组抽样和线性分位。正式分析源码/读取门仍待执行方和state完成、冻结及独立复审。

窗口合同为请求 `[T0,response_end]`、观测 `[start,end)`；观测开始等于请求末端算重叠，观测结束等于请求开始不重叠。缺相同时钟/namespace或完整覆盖时未检测到重叠只能报unknown。`observation_oracle.py`、`test_observation_windows.py` 和 `monitor_observation_cpu002` 覆盖该合同；首fsync采用docker查询结束到fsync完成的保守包含窗，第二fsync和未观测后台仍unknown。

已反馈state两项最终分析草稿的实际接口阻断，未改其文件：`e6/rres_eadp_timing_analysis_proposal_v2/normalize.py::supervisor_health` 遇到现有 `monitor/supervise.py` 真实health记录缺单调窗口/namespace会抛错，应标unknown、保留完整分母；`report.py` 虽绑定自动health events和快照metadata哈希，却未读取/披露失败、超时、生命周期及可能不完整的窗口，需保守处理并补真实旧schema/失败事件反例。state最终读取门尚未冻结；不应将此草稿当通过。

已反馈experiment：现有v4每请求EADP adapter安装/清理处在主时间窗外；真实HF配置to_dict/递归描述、逐token停止回调在窗内，processor哈希/落盘可扰动下一请求。CPU草稿 `e6/rres_eadp_timing_execution/observation_contract.py::observer_windows` 存在跨sample时间逆序可通过的实证反例，要求追加有序性检查；当前执行层尚未冻结。三锚实测beam=1、返回数1、use_cache=true，应在正式验收逐请求检查，不能只看输出相同。`generation_control_oracle.py` 与 `monitor_generation_controls_cpu001` 为独立合成反例，不构成正式准入。

当前本代理无运行中的后台重任务，未发起新GPU/设备容器。其他agent状态一度出现模型不可用错误；未知主控后续排程，恢复前不假定GPU可用。已执行/旧冻结工件均保留，接续需先核当前主控状态、GPU及重I/O排程。
