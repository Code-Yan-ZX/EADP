# E4-C 计时 CPU 合同准备（不是实际计时）

本目录独立于 `e4_b_identity/`，包含合成调度、轻量适配器、生成计时原语和 CPU 测试。参考门读取了先前已验收的 DEV 诊断记录；没有读取 GT 或执行本目录的真实 GPU 工作，也没有性能结果。

`make_schedule` 接收**已独立证明属于 DEV** 的384题（每任务128个不同图像组），对每任务按固定 SHA 顺序选择64个实测组及不重叠的10个预热组；输入包含非 DEV、重复题/图或数量不完整则拒绝。是否排除历史锚点由调用者显式传入，正式执行前应将其冻结并记录，不得根据答案挑选。对于自然与固定32新token两种协议，分别按图像/重复次数为块，以固定种子决定 C-ON/C-DROP/C-SKIP 的块内顺序。每配置每协议每任务10次预热，合计180；64图×5次×3任务×3配置×2协议，合计5760次实测。顺序和每条记录都有编号；预热不进入实测统计。

`validate_measurement` 只验证记录是否满足计时合同：T0 在图像读取之前；首 token 已选出并追加之后由 stopping criteria 回调进行一次设备同步并记时，TTFT 包含该边界同步；主计时不采集 logits 就绪时间，若以后单独使用 profiler 须另报；端到端至生成结束，纳秒转毫秒。固定32需恰好32个新token且不评分；自然生成记录实际长度。记录须声明只在边界同步、主计时没有 profiler/张量哈希/逐层同步、GPU 独占；主延迟三个配置在单模型进程交错，显存测量每配置独立进程；每题 allocated/reserved 与 KV 字节放在 memory_only；NVML 只作为完整单配置进程的采样峰值，不能当成逐题峰值。仅凭这些字段的布尔声明不能证明实际同步、真实 GPU 独占或进程隔离，真实执行须由计时实现、进程身份、设备日志与独立验收交叉证明。

当前已有模型调用和三处同步的计时原语，但真实 GPU 入口在导入 torch 前固定锁定；完整 GPU worker、实际时钟测量、真实性能曲线与轻量路径的真实模型身份验收尚不存在。可复用的 `execute_schedule` 按固定调度重放，并在每次生成结束立即落盘，异常时保留完成和失败轨迹；`input_preflight.py` 可先绑定已验收 E1 的题目和 processor 参考，且在计时外比对真实处理器张量与连续 1024 个视觉 token 的起点；它预读图像，故这一准备方式只能支持 warm-cache 口径，不能当 cold-cache。`memory_probe.py` 只在显存专用运行中记录 PyTorch 峰值和 KV 逻辑张量字节；`nvml_sampler.py` 单独采样宿主 PID 与 GPU UUID，检查 PID/start-ticks 并记录其他 GPU 进程。NVML 采样值以 MiB 取整且会漏掉瞬时峰值；真实进程启动器和采样器交接尚未闭合，当前不能声称显存数据已隔离。C-SKIP 与 C-DROP 的诊断路径已通过 B 身份门；轻量路径仍需单独与这些参考及 ON 比较，并冻结真实模型与工作负载，完善独立进程与随机交错的实际调度，确保不会因并存多个模型改变显存或吞吐。`microbench_contract.py` 已准备 32图×20次、分别针对 selector/三路 merger/注入的单独 profiler 调度（1920 次组件观测），尚未执行；还需实现真实 profiler、冷/热缓存、模型加载和潜在 batch4 协议，不能把诊断级 trace、逐层同步或微基准冒充主 TTFT。正式每阶段预算应由有界实测 smoke 更新。

`cpu_check054/verification.json` 与 `monitor_cpu_check054/terminal.json` 保留49项 CPU 测试及 worker exit0/进程组空；其中微型随机 GPT2 的真实 generate 核实 callback 在 token 追加后发生，另外用独立子进程测试单配置显存角色的隔离规则。`monitor_container_probe004/terminal.json` 是作者容器中的消息路径与导入探针，未初始化 CUDA。旧版本检查与失败工件均保留；该测试不证明真实 GPU 计时正确。

`light_reference001.json` 与 `monitor_light_reference001/terminal.json` 绑定先前通过验收的三个 DEV 锚点、八条 B 记录和四条 E4 ON 记录。计划比较16次诊断生成：4次新的 ON 诊断参考以及12次轻量 ON/DROP/SKIP；三个锚点自然生成，首个锚点另走32步固定前缀。旧 ON 的两个自然记录缺少首 logits，故新的 ON 参考先比较已有端点，再与轻量 ON 比较首 logits。`light_smoke_worker.py` 和 `light_smoke_launch.py` 已实现独立审核准入、超时与终止链；独立审核 proof 尚不存在时两者拒绝运行。上述比较尚未在真实模型上执行，不能声称轻量路径身份已通过。

`e1_gate_probe002.json` 和 `monitor_e1_gate_probe002/terminal.json` 记录真实 E1 DEV proof 与输入身份的正向 CPU 联调：384 题均绑定，通过前后源码及 proof 哈希检查，未读取 E4 预测、GT 或使用 GPU。隔离子进程避免项目中 `contract.py` 同名模块冲突。

`timing_supervise.py` 不修改共享 supervisor；记录包住完整健康快照的保守单调时钟窗口与 time namespace，`interference.py` 将重叠测次标记并保留全部记录，时钟或命名空间不匹配时拒绝比较。`monitor_wrapper_probe001` 为真实短进程监督探针，2 次快照和进程退出正常；主延迟过程不启用 NVML 轮询，NVML 属独立显存进程方案。

`light_preflight009` 保留真实作者容器无 GPU 的导入结果与 Docker 状态，`monitor_light_preflight009/terminal.json` 正常退出；环境报告 VLMEval `.env` 不存在，但前检通过。`cpu_check047` 有一项测试拒绝原因的断言错误，已保留；后续检查修正后通过。新的 ON 对照采集实际 KV 指纹（自然首步及强制33次），与轻量路径比较；历史 ON 参考未存 KV 指纹，不能宣称历史参考验证了这项内容。预算为内层3600秒、外层3720秒、进度停滞1200秒；2048 token 最慢题未实测，停滞策略可能误停，异常需保留。正式运行仍待独立复审和总调度。

独立审查指出此前 CPU 和容器前检对依赖缺项未 fail-closed；052/006 已要求完整覆盖36项准入源码/计划并逐项匹配，缺项和篡改反例均通过。正式独立复审仍在进行。

审查明确区分容器导入前检与真实 worker 前检：临时独立审核只能准入无 GPU 的完整模型加载前路径，包括实际参考、作者源码及四个权重 payload 的重算；最终审核须再绑定该独立容器与进程的真实退出。临时审核和 `light_preflight010` 的完整 CPU worker 前检均通过；最终独立 GPU 准入审核仍在进行，GPU 诊断未获放行。

独立复审发现旧前检 launcher 的 stop 异常可能遗留容器，现已增加 stop→kill→最终 inspect 和失败记录。`cpu_cleanup_probe001.json` 对真实无 GPU 休眠容器验证停止，`cpu_check054` 包含 stop 异常、信号与超时反例；旧 008 前检正常退出但使用旧 launcher，不作为新版故障分支证明。
