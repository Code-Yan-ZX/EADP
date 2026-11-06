# E4：预算—质量—真实成本的实现设计

状态：只读源码审计与 CPU 合同；尚无 E4 GPU 实现、测量或新准确率。必须先通过 E0 身份门及对应数据/输入冻结；E1 的完整同协议结果通过后才可复用其 K256 端点。先按计划用 32 题实际 smoke 测各臂耗时，冻结各阶段算力预算，不能由旧记录宣称当前性能。

## 已核实的入口与不可直接复用的部分

- 当前固定作者环境 `.../modeling_qwen3_vl.py:703–753`：视觉塔运行 `patch_embed`、位置、全部 ViT blocks；在 `deepstack_visual_indexes` 上逐个运行 `deepstack_merger_list`，最后运行主 `merger`。三个 DS merger 不在主干前馈链上，其输出被另存入列表。主视觉流仍依赖全部 ViT blocks；跳过 DS merger 不得跳过主干 block。
- 作者 `.../model_fixed_res.py:66–73,477–515` 的剪枝路径只从视觉返回值取主特征，随后调用实际 pruner 并进行完整生成。`.../model/pruner.py:250–320` 在 K<N 时计算 importance、相似度及原始 greedy；K≥N 直接返回完整特征，**不会执行 scorer 或 facility greedy**。K1024 因而是 EADP keep-all 路径，不能报告其为 K1024 greedy；它也不是自动等于原生 FULL 的证明，必须先逐项过身份门。
- `e0_identity/adapter.py` 当前只接受 256/1024，并对三路 DS 均做 index_select，不能直接处理 K128/512 或 merger 真跳过。不要改 E0/E1，另写 E4 适配并独立测试。`reproduction/20261102/qwen_k256_ds_performance/benchmark.py` 展示了旧的同步总时间和首轮语言模型 CUDA event 测法：后者不含视觉/selector，前者含完整 decode，均不是本方案要求的首 token 时间。旧试验16题、不同位置协议，不能替代本轮性能。
- 旧工作区 `EADP/Qwen_vl/scripts/m13/m13_common.py` 有遍历原生 vision blocks 并跳过各 merger 的先例，但属于旧 SDPA 栈，数值和性能不可直接移植。也不要把计算后置零称为 C-SKIP。

## E4-A：质量曲线

全三任务 DEV，每任务 128 个冻结图像组、每组 1 题，batch=1；K={128,256,512,1024} × DS={000,111}，保持 P3、模型、1024² 输入、对应任务 prompt、greedy、2048 上限。K256 两端仅在 E1 的 manifest、权重、输入、代码版本、参数、生成 token 和验收全部一致时复用；K1024 用上述 keep-all 路径，注明没有 selector，核相应 FULL/DS-off 端点。K128/512 不应拿 K256 集合截断冒充原算法，而应调用对应 K 的原作者 pruner。

每题预填充前按原始 full input IDs/grid 计算 P3 坐标，然后按全体文本位置与所选视觉空间 ID gather；decode 使用正确的 full delta + (1024−K)，人工固定前缀检查至少32步。每题记录 greedy 有序 ID（K1024 为明确的自然空间顺序及 `selection_mode=keep_all`）、排序 ID、主/DS 对应、有效视觉 mask、位置/cache、生成 token、EOS/截断；四预算不可混淆成同一 selector 成本。先固定 smoke 样本和容差，执行相同模型/selector、P3 和 DS identity；K256 必须与通过的 E1 端点闭合。若端点失败先修复，不能按分数挑路径。

## E4-B：C-ON / C-DROP / C-SKIP

固定 K256、同一输入/权重/selector/P3；三者视觉主干完整运行、每题独立执行选择和 LLM prefill。

| 模式 | 原 DS merger 实际执行 | LLM DS 注入 | 目的 |
|---|---:|---:|---|
| C-ON | 三路 | 三路且与选择对齐 | 强参照 |
| C-DROP | 三路 | 关闭，传 None | 拆分 DS 注入影响 |
| C-SKIP | 零路 | 关闭，传 None | 测 DS 专用 merger 可省部分 |

建议新建一次性 E4 adapter：对每个 DS merger 的 `forward` 临时替换为返回 `None` 的薄桩，而保留视觉塔原始 `forward`、所有视觉 blocks 和主 `merger`；原视觉返回 `(main, [None,None,None])`，作者 `unwrap_visual_output` 只取 main。单独的 E4 prefill 逻辑对 C-SKIP 不访问 DS 子集，对 C-DROP 则确实计算三路但不注入。先核模块原 forward 的调用数、视觉输出、异常恢复/卸载及返回结构；`nn.Module.__call__` 仍执行，薄桩 Python 成本也会计入实测。可另比较新 vision forward 的显式分支，但不能同时混合两种实现来归因差。设置装卸后严格恢复模块函数，异常亦恢复；不能直接继承 E0 当前会无条件 index_select 三路的 adapter。

正确性：至少在冻结锚逐项证明 C-DROP/C-SKIP 的主视觉特征、processor、selector 有序 ID、排序 ID、非视觉文本、P3、prefill input/mask、首步 logits 和自由输出 token 一致，记录预先确定的逐项容差及 argmax；同时证明 skip 原 DS merger 零执行、全部主干 blocks 和主 merger 调用一次。C-ON/C-DROP 的相同输入/选择同样核验。C-SKIP 与 C-DROP 的差才可解释为该实现下的 DS 分支计算差，不包含 scorer 算法改动。

## E4-C：时间与显存的测量边界

遵守本轮方案：每配置/任务/长度协议先 10 图预热；DEV 每任务固定64图，每图5次，配置按图分块随机交错并记录顺序。自然输出与固定32新 token 各 960 次/配置，不用固定长度输出评分。用单 A40 独占、单实例模型、batch=1；起止记录设备占用与温度、功率、时钟、GPU 显存及进程。为避免 allocator 历史，显存按配置独立进程，明确记录每进程加载、预热、reset 和初始基线；不同配置交错的单进程计时与隔离进程显存是两类测量，不把后者当同一个配对计时样本。

**TTFT** 从读取/预处理图片之前的 T0，到真实 greedy 第一个新 token 被选出并可在 CPU 读取的时刻，不把首轮 logits 完成时刻称作首 token。当前作者 transformers 4.57.1 `generation/utils.py:2492–2493,2830–2840` 在生成开始先 `streamer.put(input_ids.cpu())`，之后选 token 才 `streamer.put(next_tokens.cpu())`；hook 必须把第一次 prompt 回调单独标记/忽略，在第二次回调记录 host `perf_counter` 与 token ID，并与最终第一个实际生成 ID 核对。测量开始前做一次 CUDA 同步，首 token 到主机本身的同步包含在 TTFT 内。`streamer.put` 的 CPU 拷贝每次 decode 都同步，因此该流式轨迹产生的全程时间不得冒充未插桩端到端总耗时：同条件另以无 streamer 运行自然生成和固定32生成，对应输出 token 身份须验证，计算常规总耗时与输出速率。预填充首 logits 的 CUDA event 另列；不能将其与 TTFT 互换。**实际前缀调用次数、CPU 同步和硬件/包版本均须通过 smoke 验证。**

主计时不按模块同步或 clone/hash 大张量；将输入解析/预处理、视觉、merger、selector、prefill、decode 等分解改为单独运行的 profiler 或 CUDA event 诊断并披露额外开销，不把逐层同步的分项相加冒充无干预 wall time。记录图片缓存状态、文件 IO 边界、timer 时钟、首 token 到达、输出 token 数、EOS、decode 总耗时；显存记录 CUDA allocated/reserved 峰值、模型常驻基线、NVML 本进程显存（如能获得）及设备占用；NVML 采样峰值只代表采样观测，不保证捕获瞬时峰。KV 用独立审计路径统计各层 key/value 实际形状、dtype、存储字节与缓存生命周期，避免将重复视图重复计数。总显存和 KV/激活分开，不能把 token 减少直接写成总显存同比下降。

性能汇总按图像分块重采样，保留每图5次重复，报告逐任务配对的 median、P95 与区间；重复不是独立图。性能样本来自 DEV，供诊断及预算选择；候选冻结后另外在 FINAL 选图复验才可用于正式资源主比。任何 profiler、初始预热、非独占并发、长度不匹配或身份未通过的记录，都应标识且不能进入主表。

## 冻结与自动拒绝

在数据/代码/参数与 arm 顺序冻结后，对每个端点保存执行代码 SHA、权重 payload 身份、source/processor/manifest SHA、环境、题目、原始输出、构建选择与注入的证据。独立 runner 先核 E0 以及 E1 端点、source 路径与 SHA、准入清单和过程退出；按审计规则做 SHA 而不是使用图片名代替身份。`contract.py` 及 `test_contract.py` 仅验证 CPU 记录和反例，不证明 GPU 实现正确或计时无扰动。图表、bootstrap 与方向判断只在完整阶段终验后运行；近重复图已保守合组但不足以排除语义重复。
