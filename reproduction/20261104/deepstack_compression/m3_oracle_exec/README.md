# M3 有界 oracle：CPU 准备（尚未批准执行）

当前目录包含缺省锁定的真实数据面板 launcher 和暂定任务资源计划，已通过源码/真实模型前容器路径的独立审查并绑定批准材料，但尚未运行真实 oracle，实际启动必须按统一 GPU 队列和三天截止执行。未打开真实参考答案，未运行真实 answer-dependent oracle 或其 GPU 生成。另一个目录 `m3_gpu_smoke` 是已完成并独立验收的单题人工 token 梯度可行性测试；其结果不代表真实答案 oracle 的质量。

原 `m3_oracle/protocol_001.json` 保留。`protocol_amendment_001.json` 在本目录接触真实参考答案之前记录：原 EADP 基线必须复用已验收的同题 E1 P3-D1；每题最多六次新增自然生成，128 题最多 768 次。复用之前需要同题一致的验收记录、完整输入、processor、实际作者贪心、主视觉和三路 DS、P3 位置与 mask、generation recipe，并在读取参考前运行实际 K256/DS111 prefill、与 E1 首 logits 的字节哈希严格相同；任何差异均记录并失败。第一条人工 token 测试中的 0.125 容差不能用于这个严格复用门。

核心代码：

- `e1_admission.py` 为未来运行核对 E1 正式及独立审查的监督进程退出、认可文件、权重清单和实际四分片 SHA；运行仍须有外层独立批准。`e1_baseline.py` 在给定已验收完整字节流后定位唯一同题输入和 P3-D1 行；`baseline.py` 检查当前模型实际输入、选点、DS、位置及 logits。准备阶段没有调用这些函数处理真实预测或真实参考。
- `extract.py` 按作者路径计算完整视觉、三路 DS、指令 embedding 和真实 K256 贪心顺序；`teacher_backend.py` 使用冻结 teacher 接口实际求四路梯度和逐子集 loss。`search.py` 固定 r=8/16、三个来源各等量提案，允许跨来源共享重复的 teacher loss 缓存，并按协议处理并列。
- `selection_adapter.py` 在作者贪心实际执行并核对后，将预定子集注入主视觉流，并沿用 E4 的 P3 和三个 DS 入口核对；每次生成前必须核对真实 processor 和四路完整特征。`generate_backend.py` 在真实 model.generate 返回时立即持久记录原始 token，再进行解码和其他检查。每个新生成臂在执行前也记录曝光意图；失败后不自动重试。
- `one_question.py` 在基线门通过之前不会调用参考答案 loader；`run_control.py` 提供合作式绝对截止和单题截止，但不能中断进行中的 GPU 调用，未来 launcher 必须另行实施容器和监督器硬超时、小时快照和停止确认。

最新源绑定 CPU 合同为 `cpu_check_009/verification.json` 与 `monitor_cpu009/{started,terminal}.json`；完整模型前路径由真实 `runtime_worker.py --preflight-only` 执行，证据 `preflight_007.json`、`monitor_preflight007/{started,terminal}.json`，最终独立审查 `independent_runtime_review.json`。此前 `preflight_006` 只验证导入，不作为运行批准。前检没有加载模型、初始化 CUDA、读标签或预测。真实容器运行期间发送 SIGTERM 的有界退出反例保留在 `preflight_interrupt002*`：最终被判失败、容器已停止。早期 `preflight_interrupt001` 因宿主导入路径错误在启动 Docker 前失败，已保留；初版 `preflight_001` 虽正常退出，但其异常中断处理不充分，不作为最终前检。

正式 oracle 启动前仍需按统一 GPU 队列核对显卡空闲，并确认距离用户三天截止有完整的 36 小时加余量；实际运行须逐题通过基线首 logits 身份门。评分准备在独立目录 `m3_oracle_analysis/`，且须在全量完成和独立终审之后执行。面板草稿为失败即停止且不自动恢复，但未经真实数据运行。因此此目录的测试与容器前检均不构成 GPU、真实标签或最终统计结果的放行。

## 预定启动与预算

按用户三天绝对截止 `1794065414`，固定外层 129780 秒、容器内 129600 秒；启动前必须满足 `time.time() + 129780 <= 1794065414`，即最晚北京时间 2026-11-06 11:27:14。当前 GPU 排队为 E2/E3 → E4-C → 满足条件时 PriorTR → M3，由总协调检查 GPU 空闲后才启动。预定命令（从 `deepstack_compression` 目录运行）：

```sh
python monitor/supervise.py --run-dir m3_oracle_exec/monitor_runtime001 --timeout 129780 --stale 2400 --progress /media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/m3_oracle_exec/runtime_001/panel/panel_events.jsonl --snapshot-seconds 3600 --poll-seconds 15 --grace-seconds 120 -- /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python m3_oracle_exec/runtime_launch.py --run-id 001
```

每小时写监督快照；单题合作式检查上限 1800 秒，40 分钟无 progress 时监督器终止，进程/容器另有硬超时。生成开始和实际 token 返回均持久化；失败不自动重试，不以不完整面板选择性评分。最终验收还须独立审查且不从缺失题中挑选结果。
