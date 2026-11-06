# PriorTR 32 题诊断准备

`sample_001.json` 仅使用已验 E1 输入和 DEV 分组、按固定哈希选择 11/11/10 题；选择发生在 E4 DEV 结果之后，且 1 个 DEV 图像属于历史 E0 锚点，详见独立核验。样本不读标签或预测；32 题只用于实现、跨环境原生差异及资源预估，不能据此选出优胜方法。

计划以固定 PriorTR 环境依次运行 stock 原生、作者默认和 layer-3 K256 三臂，共 96 次自然生成。实际 CUDA 路径 `runtime.py` 和顺序/曝光日志合同 `panel_core.py` 尚无可执行入口或运行权限；真实身份、CPU 合成合同、可审执行链、同容器无 GPU 预检、完整验收和超时清理准备完并经独立审查后才可运行。前三个任务各预选第一题保留完整诊断数组，其他题保留摘要和实际执行追踪。保留 SDPA/transformers 5.2 与 EADP FA2/transformers 4.57 的差异，原生生成结果应逐题与已验 E4 K1024 端点比较；该 32 题没有已归档的跨后端原生首 logits，不能假定相等，详见 `comparison_amendment_001.json`；日志中的 instrumented wall time 不是性能基准。

`monitor_core002` 的 4 项合成测试验证手动模拟的基础顺序、生成曝光、失败留痕、EOS/K256 拒绝和单次类切换，并非实际模型验证。单题真实模型证据及局限见 `../preparation/smoke_001_verification.json` 和 `../../../e1_independent_review/priortr_run003_review.json`。预算文件为保守终止上限而非预计速度；实际 32 题运行完成后，才更新余下实验成本。
