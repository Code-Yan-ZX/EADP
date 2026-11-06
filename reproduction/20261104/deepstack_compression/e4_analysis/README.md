# E4 全 DEV 评分与配对质量曲线（准备完成，尚未运行真实数据）

`plan.json` 在正式结果可用前固定四个预算 × 两个 DS 条件、每任务128题、bootstrap 和六项比较。主要预设对比是 **K128-D1 − K256-D0** 与 **K256-D1 − K512-D0**；另外报告各预算 DS-on − DS-off。每任务分别输出八个均分、两条随预算变化的质量曲线及图像组 bootstrap 区间；这些是探索性 DEV 质量结果，不是推理性能曲线。10,000 次 percentile bootstrap，seed 20261104，整组重采样保留组内题目，区间未做多重比较校正。实际面板每任务为128个不同图像组。

`score_e4.py` 在解析任何预测或读取参考答案之前，必须取得同时匹配的全部证据：

1. 正式 `e4_panel_v2/panel_verify.py` 对 384 题、3072 个真实生成臂的 full proof；verifier 源码和该监督器 started/terminal、PID、命令及 proof 哈希一致，且 verifier 退出0、进程组为空。
2. 独立 panel reviewer 的完整 384/3072 通过 proof，绑定同一正式 proof、正式 verifier 启动与终止记录、独立 reviewer 源码，且该 reviewer 的监督器记录证明 worker exit0、进程组为空。两个 proof 均声明没有读取 GT 或评分。
3. 重算正式 proof 的输入文件哈希及 frozen 所绑定的完整 v2/相关 E4 运行源码哈希，并核对真实 launcher/container 退出、进程身份、容器已停止与 GPU 释放；再核已验证 E1 DEV 的输入/图像绑定。

`terminal.json` 由 supervisor 自己写出，反映被监督 worker 已退出及进程组状态；并非额外的 supervisor 自身退出证明。面板尚未完成或任一门缺失、版本变更、监督未完成时失败，不读取 GT。完成验收后，沿用 E1 评分器读取答案：只在字节层筛选冻结 DEV qid 后解析选中行；全 TSV 字节读取仅用于计算哈希，不解析其他行答案。整个分析不得读取 CONFIRM/FINAL 预测或解析其答案。

沿用 E1 的作者 `process_line`/匹配和独立逐题核对，包括每任务每配置独立复现 xlsxwriter→pandas 预测文件往返/NA 语义。任何作者与独立分数不一致均失败，空预测/截断保留分母。AI2D 是确定性本地精确匹配，不使用外部裁判。完整的 formal panel 仅留摘要，不能据此声称事后重建了完整高维张量。

真实评分入口在证据齐全后需要传入 `--run`、`--panel-proof`、`--verify-started`、`--verify-terminal`、`--run-started`、`--run-terminal`、`--e1-acceptance`、`--e1-started`、`--e1-terminal`、`--review-proof`、`--review-started`、`--review-terminal`、`--out`，并用独立的有界 CPU supervisor 运行，输出必须是不存在的新目录。当前没有 E4 成功 full panel / review 证明，故不可提前运行评分。

`cpu_check004/verification.json` 和 `monitor_cpu_check004/terminal.json` 记录 12 项合成 CPU 测试及实际退出；覆盖 v2 与旧版 verifier 混用、运行源码缺失及篡改、不完整/重复臂、缺少独立 review、未退出、GT 护栏、非 DEV 不可解码答案、作者 xlsx 语义、两项主要对比和图像组 bootstrap。`cpu_check001`、`cpu_check002`、`cpu_check003` 是较早版本的检查，保留不覆盖。
