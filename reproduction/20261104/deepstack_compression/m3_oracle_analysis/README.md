# M3 oracle DEV 评分准备

此目录没有读取真实 M3 生成输出、真实参考或产生统计结论。`plan.json` 在 oracle 真实答案访问之前固定：E1 P3-D1 基线复用、MAIN/MULTI/RANDOM × r=8/16 六个新生成臂；全部 128 题、768 个新生成必须完整验收并由独立 reviewer 监督退出 0 后，才允许 scorer 解析预测或读取数据集参考。

`analysis_gate.py` 先验证分析源代码、受监督 CPU 合同和独立 reviewer 完成；`score_gate.py` 再绑定全量 oracle 正式证明及其监督退出、独立复审及监督退出、128 份问题 journal 的 SHA、实际容器和用户冻结计划。未来独立面板 review `review_m3oracle_panel.py` 须提供 `formal_proof_sha256`、`formal_started_sha256`、`formal_terminal_sha256`、`frozen_sha256`、`host_receipt_sha256` 及 128 题/768 条、无评分/无参考读取字段；当前这些未来结果均不存在，因而 scorer 的面板门保持关闭。

`score_oracle.py` 使用作者本机评分及独立逐题评分交叉核对；每个任务、每个臂重现 author 的 xlsxwriter→pandas 预测列处理；AI2D 采用作者本机固定解析，没有外部 judge。空答案和截断保留分母。`paired.py` 以图像组为单位配对 bootstrap 10,000 次、seed 20261104，分别报告整体和任务结果。每个预注册 MULTI 对照同时报告增益、95% 区间及区间上界是否严格小于 1pp；恰好 1pp 不标记排除。区间不作多重比较校正，整体只是按冻结样本组合的探索性结果。

合成测试 `cpu_check_006/verification.json` 和 `monitor_cpu006/{started,terminal}.json` 已通过，包含完整 128 题正向管线、空答、计数、区间边界和未完成门拒绝。独立源码审查为 `independent_review.json` 和 `e1_independent_review/monitor_m3oracle_analysis/{started,terminal}.json`；`analysis_approval.json` 只批准这套评分代码，不批准提前读取尚未完成的 oracle 结果或参考。
