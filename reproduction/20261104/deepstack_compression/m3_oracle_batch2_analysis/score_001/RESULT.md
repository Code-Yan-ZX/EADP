# M3 oracle：第二批 DEV 探索

本批独立验证 128 题、768 条新增自然生成；逐题 896 个作者/独立分数与 16 个配对 bootstrap 对比经独立重算一致。综合 ALL 按本批 64 DocVQA / 32 TextVQA / 32 AI2D 题自然加权；它不表示任务均权或总体分布。

| 范围 | 对比 | 差值 pp [未校正 95% 区间] | 区间上界严格低于 1 pp |
|---|---|---:|---|
| ALL | MULTI-MAIN-r8 | -0.50 [-2.68, +1.50] | 否 |
| ALL | MULTI-RANDOM-r8 | +10.08 [+5.05, +15.46] | 否 |
| ALL | MULTI-MAIN-r16 | -1.00 [-3.98, +1.74] | 否 |
| ALL | MULTI-RANDOM-r16 | +11.90 [+6.95, +17.39] | 否 |
| DocVQA_VAL | MULTI-MAIN-r8 | -2.10 [-6.24, +0.92] | 是 |
| DocVQA_VAL | MULTI-RANDOM-r8 | +15.47 [+7.98, +24.10] | 否 |
| DocVQA_VAL | MULTI-MAIN-r16 | -3.09 [-8.07, +0.85] | 是 |
| DocVQA_VAL | MULTI-RANDOM-r16 | +17.86 [+9.73, +27.11] | 否 |
| TextVQA_VAL | MULTI-MAIN-r8 | +2.19 [+0.00, +6.56] | 否 |
| TextVQA_VAL | MULTI-RANDOM-r8 | +3.12 [-5.94, +12.81] | 否 |
| TextVQA_VAL | MULTI-MAIN-r16 | -0.94 [-2.81, +0.00] | 是 |
| TextVQA_VAL | MULTI-RANDOM-r16 | +5.63 [+0.00, +14.06] | 否 |
| AI2D_TEST | MULTI-MAIN-r8 | +0.00 [+0.00, +0.00] | 是 |
| AI2D_TEST | MULTI-RANDOM-r8 | +6.25 [+0.00, +15.62] | 否 |
| AI2D_TEST | MULTI-MAIN-r16 | +3.12 [+0.00, +9.38] | 否 |
| AI2D_TEST | MULTI-RANDOM-r16 | +6.25 [+0.00, +15.62] | 否 |

本批 ALL 的 MULTI−MAIN r8 为 −0.50 pp [−2.68, +1.50]，区间上界未低于 1 pp，仍未决。r16 第一批为 −0.31 [−1.36, +0.72]，第二批为 −1.00 [−3.98, +1.74]；两批单独报告，第一批阈值结果不推出第二批或其他设置等价。

第二批是在看过第一批 DEV 分数后安排的第二次探索；其分析计划在本批生成期间冻结，当时未读取本批预测或结果。以上区间未对多次查看、多任务和多对比校正，也不构成预注册合并分析。Oracle 搜索使用 DEV 参考答案，仅是答案依赖的诊断，不能作为无答案可部署方法的效果。

正式验收和独立 panel 证明分别在 `m3_oracle_batch2_verify/runtime_001_verification.json` 与 `e1_independent_review/m3oracle_batch2_panel_review.json`；逐题评分/统计独立重算 `e1_independent_review/m3batch2_scores_review.json`，SHA256 `53468f28e3ae6d31086e754ab8d7524c89175d118762d000ed58851200cca76a`，监督进程真实 exit 0。
