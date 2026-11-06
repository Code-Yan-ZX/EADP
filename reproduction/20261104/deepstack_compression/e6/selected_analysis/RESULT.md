# Div/CD DEV 配对质量结果

完整 DEV 面板共 384 题：DocVQA、TextVQA、AI2D 各 128 题、128 个互异图像组。每题四臂，共 1536 个自然生成回答。以下均值为任务指标百分数；配对差为适配臂减作者原始臂，单位为百分点。

| 任务 | Div 原始 | Div 适配 | CD 原始 | CD 适配 |
|---|---:|---:|---:|---:|
| DocVQA_VAL | 49.88 | 66.11 | 53.28 | 57.18 |
| TextVQA_VAL | 71.25 | 75.47 | 69.61 | 78.28 |
| AI2D_TEST | 66.41 | 77.34 | 69.53 | 75.78 |

| 任务 | 对比（适配 − 原始） | 配对差（百分点） | 95% 区间（百分点） |
|---|---|---:|---:|
| DocVQA_VAL | DIV | +16.23 | [+8.93, +23.69] |
| DocVQA_VAL | CD | +3.90 | [-0.78, +8.65] |
| TextVQA_VAL | DIV | +4.22 | [-2.11, +10.55] |
| TextVQA_VAL | CD | +8.67 | [+3.20, +14.45] |
| AI2D_TEST | DIV | +10.94 | [+3.91, +17.97] |
| AI2D_TEST | CD | +6.25 | [-1.56, +14.06] |

区间使用同题图像组配对 bootstrap，10,000 次重采样、固定种子 20261104；各对比和任务的区间均未进行多重比较校正，属于探索性 DEV 分析。DocVQA 使用 ANLS，TextVQA 使用共识分，AI2D 使用确定性的本地评分；作者及独立评分逐题一致。所有四臂各任务均为 128/128 EOS，空回答和截断均为 0。

作者原始路径为 P1、DeepStack 不注入；适配路径为 P3、DeepStack 注入，两种因素同时变化，因此以上差值不能拆分为单独位置或 DeepStack 的因果贡献。该面板评估质量而非计时；完整面板仅保存张量摘要，完整诊断张量来自先前 smoke。结果不用于宣称方法优胜或加速；尚未在 CONFIRM/FINAL 上验证。

证据（SHA256）：

- [预注册评分计划](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_analysis/plan.json): `f397d895e27b143e1dab1ada05f8f039e8913fcc7a651eacfac9cd13c6832b43`
- [score002 原始结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_analysis/score002/analysis.json): `771163be1ad8d70a9ac95c4c8a52bc86ae2f305537c50d8770999d8cdf08fddc`
- [score002 逐题评分](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_analysis/score002/scores.jsonl): `dc5075d85b415b8958a7093c27ffbc0dd049d9221d58feda529ed3c711603873`
- [score002 冻结身份](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_analysis/score002/frozen.json): `bb9db071d800fd2a39407a3d3dc28c561100c2032ceb9498bfdaea473959f9f5`
- [评分监督退出](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_analysis/monitor_score002/terminal.json): `a38325b4f935b6e50b49ed0d5fe570ac112ac3b97bf657f71a690a587a94f658`
- [面板正式证明](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e6/selected_full/panel_001_verification.json): `5b20e3f9dd1541994c89ddddff19a5f01a237b984cb772ad035a7c7b0b6677c7`
- [面板独立复核](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/selected_full_panel001_review.json): `8d54ca95f2e2d38ef340592c7d8b49765e6691563577cf147774547d1ccd7d0d`
- [评分独立复核](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/e6_selected_scores_review.json): `a76d68f1b429be2980a16bfc4fb98de4f232757f114420b0d9c5f7e195099b1a`
- [评分独立复核监督退出](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/monitor_e6_selected_scores001/terminal.json): `fc22d5d2f73adae7fc9ca3f0158d181d1f6f1036668cc93328c4f28bde1339e0`

第一次 score001 因 DEV 面板筛选合同错误在读取真值前失败；失败日志保留于 `monitor_score001/`。修复后 score002 经过新的合成测试和独立启动前审查。
