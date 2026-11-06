# E4-C 主时延：DEV 单卡结果

在单张 A40、batch=1、单模型进程、热缓存下，按图像块交错测量 C-ON、C-DROP、C-SKIP。每任务 64 个不同 DEV 图像组，每配置和输出协议各重复 5 次，合计每格 320 次；三个任务、两种协议和三配置共 5760 次测量，另有 180 次预热。TTFT 从图片读取/预处理前开始，到首个 token 选出并追加后的同步边界；端到端到生成结束。自然输出和强制32 token 分开，后者不评分。

C-ON 使用三路 DeepStack；C-DROP 仍计算 merger 但不注入；C-SKIP 跳过三路 merger 并不注入。三者使用已验收的轻量路径。比较是本设置下的实际操作差异，不能把差值直接当作独立 kernel 的普适成本。

## 配置量值

| 任务 | 输出协议 | 指标 | ON 中位数 / P95 | DROP 中位数 / P95 | SKIP 中位数 / P95 |
|---|---|---|---:|---:|---:|
| DocVQA_VAL | natural | TTFT ms | 393.91 / 411.82 | 392.69 / 410.04 | 390.39 / 408.10 |
| DocVQA_VAL | natural | 端到端 ms | 614.01 / 925.54 | 610.55 / 921.51 | 609.98 / 915.54 |
| DocVQA_VAL | fixed32 | TTFT ms | 393.93 / 413.22 | 392.50 / 411.40 | 389.98 / 414.96 |
| DocVQA_VAL | fixed32 | 端到端 ms | 1777.62 / 1827.16 | 1777.08 / 1835.20 | 1776.41 / 1841.19 |
| TextVQA_VAL | natural | TTFT ms | 351.83 / 370.44 | 350.02 / 362.50 | 347.53 / 362.11 |
| TextVQA_VAL | natural | 端到端 ms | 442.47 / 611.15 | 441.67 / 606.49 | 438.65 / 608.10 |
| TextVQA_VAL | fixed32 | TTFT ms | 351.63 / 365.27 | 350.35 / 364.29 | 347.55 / 363.78 |
| TextVQA_VAL | fixed32 | 端到端 ms | 1736.76 / 1783.69 | 1735.62 / 1784.97 | 1732.07 / 1785.17 |
| AI2D_TEST | natural | TTFT ms | 345.83 / 384.34 | 345.62 / 383.71 | 343.20 / 381.14 |
| AI2D_TEST | natural | 端到端 ms | 397.97 / 537.08 | 395.57 / 519.49 | 394.94 / 520.87 |
| AI2D_TEST | fixed32 | TTFT ms | 346.74 / 384.56 | 345.56 / 382.81 | 342.87 / 380.61 |
| AI2D_TEST | fixed32 | 端到端 ms | 1732.74 / 1786.34 | 1734.44 / 1787.29 | 1727.74 / 1788.61 |

## 同图配对差与相对变化

以下每行都按同一 64 图像组重采样，保留所抽图的全部 5 次重复；差值是两配置各自 pooled 分位数之差，相对变化是两分位数的比值减一。括号为 10,000 次图像组 bootstrap 的未校正 95% percentile 区间，seed 20261104。百分比为负表示左侧配置该分位数更短；不能将这些探索性区间当作多重校正后的晋级结论。

### TTFT 中位数

| 任务 | 协议 | 比较 | 差值 ms [95% CI] | 相对变化 % [95% CI] |
|---|---|---|---:|---:|
| DocVQA_VAL | natural | C-DROP − C-ON | -1.22 [-2.32, -0.30] | -0.31% [-0.59%, -0.08%] |
| DocVQA_VAL | natural | C-SKIP − C-ON | -3.52 [-4.71, -2.58] | -0.89% [-1.19%, -0.66%] |
| DocVQA_VAL | natural | C-SKIP − C-DROP | -2.30 [-3.15, -1.53] | -0.59% [-0.80%, -0.39%] |
| DocVQA_VAL | fixed32 | C-DROP − C-ON | -1.43 [-1.87, -0.18] | -0.36% [-0.47%, -0.05%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-ON | -3.96 [-4.72, -2.73] | -1.00% [-1.20%, -0.69%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-DROP | -2.52 [-3.58, -1.90] | -0.64% [-0.91%, -0.48%] |
| TextVQA_VAL | natural | C-DROP − C-ON | -1.81 [-2.56, -0.74] | -0.52% [-0.73%, -0.21%] |
| TextVQA_VAL | natural | C-SKIP − C-ON | -4.31 [-5.06, -3.25] | -1.22% [-1.44%, -0.93%] |
| TextVQA_VAL | natural | C-SKIP − C-DROP | -2.49 [-3.36, -1.70] | -0.71% [-0.96%, -0.49%] |
| TextVQA_VAL | fixed32 | C-DROP − C-ON | -1.28 [-2.33, +0.02] | -0.37% [-0.66%, +0.00%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-ON | -4.08 [-4.71, -2.83] | -1.16% [-1.34%, -0.81%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-DROP | -2.80 [-3.84, -1.58] | -0.80% [-1.10%, -0.45%] |
| AI2D_TEST | natural | C-DROP − C-ON | -0.21 [-1.61, +0.47] | -0.06% [-0.47%, +0.14%] |
| AI2D_TEST | natural | C-SKIP − C-ON | -2.63 [-3.81, -0.96] | -0.76% [-1.10%, -0.27%] |
| AI2D_TEST | natural | C-SKIP − C-DROP | -2.42 [-3.46, -0.64] | -0.70% [-1.00%, -0.19%] |
| AI2D_TEST | fixed32 | C-DROP − C-ON | -1.19 [-2.14, +0.18] | -0.34% [-0.62%, +0.05%] |
| AI2D_TEST | fixed32 | C-SKIP − C-ON | -3.88 [-4.98, -2.78] | -1.12% [-1.43%, -0.80%] |
| AI2D_TEST | fixed32 | C-SKIP − C-DROP | -2.69 [-4.03, -1.65] | -0.78% [-1.17%, -0.48%] |

### TTFT P95

| 任务 | 协议 | 比较 | 差值 ms [95% CI] | 相对变化 % [95% CI] |
|---|---|---|---:|---:|
| DocVQA_VAL | natural | C-DROP − C-ON | -1.78 [-10.21, +5.64] | -0.43% [-2.40%, +1.37%] |
| DocVQA_VAL | natural | C-SKIP − C-ON | -3.72 [-10.50, +8.50] | -0.90% [-2.42%, +2.04%] |
| DocVQA_VAL | natural | C-SKIP − C-DROP | -1.94 [-6.23, +10.12] | -0.47% [-1.53%, +2.46%] |
| DocVQA_VAL | fixed32 | C-DROP − C-ON | -1.83 [-7.09, +6.68] | -0.44% [-1.69%, +1.60%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-ON | +1.73 [-7.81, +6.02] | +0.42% [-1.87%, +1.47%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-DROP | +3.56 [-8.78, +7.17] | +0.87% [-2.08%, +1.76%] |
| TextVQA_VAL | natural | C-DROP − C-ON | -7.93 [-11.90, -1.23] | -2.14% [-3.18%, -0.34%] |
| TextVQA_VAL | natural | C-SKIP − C-ON | -8.33 [-13.16, -0.81] | -2.25% [-3.55%, -0.22%] |
| TextVQA_VAL | natural | C-SKIP − C-DROP | -0.40 [-5.49, +4.98] | -0.11% [-1.51%, +1.37%] |
| TextVQA_VAL | fixed32 | C-DROP − C-ON | -0.98 [-6.00, +4.50] | -0.27% [-1.63%, +1.24%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-ON | -1.49 [-7.03, +3.15] | -0.41% [-1.91%, +0.87%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-DROP | -0.51 [-5.95, +3.96] | -0.14% [-1.62%, +1.10%] |
| AI2D_TEST | natural | C-DROP − C-ON | -0.62 [-4.80, +3.57] | -0.16% [-1.24%, +0.95%] |
| AI2D_TEST | natural | C-SKIP − C-ON | -3.20 [-6.52, +1.23] | -0.83% [-1.68%, +0.33%] |
| AI2D_TEST | natural | C-SKIP − C-DROP | -2.58 [-5.86, +1.27] | -0.67% [-1.53%, +0.34%] |
| AI2D_TEST | fixed32 | C-DROP − C-ON | -1.75 [-4.37, +3.49] | -0.46% [-1.14%, +0.92%] |
| AI2D_TEST | fixed32 | C-SKIP − C-ON | -3.95 [-6.54, +1.09] | -1.03% [-1.70%, +0.29%] |
| AI2D_TEST | fixed32 | C-SKIP − C-DROP | -2.20 [-5.91, +1.62] | -0.58% [-1.52%, +0.43%] |

### 端到端 中位数

| 任务 | 协议 | 比较 | 差值 ms [95% CI] | 相对变化 % [95% CI] |
|---|---|---|---:|---:|
| DocVQA_VAL | natural | C-DROP − C-ON | -3.46 [-44.68, +28.93] | -0.56% [-7.03%, +4.92%] |
| DocVQA_VAL | natural | C-SKIP − C-ON | -4.03 [-47.16, +25.15] | -0.66% [-7.47%, +4.33%] |
| DocVQA_VAL | natural | C-SKIP − C-DROP | -0.57 [-9.95, +4.36] | -0.09% [-1.53%, +0.72%] |
| DocVQA_VAL | fixed32 | C-DROP − C-ON | -0.54 [-2.68, +1.98] | -0.03% [-0.15%, +0.11%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-ON | -1.21 [-4.11, +0.99] | -0.07% [-0.23%, +0.06%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-DROP | -0.67 [-3.62, +1.06] | -0.04% [-0.20%, +0.06%] |
| TextVQA_VAL | natural | C-DROP − C-ON | -0.80 [-1.92, +1.10] | -0.18% [-0.44%, +0.25%] |
| TextVQA_VAL | natural | C-SKIP − C-ON | -3.83 [-5.02, -1.62] | -0.87% [-1.14%, -0.37%] |
| TextVQA_VAL | natural | C-SKIP − C-DROP | -3.03 [-4.37, -1.35] | -0.69% [-0.99%, -0.31%] |
| TextVQA_VAL | fixed32 | C-DROP − C-ON | -1.14 [-3.47, +1.94] | -0.07% [-0.20%, +0.11%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-ON | -4.69 [-7.13, -2.02] | -0.27% [-0.41%, -0.12%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-DROP | -3.55 [-5.95, -1.45] | -0.20% [-0.34%, -0.08%] |
| AI2D_TEST | natural | C-DROP − C-ON | -2.41 [-9.09, +0.49] | -0.60% [-2.23%, +0.12%] |
| AI2D_TEST | natural | C-SKIP − C-ON | -3.03 [-9.81, -0.02] | -0.76% [-2.40%, -0.01%] |
| AI2D_TEST | natural | C-SKIP − C-DROP | -0.63 [-3.28, +1.15] | -0.16% [-0.82%, +0.29%] |
| AI2D_TEST | fixed32 | C-DROP − C-ON | +1.70 [-1.62, +7.26] | +0.10% [-0.09%, +0.42%] |
| AI2D_TEST | fixed32 | C-SKIP − C-ON | -5.00 [-8.15, -1.61] | -0.29% [-0.47%, -0.09%] |
| AI2D_TEST | fixed32 | C-SKIP − C-DROP | -6.70 [-12.28, -2.99] | -0.39% [-0.70%, -0.17%] |

### 端到端 P95

| 任务 | 协议 | 比较 | 差值 ms [95% CI] | 相对变化 % [95% CI] |
|---|---|---|---:|---:|
| DocVQA_VAL | natural | C-DROP − C-ON | -4.02 [-95.83, +50.37] | -0.43% [-10.07%, +5.94%] |
| DocVQA_VAL | natural | C-SKIP − C-ON | -9.99 [-99.00, +46.42] | -1.08% [-10.64%, +5.65%] |
| DocVQA_VAL | natural | C-SKIP − C-DROP | -5.97 [-25.01, +15.08] | -0.65% [-2.75%, +1.75%] |
| DocVQA_VAL | fixed32 | C-DROP − C-ON | +8.04 [-5.46, +20.43] | +0.44% [-0.30%, +1.13%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-ON | +14.03 [-7.36, +32.19] | +0.77% [-0.40%, +1.76%] |
| DocVQA_VAL | fixed32 | C-SKIP − C-DROP | +6.00 [-14.85, +22.12] | +0.33% [-0.81%, +1.20%] |
| TextVQA_VAL | natural | C-DROP − C-ON | -4.66 [-26.66, +35.02] | -0.76% [-4.16%, +6.15%] |
| TextVQA_VAL | natural | C-SKIP − C-ON | -3.04 [-15.79, +14.68] | -0.50% [-2.85%, +2.66%] |
| TextVQA_VAL | natural | C-SKIP − C-DROP | +1.61 [-35.90, +18.15] | +0.27% [-6.00%, +3.00%] |
| TextVQA_VAL | fixed32 | C-DROP − C-ON | +1.28 [-13.15, +14.94] | +0.07% [-0.74%, +0.84%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-ON | +1.48 [-9.87, +8.92] | +0.08% [-0.55%, +0.50%] |
| TextVQA_VAL | fixed32 | C-SKIP − C-DROP | +0.20 [-13.56, +9.78] | +0.01% [-0.75%, +0.55%] |
| AI2D_TEST | natural | C-DROP − C-ON | -17.59 [-70.73, +29.66] | -3.28% [-12.21%, +6.11%] |
| AI2D_TEST | natural | C-SKIP − C-ON | -16.21 [-74.43, +28.43] | -3.02% [-12.96%, +5.81%] |
| AI2D_TEST | natural | C-SKIP − C-DROP | +1.38 [-14.29, +10.73] | +0.27% [-2.85%, +2.07%] |
| AI2D_TEST | fixed32 | C-DROP − C-ON | +0.95 [-8.19, +27.15] | +0.05% [-0.46%, +1.52%] |
| AI2D_TEST | fixed32 | C-SKIP − C-ON | +2.26 [-10.74, +14.99] | +0.13% [-0.60%, +0.84%] |
| AI2D_TEST | fixed32 | C-SKIP − C-DROP | +1.31 [-22.10, +10.06] | +0.07% [-1.22%, +0.56%] |

## 输出与限制

| 任务 | 自然输出 token 中位数（ON / DROP / SKIP） | 自然输出最大值（ON / DROP / SKIP） | 自然 EOS / 截断（各臂） |
|---|---:|---:|---|
| DocVQA_VAL | 6 / 6 / 6 | 15 / 15 / 15 | 320 / 0 / 320 / 0 / 320 / 0 |
| TextVQA_VAL | 3 / 3 / 3 | 16 / 12 / 12 | 320 / 0 / 320 / 0 / 320 / 0 |
| AI2D_TEST | 2 / 2 / 2 | 7 / 6 / 6 | 320 / 0 / 320 / 0 / 320 / 0 |

自然输出共 2880 次均 EOS、无截断；最长 16 token，不能评估长答案尾部。固定长度每次均为 32 token。自然端到端时长受各配置输出长度影响；固定32才是这里的长度控制实验。

本次为探索性 DEV、单机单轮；图像组 bootstrap 保留每组重复，但不能单独估计重复测量噪声、主机负载时间漂移、跨运行变异或设备泛化。中途已知先前 pilot 值，分析计划在本次运行中、查看本次完整时延前冻结，并非运行前预注册。不据此宣称 FINAL 晋级、batch4、冷缓存或显存结论；组件微基准和独立显存实验仍未完成。

正式监督器健康快照直接重叠 1 个测量；两个人工检查完整窗口与 4 个测量重叠，其中 NVML 子窗口与 1 个测量重叠。所有测次保留。约 07:03 的两次日志/status 扫描无精确单调时钟边界；运行期间还有轻量 stat/ps、小源码读取、备忘写入、远端查询及非项目 Java/容器重启，不能逐样本定位。健康日志尾部写入及窗口前后残余影响未被覆盖；无直接重叠不等于无干扰。

## 证据

- [预先冻结的运行计划](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_full/full_launch_plan001.json): `356dc2a0f35042a4fd730ace8aba2e47f9cb46a76ea418c32ddf3b12b62c5968`
- [运行中盲态统计计划](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_analysis/analysis_plan001.json): `83d617c735ca12332783b92350c0448106f34b2c0eb2bf6b374006d980c30ef1`
- [统计计划补充](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_analysis/analysis_plan001_supplement.json): `7fb05c058125f038c1303eb23f722147500c27535337dfb7ad46641d45402636`
- [正式验收](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_full/full_001_verification.json): `b5998317e25511250c595862d06499d44596783bef4712cdd6552078bae0733c`
- [完整运行独立复核](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/e4c_latency_full001_review.json): `fa1b3cacc431c0f02c2d6c3046b60c50f7b3f568f9a8609f032b06d419cf9f4d`
- [人工窗口逐条独立复核](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/e4c_latency_manual_overlap_review.json): `8bcfb82dcc5a8ca8fe406ccbfe1fcda22a578ba56053bc8a49437e1d3474168c`
- [统计全量结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_analysis/analysis001/analysis.json): `19760878a44e1eb16e9b82625ea826e0b88ea9de08c77117cf23c377e8e9183c`
- [全量逐任务记录](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_analysis/analysis001/records.jsonl): `b8edbdd183ebe1adac4f7a5b5e109f8e42ec0a63df735b1fd60d5b5b0e132f63`
- [统计监督退出](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e4_c_latency_analysis/monitor_analysis001/terminal.json): `93fb46904b5d60392d98354b68e48410ba4a5b0befa74d0341fdde08c7c8deed`
- [统计独立复算](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/e4c_latency_analysis001_review.json): `6f84fcee412e335b17e6224fd36470273401a6be8b5bfd94b5788fc5deb9a9bd`
- [统计独立复算监督退出](/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/e1_independent_review/monitor_e4c_latency_analysis001/terminal.json): `0eee1137f04e361ccdf714c3527f92d40c4798715384653d9add038c80ae0908`

统计启动计划使用绝对路径，实际执行从记录的工作目录使用等价相对路径；argv 非逐字相同，路径归一校验见 `analysis_launch_resolution001.json`，并已纳入独立复核。
