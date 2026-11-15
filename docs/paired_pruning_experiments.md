# 同页问题干预与多个充分选集：运行协议

状态：实现与 CPU 验证；真实 GPU 门禁、标签、训练和结果由启动脚本产生。实现完成不代表方法有效。

## 启动

在任意目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/paired_pruning/run_experiments.sh
```

脚本默认使用 `/home/dell/miniconda3/envs/qwen3vl_clean/bin/python`，依赖本机已有的 Transformers 4.57.6、PyTorch、PyArrow 和 Qwen3-VL-8B-Instruct 缓存，不自动安装或下载。`CUDA_VISIBLE_DEVICES` 选择物理卡，进程内部使用 `cuda:0`。用 `PYTHON=/path/to/python` 覆盖解释器。

只做 CPU 环境、依赖和本地文件预检：

```bash
bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/paired_pruning/run_experiments.sh --check-only
```

默认输出目录：`Qwen_vl/outputs/paired_pruning/run_20261111/`。同一启动命令会验证已有冻结产物并续跑；修改数据规模、种子、步数或源码后，应使用 `--run-dir /absolute/path/to/new_run`。不要修改运行中的源码。

后台启动例子：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/paired_pruning/run_experiments.sh > /media/disk2/YZX/research/EADP/paired_pruning_20261111.log 2>&1 &
```

状态见运行目录的 `status.json`；各阶段日志在 `logs/`；最终结果在 `evaluation/report.md`、`evaluation/report.json`。失败会保留错误和已有产物，脚本以非零状态退出。重复原命令即可继续；数据或源码不同会明确拒绝接续。各阶段也可用 `--only STAGE` 单独运行，参数必须与原协议一致。

## 数据与冻结

- 使用本地原始 DocVQA TRAIN 36 个 parquet 分片，保留原始图像字节和全部人工答案别名。排除 DocVQA VAL 父文档、历史实际使用的面板和旧学生 fit/val/test/dev。
- 默认 dev 1、fit 128、val 24、test 64 个独立父文档，每个父文档仅一页，每页 3 个不同原始问题，至少一个非数字问题；合计 217 页、651 问。拆分之间父文档、图像字节及精确 RGB 像素均隔离；不会因样本不足静默减少数量。
- 默认规模来自实际可用性检查：按历史 manifest 和旧 fit roster 排除后，注释层面有 253 个满足三题及非数字条件的独立父文档，217 页给像素去重保留余量。旧数据审计 roster 的 validation 标签仅是暂定候选分区：其审计明确未提取问题或答案、未看预测、未训练，因此不凭标签将其全体视为已用；实际用过的 validation 页仍通过对应实验 manifest 排除。
- 实现阶段已用真实原始数据完成一次仅 CPU 的足额检查：217 个独立父文档、651 问，候选选中阶段无额外 byte/RGB 重复；全体问题加提示最长 43 token，fit/val/dev 的 459 个首答案最长 24 token。test 答案未进入 teacher tokenization。按纯数字答案分类，fit/val/test/dev 分别 13/3/8/0 问，因此数字子组样本很少，不能把其区间或点估计单独推广。
- 当前不是官方独立测试，也不能排除基础模型预训练见过这些资料、版式相似或有损重编码重合。先前 DocVQA VAL 全量分数不与这里的 TRAIN-heldout 分数混表。
- 数据构建和解码 RGB 排重是 CPU/磁盘阶段。随后冻结源码、参数、清单、模型 metadata SHA256 及权重文件 size/mtime；权重审计没有逐字节哈希全部 16GB 权重。
- 先在独立 dev 页验证原生 dense prefill/decode、gold-prefix 接口、四路梯度、K256、三路 DeepStack、原始三维位置和 R-res 实际 token 数，再构建离线标签。

## 共用执行条件

Qwen3-VL-8B-Instruct；完整视觉输入 N1024，硬选择 K256。所有流使用同一组排序索引，走已有 `NativeEngine` 的 stock decoder，保持 DeepStack 和原生三维 mRoPE。统一仓库的 concise VQA 提示，greedy、正常 EOS，真实 QA 上限 128 新 token。过长问题或答案显式报错，不静默截断监督。

梯度 teacher 只在 fit/val 和独立 dev 使用原始第一答案别名，输出 magnitude V 为 query 训练目标。它是离线候选信号，不能称为部署方法、真实删除效应或严格上界。测试 scorer 输入只有四路视觉特征和问题 token embedding；真实答案仅用于选集冻结后的计分。

## 实验一：问题变化监督

三臂具有相同模型结构和参数数量，仅位置开关或训练目标不同：

1. `bag_rank`：轻量文本 Transformer 不加位置编码；其跨模态结果对问题 token 的共同置换不变。
2. `ordered_rank`：增加确定性的 sin/cos 文本位置，训练相同 teacher-anchor 目标。
3. `ordered_pair`：相同保序架构与 anchor，再加入同页不同问题之间的 token utility 差分。

问题 token 来自冻结底座 embedding；轻量 Transformer 与视觉→问题交互均由学生训练，未让全部视觉 token 先经过底座 LLM。utility 对 magnitude map 做逐题相对幅值、log 压缩和标准化，再监督同页题目间的变化。问题配对、页采样、更新数在三臂间匹配。该目标仍是 teacher proxy，必须看最终 QA；不声称已经实现人工核验的语义改写或图像编辑不变性。

## 实验二：多个完整成功选集

每个 fit/val 问题预先产生默认 12 个不同 K256 候选：magnitude/signed teacher、B1/B2、随机、均匀覆盖及多个确定性 Gumbel 扰动。候选顺序不依赖 QA 成败。每个候选实际生成回答并计算全部答案别名的 ANLS，同时记录第一答案的完整 NLL。

成功要求：dense 与候选均未截断，dense ANLS≥0.9，候选 ANLS≥max(0.9, dense ANLS−0.02)。原始问题不因 dense 失败被删除。没有成功候选时，三个训练臂统一回退到相同 teacher-anchor 监督，显式记录回退比例；这种回退不能计作成功选集监督。

三臂共享相同 4 个完整 mask 模式头与 gate：

1. `canonical`：监督成功候选中 NLL 最低的一个完整 mask。
2. `token_marginal`：监督全部成功 mask 的逐 token 均值。
3. `multimode`：以完整 mask 的 BCE 定义每头 compatibility，在当前有限候选 bank 上归一化，最大化成功 mask 总质量，并使用成功/失败 margin、成功 mask 的 best-head 与一对一匹配 anchor，保留相容的完整模式。每个模式头还拟合离它最近的一个完整成功 mask；gate 用各头重构完整成功 mask 的质量接受监督，避免成功模式少于头数、所有候选都成功时选到未学习的头。

这是有限候选 bank 上的 compatibility objective，不是对全部 N-choose-K 子集的精确概率模型。部署只执行一次轻量学生前向，gate argmax 选择一个模式，然后该模式稳定 TopK、恢复 raster 顺序；不运行多个 LLM 回答后择优。

## 训练、机制诊断和真实评估

默认每臂 seed 11/23/37，各 900 更新，AdamW、lr 0.001、weight decay 0.0001、每步 2 页、每 30 步验证、梯度裁剪 1。query pair 权重 0.25；bank temperature 0.1、margin 0.05。所有参数在运行前冻结，checkpoint 仅按对应 validation objective 最小值选点，平局取最早，不按测试挑 seed。

机制诊断只在 validation：本题 teacher 选集与同页另一题选集对照；找到 Jaccard≤0.8 的成功选集对，固定 K 交叉混合其独有成员，检查是否失败。全 fit/val 统计成功候选数量。这些诊断只报告，不据此删测试题或只留下有利 seed。

真实 test 对照包含 full、EADP facility B1、block8 B2、R-res512、random 和全部 18 个学生 checkpoint，默认 192 问×23 臂=4416 条真实生成。每条保存回答、生成 token、截断、实际选中索引、DS 长度与位置审计。主指标是官方约定的 ANLS，同时报告严格字符串准确率、数字/非数字子组、修复与破坏、父文档配对 bootstrap 95% 区间，以及所有种子的均值和范围。任一缺失输出均不缩小主分母。

dev 三题上每臂默认 3 预热、5 个配对 block，分别计 fresh 图像预处理→ViT→选择→首 token 的 TTFT，以及忽略 EOS 的固定 16-token 总耗时；单请求不复用视觉缓存，模型加载不计入。计时终点在 stock decode 同步完成后、额外位置/DS 审计和哈希前；后置审计开销另行记录。该小 dev 计时不是跨页、官方 benchmark 或 serving 吞吐结论。

CPU 测试、GPU 门禁、缓存、机制诊断、两组训练、完整 QA/计时、汇总依次执行。CPU 测试不加载 8B 模型；正式 GPU 结果必须在启动后才会产生。缓存、候选逐项与训练状态及 QA 逐题均可续跑，已完成且被冻结的训练摘要保持字节不变。

实现交付时 52 项 CPU 测试通过，包含两个训练实验的合成端到端、checkpoint 事务恢复、完整评估/计时/报告与再次续跑；本机环境与模型注册/原生 DeepStack 接口检查通过。真实 GPU 门禁和结果尚未运行。首次 CPU 真实数据检查的旧串行路径约 17 分钟；正式脚本使用流式 Parquet 和 4 线程、有界队列的历史 RGB 哈希，并打印进度。后续续跑核验已冻结清单和源文件，不再扫描全部历史 RGB。

## 相关工作边界

本次不把 query-aware、条件集合效用、真实损失搜索或多充分集存在性本身当作新发现。候选差异是同页问题变化监督，以及保留多个相容成功模式的整集蒸馏。需结合 [EvoComp](https://arxiv.org/abs/2604.17087)、[Visual Sufficiency Landscapes](https://arxiv.org/abs/2610.24968)、[CoRePrune](https://arxiv.org/abs/2609.26484) 和其他直接相关工作判断最终贡献。只有真实 held-out QA 与成本支持，才能推进方法有效性的主张。
