# M3 有界 oracle：样本准备

`sample_001.json` 按计划中的 `SHA256("20261104|group_id")` 已有顺序，固定 DEV 的 DocVQA 64 组、TextVQA 32 组、AI2D 32 组；只连接已验收的 E1 输入身份，不读取参考答案或预测。样本在 E4 DEV 分数产生后冻结，但排序和组数由先前计划确定，未按分数、答案或模型输出选择。`monitor_freeze002/terminal.json` 记录真实退出 0。

`monitor_freeze001/` 保留第一次失败：误把已有 manifest 验证的状态字符串 `pass` 认作 `passed`，在写出样本前退出；修正并核上游监督退出后另起 `monitor_freeze002/`，没有覆盖失败记录。

当前仅冻结问题，不含真实梯度、teacher-forced 搜索、生成、得分或性能数据。128 张图和固定参考答案仅可用于计划规定的有界诊断，不能当可部署方法或学生确认；实际 oracle 之前仍需冻结具体实现、输入和资源方案，审查并运行小样，记下原始 GPU/失败与退出证据。

`proposals.py` 是 CPU 合同原型：对四路已有 utility 数组作百分位排序（同值时较小空间 ID 排名较高），从落选 768 个位置构建 MAIN、MULTI 和 RANDOM 各64候选；固定删除贪心顺序末尾 r，r=8/16 时每来源最多8个不重复提案，并在不足时三来源等量缩减。计划中未详细指定的 seed 字节格式在此明确为 UTF-8 管道分隔字段的 SHA256 前8字节、大端整数，随机数为 Python `random.Random`，未来实际运行必须记录 Python 版本、所有提案和源码 SHA。`monitor_proposal002/terminal.json` 是3项 CPU 合同测试的真实退出0；不证明 utility、模型 backward、teacher loss 或实际生成已实现。独立审查和协议冻结仍在实际搜索之前完成。

`teacher_contract.py` 和 `saliency.py` 是未接入真实权重的教师诊断原型：按 prompt 最后位置预测第一答案 token，依模型的 P3/rope delta 为后续答案前缀定位，使用参考 token 平均交叉熵；utility 在 float32 下计算 `mean(abs(feature * gradient))`。`monitor_all_cpu005/terminal.json` 记录当前 CPU 合同测试退出0：随机小型 Qwen3-VL 文本模型的主流及三路 DS 均有梯度，联合方向导数与有限差分一致，teacher-forced logits 与逐步缓存解码对齐。该测试不验证真实 8B 模型、真实图片/答案、GPU 数值、显存或速度；实际运行前仍须固定答案 tokenization、输入提取、数值容差和资源上限，且完成独立审查及真实模型 smoke。

`oracle_teacher.py` 另提供无梯度的提案 loss-only 与四路叶张量梯度两种路径；`compact_prompt` 显式用同一空间索引保留原位置并调整 rope delta。`cpu_validation001.json` 将当前源码在测试前后分别哈希，`monitor_cpu_validation001/terminal.json` 证明全部 CPU 合同实际退出0；其中含丢弃候选产生重复而需三来源同等缩减、压缩后解码位置不变、以及 loss-only 与梯度路径 loss 对照。此证明不代替独立源码审核或真实模型验证。

`protocol_001.json` 在读取任何 M3 参考答案之前冻结人工约定：Doc/Text 原样取第一条参考、AI2D 取正确选项字符串；使用 tokenizer 的 `encode(..., add_special_tokens=False)`，不额外加空格或 EOS，最多128 token，记录截断，空参考不换用其他答案。它也绑定固定样本、CPU 合同和逐来源相同提案预算，见 `monitor_protocol001/terminal.json`；原实验计划中未细化的 tokenization 与 seed 序列化在此明确。该协议仅为预先记录；实际权重、资源和运行门及独立审核尚须完成。
