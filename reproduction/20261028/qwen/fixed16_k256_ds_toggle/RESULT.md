# Qwen DocVQA K256 固定16题 DeepStack 同选择对照

2026-10-30，单个 Qwen3-VL-8B 实例，作者公开 EADP K256，α.5/β2/q.2、FA2/BF16。16 题先走原生路径关闭 DeepStack，再使用同一实际空间排序的 256 个选择位置，在首次 prefill 注入同次视觉塔三层 DeepStack 及相应 256 位置 visual mask。该开关修改了作者公开实现，是机制诊断，不是论文的已确认配置。

- 16/16 原生路径通过历史实际选择及视觉特征锚点、历史完整运行的回答文本锚点。旧完整生成未保存 token IDs 或 prefill，只能锚定文本。
- 两臂均实际执行作者选择器；原有视觉和非视觉 embedding、attention、生成参数、初始 prefill/位置等逐字节一致，新增字段是 DeepStack 和对应 visual mask。独立验证器检查真实保存的数组、三层同索引、历史缓存及 32 次 token 重解码。
- 16 题中 **11 题回答文本改变，11 题输出 token IDs 改变**；32/32 输出含 EOS，0 次达到2048新 token 上限。共同生成前缀内 34 个续接步骤的位置、cache position、rope delta 均一致；不声称干预后 KV 相同。
- 这 16 题未读取参考答案也未评分；回答变化不能判断是否正确或准确率提高，不能将 FULL 的增益外推到 K256。需要完整同实例配对才可测定此实现修改在完整数据上的分数差；即使分数接近论文，也不能证明论文原始配方。

生成容器退出0，独立 CPU verifier退出0，根 PTY 56009 实际退出0，原 controller/docker/模型容器进程均已退出、GPU 空闲。[独立逐题验证](execution/independent_verification.json) SHA-256 `f6950c11e9f189cdf0a882c30da7206c48e630f1c4c68928e3549c0a3a175f92`；[根退出证明](root_exit_proof.json) SHA-256 `ab4c2e905e6941d378eb4a6c71082b32ba9d02fd1df664b6ac023d3d33a1db7f`；冻结计划 `cdae9b69ee7089e5e138f11537e67df7f4032a3d6f33003f7feb051e06e2c61d`。

独立只读终审通过：复核 32 份新 NPZ 的 1664 个数组、16 个历史锚、两臂选择与注入、共同前缀和退出证明；复核未重跑模型或读取 GT。此前独立 CPU 验证器已重解码全部32条输出。
