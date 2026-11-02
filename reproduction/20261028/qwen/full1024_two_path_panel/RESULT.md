# Qwen Doc 固定 16 题：两条完整视觉 token 路径

在同一个 Qwen3-VL 8B、同一 FA2 环境、processor、权重和固定 16 题上，分别直接调用作者类的原生 FixedRes 多模态生成与 EADP `visual_token_num=1024` 绕过剪枝后的 `inputs_embeds` 生成。1024 未在作者 EADP 注册列表中，是显式构造的诊断配置，不是论文主表命令；每图原有 1024 token。两臂共实际生成 32 次，不读取 GT、不评分。

- 16/16 原生视觉特征、LLM prefill 输入 embedding 和 attention mask 两臂逐字节一致；EADP pruner 输入与输出逐字节一致，全部保留。新视觉特征与先前同环境固定 16 缓存全部一致。
- 16/16 实际 prefill position IDs 不同。原生路径 16/16 带 visual mask 和三组 DeepStack 输入，EADP 1024 路径 16/16 两者均未传入。
- 3/16 题的输出 token 序列和原生解码回答不同，其余 13/16 相同。该固定面板不读取标准答案，不能据此判断哪条路线准确率更高；输出文本未做独立解码复核。
- 32 次原生生成、保存的数组、实际 generate 输入和参数、路径输出 suffix 以及原进程退出均经 CPU 终端核验。observer 在原生边界克隆张量，增加显存和时间开销；实验对照整个入口路径，不能单独归因位置编码或 DeepStack。
- 容器、原 controller、docker client 和 native host init 已退出；根 PTY69304 退出码 0，GPU 空。完整5349题同环境校准尚未运行，论文复现目标仍未完成。

根退出证明 SHA `400669950aa064a2ce2f6de80c21af9cd0bc5451e75a867a002b9ab1a1632a25`；独立 CPU 验证 SHA `83b2e2fabbf8c946ab3ccbde0dc65294bc72d760c1b6f2acaf655f9f790501e7`；计划 SHA `5fbf6c3c4355d55a26fe2f2ff8395a47bb6d39e14ed35046214f6a0459060e0c`。

Qwen固定16双路径独立只读终审通过：独立逐一解压32份NPZ并核672数组、真实generate输入/输出suffix、30冻结SHA、根退出链，旧visual BF16锚16/16一致。视觉塔自身三层DeepStack输出两臂16/16一致；差别是实际LLM prefill仅原生臂接收DeepStack及visual mask。两臂共32条均含EOS、未达2048上限；回答只核原记录，未独立tokenizer重解码。无GT/评分，不归因单个机制。
