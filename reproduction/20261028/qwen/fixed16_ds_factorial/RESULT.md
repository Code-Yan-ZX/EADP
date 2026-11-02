# Qwen 固定16：同路径 DeepStack 开关

使用当前 FA2/BF16、同一个 Qwen3-VL 8B 模型和既有16个固定 DocVQA 输入。先重新生成并逐字节核对 plain FULL 与 EADP K1024 的32个端点，全部通过；再在各自原生路径的首次 LLM prefill 处，仅关闭或开启 DeepStack 与视觉位置 mask，生成另外32次。未读取GT、未评分，K1024是未注册的无裁剪诊断。

- 原生 FULL 路径关闭 DeepStack：16题中 **2题**的输出 token 和原生解码回答变化，14题不变。
- EADP K1024 路径开启本次视觉塔实际产生的 DeepStack：16题中 **2题**的输出 token 和原生解码回答变化，14题不变。
- 两种开关的其他 prefill 参数、视觉特征和原始位置状态均与其各自端点逐字节一致；在相同生成前缀的对应续接步骤中，分别检查87和89步，position IDs、cache position、rope delta 均没有发现差异。分叉后的续接步骤不做同状态比较；DeepStack 可自然改变后续 KV 内容。
- 全部64次生成均有 EOS；3378个实际数组经CPU验证。原始 native 输出已保留，token切片经核验，回答没有独立tokenizer重解码。该16题面板无法计算精度方向或把全5349题差距分配给DeepStack。
- 两条原生路径的生成API和位置策略仍有其他差异，不能把该四组合称为完全隔离的位置因素因果实验。观察与注入增加显存、同步和时间开销；不作为效率测试。
- 根PTY71385实际退出0，controller/docker/模型容器原身份均退，GPU空。

根退出证明 `fac478dc842c870ae9f2f360441d5880146b425984c2f07d82a1e04087a851ee`；CPU验证 `115e4828cae49329f7dc7f83e7f7d5d882bd20c7b38b036191fcade73622209a`；计划 `255e5a7dc9d97de0c597e8220d2cd7e1c1babe54aa1f1b26d90f413d30c91e76`。

固定16 DeepStack开关独立只读终审通过：另一CPU进程exec81888 exit0，另解压64个新NPZ校验3378数组、32旧端点与真实mask/三层视觉DS、84冻结SHA及根退出链。plain关闭DS改变position2507、3510两题；EADP1024开启DS改变1504、2507两题。共享输出前缀内累计176个续接步骤position/cache/rope一致；3个变化案例首token即分歧，没有可比后续位置。全部EOS且未到2048上限；无GT、评分、模型、独立重解码。
