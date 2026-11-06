# E2 / E3：CPU 干预合同

这里只准备并测试三个已选中、已按空间顺序对齐的 `[256, hidden]` DeepStack 张量的干预语义。它不是模型 runner，没有调用 GPU、读取问题答案或计算精度；E1、E4 的质量和成本结果及所有准入核验完成之前，不启动 E2/E3 GPU 生成。

- E2 的 `e2_slots` 以 LLM 注入槽顺序解释 `000` 到 `111`，始终返回长度为 3 的列表；关闭的槽位为原形状、dtype 和 device 的零张量，不缩短列表、不移动后续层。该干预保留完整视觉塔和三路 merger 计算，不能声称计算节约。
- E3 的 OFF 是三个零槽，须依赖 E0 证明其与完全不传入的端点相同；MATCH 不改变三个张量；PERM 对三个槽使用同一个不含不动点的行置换；MEAN 对每一路分别计算 256 行的均值并广播。主视觉流、原空间 ID、P3 位置和文本由未来的 runner 保持不变。
- PERM 的 image hash 是实际图像字节的小写 SHA256；种子是 UTF-8 字符串 `20261104|<image_sha256>|perm` 的 SHA256 前 8 字节、按大端解释的无符号整数。当前固定 Python `random.Random` / Sattolo 算法生成一条 256 元循环，确保无不动点且三层共用完整置换；这不是在全部 derangement 上均匀抽样。正式 manifest 必须记录 Python 版本、完整 256 ID、实现 SHA 及每层 multiset/norm 等摘要。额外的三个 CONFIRM 置换种子要在 CONFIRM 前另行冻结，不能从结果挑选。
- PERM/MEAN 的结果只能解释该干预，不能单独归因为 OCR 或局部像素机制。E2 的 `111`/`000` 端点必须与已通过身份检查的 E1/P3 记录闭合，E3 的 MATCH/OFF 可复用相同输入、模型和实际输出身份的端点；需保留主流、三个槽的身份、输出及真实退出证据。

`python -m unittest discover -s reproduction/20261104/deepstack_compression/e2_e3_contract -p 'test_*.py' -v` 为纯 CPU 合同检查：覆盖全部 8 种开关、零槽位置、预定 seed 及无不动点、三层共享置换、均值广播和非法输入。测试不证明真实模型 hook、生成端点或性能实现已完成。

`model_adapter.py` 补充了 E2/E3 的诊断 hook 原型：继承 E4 预算 adapter，在选中并排序后改变三路注入，并通过父类 observer 检查 LLM 实际收到的 mask、位置和张量。原始与干预后的选择流分别保留，PERM 记录完整行置换；所有视觉 merger 仍然运行。CPU 测试包括模拟父 prefill 的各干预及篡改检测、fake 模型上的完整 hook 注册与执行、传递依赖的导入污染反例；记录 `model_adapter_cpu002.json` 包含源码摘要。它们**不是**真实 Qwen 模型 smoke；任何 GPU 运行之前仍须固定源码与导入路径、独立审核，并对 000/111 及其他干预做真实模型端点和输出核验，不得凭这些单元测试直接跑全量。
