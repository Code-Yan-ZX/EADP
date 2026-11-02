# 固定16题：TextVQA question-only CLIP guidance 结果

32次原生生成及完整CPU终端门通过。**16/16题的crop quota、实际mask和有效greedy前缀改变，2/16题的输出token IDs及原生回答改变。** 未读GT、未评分，不能判断这两处变化是提高还是降低，更不能外推全5000或宣布复现论文。

全部16个source先通过旧native缓存和完整K64运行锚定，之后才开始question-only。配置固定为CLI64／名义320、alpha.5／beta1、greedy128、SDPA／FP16、vicuna_v1及原三wait。唯一输入假设是把CLIP `texts=` 从完整问句＋OCR改为`split('\nReference OCR token')[0]`，再按原规则删Answer suffix；LLM完整OCR prompt及实际input IDs均未改。

| 完整16题观察 | Source：含OCR guidance | Question-only guidance |
|---|---:|---:|
| 实际每题保留视觉token范围 | 317–318 | 316–319 |
| 16题实际保留token合计 | 5081 | 5074 |
| 输出末位为EOS2的题数 | 16 | 16 |
| 实际生成新token范围（含EOS、不含起始BOS） | 2–18 | 2–18 |
| 新NPZ压缩缓存字节 | 657,882,760 | 654,966,418 |
| 已核shape／dtype／SHA／有限性的数组 | 224 | 224 |

32个新NPZ合计 **1,312,849,178 bytes**，448个数组的有限性及元数据经worker与内置CPU终端门核验。本结果整理仅消费完整报告和32份case元数据，没有再次运行verify、模型、GPU或评分。新缓存保存真实FP32 CLIP `image_embeds`、原生text states、投影features及实际prepared embeddings，保留今后独立复核所需数据；没有补造原缓存缺失的FP32向量。

两处原生回答变化如下；表中只转录输出，不解释其对错：

| full位置 | question ID | Source回答 | Question-only回答 | 实际总视觉tokens |
|---|---|---|---|---|
| 820 | `ea4cd165454e3c18` | Winter | ALIVE | 318 → 318 |
| 3007 | `5b91bdbc8bf37fe4` | Ben hur | Lew wallace | 318 → 317 |

全部16题的source FP16 pixels／projected和FP32 global／dense／visualsim／beta1 importance逐bit匹配旧native16；source实际quota、mask、有效前缀、完整LLM IDs及输出IDs／文本逐项匹配完整K64对应题。两臂真实projected、CLIP image_embeds、visualsim和图像tensor逐bit相同；实际native prepare的非视觉嵌入前后缀和完整LLM IDs也保持。视觉内容／数量按原native选择变化，不强制补齐320。

这不同于旧valid-dense／bounded-chunks控制：那些控制保留OCR内容，本次仅改变CLIP guidance边界。作者源码第113行存在该inactive split注释，因此这是有来源支持的窄输入机制问题。论文没有唯一规定TextVQA应剥离OCR，且方法描述称CLIP编码同一prompt，故本结果仍是输入假设，不能叫已确认bug修复或paper argv。移除OCR也消除了这些输入的多segment暴露，不能仅凭本对照区分OCR语义与分段处理各自的贡献。

本次两臂均在128新token上限前实际以EOS2结束；这不沿用先前cached-mask单例“无EOS／达到max128”的结论。14题相同离散输出也不能排除logit差异。文本由worker原生tokenizer解码；独立CPU门核对IDs／source文本／records及BOS／EOS／长度，**没有另一次独立token重解码**。

根PTY76915实际返回0并closed；controller2837356/ticks1333382280、native2837368/ticks1333382361均退出，根确认GPU空。没有自动重试、放宽source门或新benchmark主行。七个剪枝目标仍未复现。

| 终端证据 | SHA256 |
|---|---|
| 冻结plan | `812c39f98bb238c0f8bf513c0b730fb85a41c5ee3603510f82b2db16592dfe52` |
| root实际退出proof | `272ae626ed2bc36b03ea69d38bb21c9f11a6cb9a82df4b0f87f297693a23e356` |
| 完整CPU独立报告 | `f9d9076891883ea98d8569d9907e43e056a34303211339fe8fd6728c8ee8479b` |
| release | `0bae00bcf12b649101e6528ac7e706f2b7ac7f5ae73f6a8898d732b070ca66f6` |
| native complete | `504ec2b84c3f54dd2e698fbeb2fb75d4fb84e3b9dc2f1e33aea8f97c572bdfb1` |

完整逐题mask增删、crop quota、输入及缓存SHA见[完整报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/question_only_guidance_panel/execution/independent_verification.json)。
