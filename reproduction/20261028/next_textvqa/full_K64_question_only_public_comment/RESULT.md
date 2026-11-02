# TextVQA 全5000：问句指导假设

独立精确分为 **54.958%**，原公开默认完整OCR指导为54.900%，变化 **+0.058个百分点**。论文名义320-token行为57.0，本次差 **−2.042个百分点**；该输入假设没有恢复论文结果。两份完整结果分别保留。

| 同K64/.5/1/三wait完整实验 | 本机 | 论文 | 本机−论文 |
|---|---:|---:|---:|
| 完整OCR用于CLIP指导 | 54.900 | 57.0 | −2.100 |
| 仅问句用于CLIP指导 | 54.958 | 57.0 | −2.042 |

唯一改动是在fresh作者`eval_model`中启用原inactive注释的`question.split('\nReference OCR token')[0]`，随后保留原answer-suffix移除。完整LLM OCR提示、实际LLM input IDs、5000题/3166图、CLI64/名义320、alpha.5/beta1、SDPA/FP16、greedy128与三wait保持。原始入口文件保留；实际执行的patched AST和函数code SHA另行绑定，逆变换恢复全部原入口AST。该公开注释支持一个输入假设，未确认论文主表argv。

全部5000实际guidance哈希、LLM IDs、crop quota/mask/有效greedy前缀、输出token重解码、作者与独立逐题共识分，以及源码/输入/权重stat门通过。作者原生值54.95800000000012，独立精确54.958，最大逐题尾差2.22e−16；fresh作者CLI显示5000题、54.96%。空答和失败占位均0；实际316/317/318/319视觉tokens分别225/2244/2308/223题，沿原floor/min1分配语义。4986条输出包含EOS2，14条在128新token范围内没有EOS，全部保留在5000分母；固定16题均EOS的观察没有外推为全量自然终止。

已审的complete-only配对脚本实际执行一次，消费两端已接受的5000逐题分，不再次归一化答案或评分。两端已记录的包、后端flags、loaded model/dtypes/source、wait、实际生成参数一致，全部有序question ID/full OCR prompt/image/LLM IDs配对通过。

| 全5000配对观察 | 题数 |
|---|---:|
| 原始回答改变 | 684 |
| 输出token IDs改变 | 689 |
| 分数提高 / 降低 / 不变 | 151 / 148 / 4701 |
| quota向量改变 | 4999 |
| 实际保留总数改变 | 2901 |
| 最终mask / 有效有序前缀改变 | 5000 / 5000 |

提高与降低的既有得分合计分别为1222和−1193个tenths，净29，折合全5000的+0.058点。预先按完整指导长度分层：881题>77的净24tenths，对全量贡献+0.048点；4119题≤77的净5tenths，贡献+0.010点。分层覆盖全部题并重构总和，未按结果选样。移除OCR同时改变文本语义和原长指导的分段暴露，本实验没有单独识别两者各自贡献。

root PTY60045实际返回0，controller2861700/ticks1333486787和native2861704/ticks1333486819的原boot身份均已退出，根确认GPU空。内置终端验证完成后才登记本次分数；独立终端peer另外读取完整5000记录、59项source/metadata SHA、20项strict model stat及3166图stat并核对已接受逐题分，通过，未调用模型或再次评分。v3台账现11完整/0pending/0验证失败，原10行与两代旧ledger/builder保持。七个主要剪枝目标仍未复现。

| 证据 | SHA256 |
|---|---|
| plan | `3a1bfff5c78c47d7a0676f471eadcbff89aa3b98e9407132ffeaece3dbdce7e2` |
| 原始5000回答 | `0166ddd7fbdc3f6a15e579538236dcefaee3a892b9447397ef6d6297eca142c5` |
| 独立终端报告 | `1f66d83e5e213db974d297d2bb402dc7283130dd17bbb3ad1583d81124af05ed` |
| root实际退出证明 | `a0e9c852029c96099021328d69d1cf8ab2f62120ef6b50268344c0b150c46925` |
| release | `86497d5be61e30412b5087bb5ad7923e3b0061c4531dd17c8eb6d2f10e05de9f` |
| 完整配对报告 | `419a3c91bc10e7d348547e14b69ca49fcc1bd8f250fec5c107c751adae682c6b` |

完整逐题变化和预定分层见[配对报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/question_only_completed_pairing/comparison_report.json)，所有原始与验证工件路径见[独立终端报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/full_K64_question_only_public_comment/execution/independent_verification.json)。本结果整理没有启动新的模型、GPU生成、评分或参数搜索。
