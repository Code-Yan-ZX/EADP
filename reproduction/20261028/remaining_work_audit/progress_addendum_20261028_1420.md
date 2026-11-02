# EADP 复现进展附录（2026-10-28 14:20 UTC 标记）

七个剪枝目标仍未复现。原审计的主要直接下一项已基本完成，新增结果没有恢复论文分数。本附录保留原冻结审计和旧 ledger，只汇总已接受的终端证据，不重新评分或运行模型。

目前没有找到同时具备“明确公开依据、尚未测试、足以支持修复实验”的新机制。NeXT CLI64/.5/beta1 全5000已于14:32UTC（22:32北京时间）由根另行分配A槽启动，待完整终端核验；它属于公开默认配方覆盖缺项，不能预称修复。用户没有额外作者材料；这不要求等待材料，也不能把公开源码执行成功改称论文复现成功。

## 七项目标

| 目标 | 论文 | 已接受本机结果（全部保留） | 当前仍缺什么 |
|---|---:|---|---|
| v15 MMB-EN K128 | 62.7 | .5/1=61.564626；.5/2=61.394558 | 剪枝行仍低；论文题集/行身份未公开绑定 |
| v15 POPE K32 | 86.7 | 1/1/wait=82.220237；.5/1/wait=82.706013；历史.5/2=84.073600 | 同步问题修复后残差仍在；不能据较高历史值挑配方 |
| NeXT TextVQA 名义640/CLI128 | 59.2 | .5/1新全量56.944；历史.5/2=57.866 | K128已经完成；公式个案尚无完整准确率效应 |
| NeXT TextVQA 名义320/CLI64 | 57.0 | 历史.5/2=55.410；.5/1已独立启动（待全量核验） | 公开默认中间预算的完整覆盖运行中，尚无合格新分数 |
| NeXT TextVQA 名义160/CLI32 | 54.2 | 历史.5/2=52.086；.5/1=50.966，均补核全5000 | 两份已合格，不应再记成“未跑”或盲目重复 |
| Qwen DocVQA K256 | 62.8 | 作者FA2/.5/2全5349：61.166042 | 当前公开路径已完整运行；论文实际特征/池化实现身份未闭合 |
| Qwen AI2D K256 | 75.7 | 作者FA2/.5/2全3088：74.352332（原生exact fallback） | 实际论文judge身份未明确，辅助解析值不能替换主分 |

以上是不同明确recipe的并列记录，不按最高值挑选，不补缺失平均分。当前本机新生成的完整剪枝行仍为旧 ledger 的9项；MMB FULL和三个公式面板不充当新的剪枝主行。

根已另建ledger_v2，旧9行序列化JSON完全一致、18个原raw/verify SHA复核通过；新K64只登记为pending，不读取新raw或评分前缀。该导出快照为9完整＋1pending，旧ledger的0pending仅是旧冻结快照。14:32UTC真实启动元数据为controller2734202/ticks1332948026、native2734206/ticks1332948051、rootexec63618、A槽；本附录只读计划/分配/claim/控制器元数据，未读新前缀。

## 原直接下一项的完成情况

- NeXT AnyRes真实16图诊断已完成320case、ragged0、16原生锚点、80identity。五个既有recipe全部无quota变化。Eq6仅beta2/CLI128在1图改变最终mask：full位置3632、crop3移除330/新增115，配额仍[146,162,106,130,93]共637；Eq10全部最终mask不变。实际中介个案成立，完整5000精度影响未测，不能把1/16外推成修复收益。
- MMB同协议FULL已完成4377行/1176组：757组正确，64.370748，低于paper FULL64.7约0.329252。相对beta1/beta2，FULL分别净恢复33/35组，剪枝损失2.806122/2.976190点；比论文报告损失2.0分别多0.806122/0.976190。此为数值分解，不是论文差距因果证明。
- MMB长短分层已完成：703长CLIP行涉及198组；978组所有旋转均≤77。beta1长组净损失4/短组29，beta2长组6/短组29。多数本地净损失在短组，削弱“长文本分段解释全部本地剪枝损失”的说法；但没有论文逐组结果，不能证明长组改法不可能补偿总分，也不指定唯一截断/分段修复。
- Qwen真实16题/16唯一图诊断已完成64case、ragged0、464有限数组、16原生锚点及identity。官方BF16 embedding实际token行逐位一致。Eq6在10题各追加1候选，最终importance、256有序ID和空间序列全部不变；Eq10每题importance有约9.60e-6至1.85e-5微扰，选择仍全部不变。
- Qwen候选新增权重的CPU参考约8.08e-19至3.70e-9，都是可表示的正常正FP32，不能称下溢或NaN。CPU镜像仅2题local有极小变化；原生GPU最终importance不变已由真实缓存核验。未保存的GPU中间量首次抵消位置不能由CPU参考推定。
- 两套当前真实processor完整5349输入比较已完成，1286图、图像及prompt差异均0。历史FULL Doc5349已资格核验94.483963，按论文精度同为94.5；它不是当时运行环境复刻或新FA2同环境因果基线。
- 旧固定显式标签规则已经原样用于新AI2D全部60未解析题：57条literal-label匹配、26个标签正确，辅助75.194301；余3全对的条件值75.291451。正式74.352332不变；条件规则不限制任意judge的可能结果。
- NeXT历史小预算15000逐题补核已完成，只有CLI64/.5/beta1公开默认完整覆盖正在运行、待终端核验；CLI128/.5/1及历史CLI32/.5/1已经完成。

## 已知公开命令与不能识别的部分

以下是作者仓库的命令模板，模型/数据占位路径须按已核身份替换；本机实际冻结命令及三wait/overlay修改均另有记录。不是本附录执行的命令。

| 配方入口 | 公开明确值与状态 | 无法由公开信息唯一识别 |
|---|---|---|
| `bash scripts/v1_5/eval/pope.sh 32` | 默认alpha1/beta1，greedy128，q实际硬编码.2；原码与wait全量均完成，单alpha.5也完成 | 主表究竟用shell1/1还是generic .5/1，主表q及实际mask/pred绑定 |
| `bash scripts/v1_5/eval/mmbench.sh 128` | 旧EN20230712，alpha.5/beta1，greedy1024；原码/wait/beta2/FULL均完成 | 论文仅MMB-EN标题不能唯一反推split SHA、外部服务器/judge身份 |
| `bash scripts/v1_6/eval/textvqa.sh 128/64/32` | alpha.5/beta1、greedy128、名义5×CLI，原floor/minimum-one quota不填尾数且不强制总数；128和32有完整结果，64已启动待全量 | 主表具体beta/q及逐crop mask/预测没有公开manifest；beta2历史文件名不等于paper argv |
| `MODEL_FAMILY=qwen3-8b TOKENS=256 DATASETS="DocVQA_VAL AI2D_TEST" EADP_ALPHA=.5 EADP_BETA=2.0 bash scripts/run_eadp.sh` | 两任务全量完成；固定1024视觉token、FA2、greedy2048；原LLM embedding mean及legacy输入路径已实际闭合 | 论文CLIP/EOS与源码LLM/mean适配的主表身份，AI2D实际judge/endpoint/cache；历史精确包/backend不可回填 |

公开执行性与论文行身份是两件事。不能再把已经完整执行且配置已知的本机命令笼统叫“argv未知”；同样不能从分数、消融最高格或文件名推定论文主表命令。

## 下一步判断与边界

1. 继续已启动的CLI64/.5/beta1全5000覆盖并等待完整终端门。它可回答公开默认预算曲线是否完整，不能保证恢复57.0；本附录不授予GPU或预记结果。
2. 暂无新的明确公开实现冲突，可直接转成有充分依据的修复实验。POPE/Qwen面板没有实际选集变化，不能再为“看分数”启动全量公式变体。NeXT只有一个真实选集个案；其后续生成效应是尚未回答的窄因果问题，不能自动升级为全量改法。
3. 论文把V定义成vision-encoder tokens，源码LLaVA在mm_projector之后构造facility kernel。这里存在符号解释空间，但未发现唯一公开的“应改成pre-projector”配方；两peer也未发现已审过的该控制。不能据符号简写宣布实现错误、替换空间或新排GPU。Qwen的LLM-space/mean适配同理：差异明确，唯一paper-faithful替代未公开。
4. AI2D实际judge、MMB精确split及各主表运行身份仍无法从已查公开材料唯一恢复。此为具体身份限制，并不证明所有误差均由未知recipe造成，也不要求等待用户无法提供的资料。若没有新增一手证据，应交付这些可重复负结果及边界，而非无限clone、重复hash、同recipe重跑或参数试配。

新ledger-v2代码已只读审查：只有完整5000、官方及独立逐题门、SHA互锁和根持有的实际exec退出证明同时成立，才可新增第10个完整剪枝行；pending不读新预测。未执行其builder或新validator。

同环境Qwen FULL可进一步收紧当前基线归因，但历史FULL已在论文显示精度上校准，尚无证据预言新FULL会恢复剪枝行，故不列为下一修复实验。既有padding/NaN/符号/tower、valid-dense/bounded-chunk阴性与原失败均保持。

完整路径、SHA、逐目标状态及证据见[结构化附录](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/remaining_work_audit/progress_addendum_20261028_1420.json)。原冻结审计和旧ledger保持原SHA。

## 后续终端补记：K64公开默认完整结果（15:00 UTC／23:00北京时间后）

本节追加在原快照之后。上文“运行中／pending”和九项新完整的描述保留其14:20标记、14:36定稿时的历史语义；当前状态由本节更新，不覆盖原冻结审计、原九行ledger或任何预测。

NeXT CLI64／名义320／alpha.5／beta1／三wait已完整5000：作者54.90000000000014，独立精确值54.9，论文57.0，差 **−2.1点**，仍未复现。全部5000输入、输出token重解码、实际quota/mask、逐题官方／独立分及完整分母门通过；保留数分布316:211、317:2291、318:2274、319:224，不强制恰好320。

根PTY63618实际返回0，controller2734202/ticks1332948026与native2734206/ticks1332948051均已退出，原PID不在/proc/GPU。根终端证明SHA为`100454d855cd853632a389693312cff5de9702fd5124a512a292e3ef3b16eae4`；独立report为`da9bc500921add4500a7d52063431b3df558e3d27a603c7016a60cf462762ede`。新raw SHA为`27ac3f1a2665b29262db2903961bff1f0fb4c29c206594b83d55dfe078de937c`。完整新证据路径、SHA与字节数列于JSON的`terminal_followups[0].evidence`。

已预审的终端消费者由根执行一次，只消费已接受逐题分，未重新评分。5000题精确配对结果为：

| 新beta1减历史参考 | 原回答改变 | 分数提高／降低／不变 | 精确净差 |
|---|---:|---:|---:|
| 同预算beta2（55.410） | 728 | 147／173／4680 | −0.51点 |
| FULL（60.370） | 1684 | 307／601／4092 | −5.47点 |

配对report SHA为`387049ae15e93ded44bb6e11ac3564cc4d1b32c6a6f56ec62cc6e25bb040464a`。历史wrapper、runtime观测和低层调度不同，这不是单beta因果实验。FULL逐题文件没有单列image／GT key，但已接受报告闭合同一canonical题集、GT和图像；按position／questionID／完整prompt SHA精确配对，不硬配缺失键，旧selected IDs缺失也不推断内容级因果。

本机FULL60.37→新54.9的剪枝损失为5.47点；论文FULL60.3→57.0为3.3点，额外2.17点只作数值分解，不能宣称由某一公开／未公开recipe造成。

原direct-next第6项现已完成：公开默认CLI128／64／32都有合格完整5000，其中32沿用已独立补核的历史beta1记录。当前该NeXT预算曲线没有剩余覆盖缺项；不把已完成的128／32再记pending。新ledger-v2为10完整／0pending／0failed，原九行序列化JSON保持一致，原ledger SHA仍为`cb3f29cc5197bebc735f7f99f0b9885c96208f05e960c5eafb0266f3571eed32`。K64正式JSON／CSV均为54.9／57.0／−2.1，当前查看表对应行一致。**七个剪枝目标仍全部未复现**，覆盖闭合不改变该结论。

根已另在空A槽启动既有changed-mask个案的source／Eq6两臂generation34b诊断（PTY89061），用途只是一个固定例子的下游因果判别，不读GT、不算accuracy或benchmark。本次补记未读取该运行的半成品，也不预判结果；没有新增GPU授权给本审计。K64负结果不支持继续参数网格、重复同recipe或直接全量公式变体。

## 后续终端补记：已知mask个案两臂完成（15:05 UTC／23:05北京时间后）

上一补记所登记的generation34b个案现已完整结束，经根真实退出证明及新的独立CPU逐bit审查通过，不再处于运行中。原先各快照、前一补记及原冻结工件不改；本节仅更新该个案终端状态，不增加benchmark主行。

根PTY89061实际exit0，controller2774786/ticks1333118631、native2774847/ticks1333118719的原/proc路径及GPU PID均已消失。根终端证明SHA为`77dca7df064b943cd61b309943800827c9dd1622362e7a2c126a69968c86a068`；原独立报告SHA为`637a1adf82a3757dbe8b14f6e927dc5c5d4b4a608facf0ebce14cbe638f8ef7a`；新peer审查SHA为`a5fe64f95e337624bb48c1f8115f5ac1293c38d1905689aaeb88046b2fe5a1f6`。

对固定3632题／原面板image11，source和Eq6两臂各637视觉token、quota同为[146,162,106,130,93]。独立按原源码base＋2×2 local crop空间顺序重建后，实际`inputs_embeds`各637视觉行与原生lossless FP16缓存逐bit一致；非视觉35＋212行也逐bit相同。crop3移除330／新增115导致83行视觉序列位置改变，证明干预确实到达模型实际输入。

**两臂返回129个ID和文本完全相同，但只成立于max128 horizon。** 模型config与generation_config均确认BOS1／EOS2；129包含起始BOS和128个新token，两臂均没有EOS2，不能称自然结束回答。未记录logits，不能排除其差异、后续更长生成的分歧或其他图片的效应。

原native feature／selection锚点是beta1／K128；本次beta2 source mask是同一缓存上的原source-helper replay，不是另一份beta2端到端提取锚点。此为一个事先已发现changed-mask的cached-input因果个案，不读GT／旧答案，不作accuracy或全5000外推。它没有提供全量Eq6准确率修复的正证据，不自动启动全量变体或延长生成。

独立审查未运行原verify、生成、评分或模型；12份source、9份冻结证据、20份model stat及SHA链通过。完整新证据在JSON的`terminal_followups[1]`和[个案结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/known_mask_case_generation/RESULT.md)。正式ledger仍为10完整benchmark运行／0pending／0failed；本个案不增加主行，七个剪枝目标仍未复现。
