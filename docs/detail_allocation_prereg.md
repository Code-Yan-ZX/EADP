# 固定256-token预算的单区域/分离双区域诊断（2026-11-14）

## 目的与边界

用户授权在R-res强参考基础上思考下一步并实际跑实验。已有13集结果不支持把当前学生直接当作R-res的通用修复。下一步仅检验：在固定全局覆盖、细节面积、候选数量和空间覆盖机会后，两个分离细节区域是否有单个连续区域无法替代的质量空间。先诊断，不训练新网络，不宣称新方法或新颖性。

这不是实际crop编码、降分辨率改进或可部署路由。所有几何动作先用完整1024-token原生视觉编码获得V及三条DeepStack，再选择256个原生位置。各token已含视觉encoder的全局上下文；不能从选区结果推断原图中证据的严格因果局域性，也不能继承R-res的低分辨率编码速度。Q-Zoom、GapSight、Foveated Compression已有相关区域/收益监督，本轮不将其普通组合重新命名为创新。

## 数据与冻结

InfoVQA TRAIN和TextVQA TRAIN各192主图+2开发图，共388独立原图，每图1道原始人工问题。只用各源第一个TRAIN parquet分片：InfoVQA固定HF revision 539088ef8a8ada01ac8e2e6d4e372586748a265e，首片998题/211图；TextVQA固定本地快照首片1731题/1116图。因此这是两个首分片的小面板诊断，不是全TRAIN均匀抽样或官方benchmark。按图像输入身份的稳定hash选图、再按原题身份hash选题，不查看已有模型预测、不按full成功/R-res失败挑图。TextVQA保留全部10个人工答案票，InfoVQA保留原别名。

排除当前学生全部217页fit/val/test/dev及旧13集的原/物化图片字节、源RGB像素和dHash近重复。不能据此保证底座预训练未见。OCR、答案、类别和历史模型输出不进入生成或固定选区；输入白名单仅dataset/item_id/image_path/image_sha256/question。

正式freeze和任何GPU前，将CPU初稿的每集96主图扩为192，规则与开发图不变、主图嵌套。原因是旧TextVQA full−R-res仅约9pp，96图预期有差异案例可能不足预设12个，容易得到缺乏功效的HOLD；该样本量规划只参考已有集合的差距，没有使用新面板的任何模型结果。原CPU manifest备份为manifest_panel96_preflight.json，不删除历史记录。

主面板和开发图独立。开发图只用于接口门禁和计时，不按其问答效果调规则。QA前冻结数据、原图片、parquet来源、checkpoint、代码、动作池和本协议。复用原DocVQA TRAIN训练的bag_rank seed11/23/37，保持源validation选点、全部报告；不调整学习参数或问题长度保护。原生简短VQA提示统一，正常greedy/EOS、128新token上限、原问题不截断；InfoVQA用官方ANLS，TextVQA用原10票leave-one-out VQA consensus，截断保留主分母。

## 完全匹配的几何动作

32×32原生网格，固定64个全局锚：行、列均为0 mod4。一个8×13基本窗口有104个原位置，恰含8锚，故新增96。单区域由两个相邻基本窗口构成16×13连续区；双区域由两个不相交、存在空间间隔的相同窗口构成。两者都为G64+96+96=256，额外细节面积、密度、单元形状相同，无合并、补齐或重编码；加入精确转置方向。

每池12个动作。单区纵向A位于r=0/8/16、c=0/16；B位于r_A+8、同列。双区只对同一B名册作固定置换：ri→(ri+1)%3、c→16−c；横向为逐项转置。两池A/B边际以及逐token出现频数完全一致，24个mask互异；两池各覆盖988个位置，同样36个位置从未被池覆盖。故本轮只比较该有限池，不能称所有可能连续选区的上界，负结果也只否定本几何。

7基线：full1024、原生EADP B1、R-res512（真实低分辨率256视觉token）、均匀256原生格、三个冻结bag学生。加24个固定几何，共31个固定输入臂。所有V、DS和原始三维位置由同一个递增原生keep_idx选择，保留fresh KV及原生decode位置。

## Oracle与替换验证

每个固定候选的生成不看答案。生成全部12+12后，分别取原QA分数最大值作为有限池oracle；同分选action_id字典序最小。这个max使用标准答案，不是部署方法的效果，也不代表可学习性。

每道主问题都对QA最佳dual AB运行两组预固定同图donor替换，共6个额外控制：G+FA+FB（00）、G+A+FB（10）、G+FA+B（01）；G+A+B（11）已在dual池。每个donor与A/B及另一个donor的原始窗口不重叠，都恰新增96，六个实际mask互异。donor来自预注册70个合法几何单元，按固定顺序选择，不按答案挑。控制的实际mask在QA最佳dual选出后才决定，因此必须标记selector_answer_dependent=True、audit.answer_in_selection=True、selection_scope=offline_oracle_control；生成本身仍不接收答案。不能沿用旧helper硬编码的答案无关审计。每题共37输出，主QA总数384×37=14208。

主要连续终点为dual oracle−single oracle；每数据集独立父图配对bootstrap10000次，另报等数据集权重汇总（两集独立重采样）。不按test选择种子、动作池、donor或阈值。

严格候选要求同时满足：full分≥0.8、R-res<0.5、最佳single<0.5、最佳dual≥0.8，且两组donor的00/10/01全部<0.5。它只说明在这些原生特征/窗口与替换条件下的共同收益，不能直接证明语义多跳或原图因果必要性。全部原回答、分数、选区、位置hash、donor及失败/截断信息保留，供复核。

预注册研究资源决策（非自然真理或论文显著性标准）：若任一集full≥0.8且R-res<0.5不足12题，HOLD_INSUFFICIENT_GAP；否则，仅两集连续差都>0、等权汇总差≥3pp且95%CI下界>0、严格候选总数≥8且每集≥2，才GO_DIAGNOSIS_ONLY。其余STOP_THIS_GEOMETRY。GO也不自动启动训练；需先读具体案例及与现有工作的差别。

## 门禁、计时与运行

GPU先验证原生提示、stock全保留prefill/decode identity、真实32×32位置坐标、视觉与问题缓存边界、全部31固定臂共享/新鲜编码输出一致、factorial答案依赖审计、正常EOS、冻结参数。全部通过才开始正式记录。

4张开发图，每臂每cap预热2、3配对block，cap1/16；计时7基线和固定首个single/dual，共216次。每次fresh图像准备、视觉编码、选择、原生生成，停止在decoder同步完成后、诊断hash前；仅计时忽略EOS强制token数。它测固定mask成本，不计答案依赖oracle搜索或尚不存在的部署selector，不能把这点写成部署速度；原生高清臂包含完整1024-token ViT成本。

脚本先完成CPU数据/冻结，等待A40实际空闲至少24576MiB并且没有其他报告的compute进程，连续两次确认才加载8B；期间状态waiting_gpu，每30秒轮询，默认最长24小时。不结束ollama或用户其他进程。计时如发现其他compute进程也等待。逐条SQLite提交、同冻结可续跑；失败保留并退出，不静默删题/改预算。冻结后修改源码/配置要使用新的run目录并保留旧失败证据。

启动/同冻结续跑：`CUDA_VISIBLE_DEVICES=0 bash /media/disk2/YZX/research/EADP/Qwen_vl/scripts/detail_allocation/run.sh`。

默认run：`Qwen_vl/outputs/detail_allocation/run_20261114/`；状态status.json、后台日志run.log、最终evaluation/report.md与report.json。新代码在scripts/detail_allocation，不修改任何旧冻结代码、checkpoint或结果。
