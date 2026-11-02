# POPE固定16：alpha1、beta1→2原生生成对照

全部16个源锚点通过，随后16个beta2变体完成。仅把实际native model.beta从1设为2，保持alpha1、q.2、K32和greedy128。两臂importance、完整gain trace、有序选择、最终mask及实际送入LLM的FP16视觉序列均16/16改变；quota全部保持32。完整LLM input IDs、图像/文本特征、相似度、像素与非视觉embeddings逐字节相同。原始回答及输出token IDs各1/16改变；未读GT或评分，不能判断改变是改善，也不能由面板外推8910分数。

原固定index6/question2167的输出变化见独立终端报告；全部32条输出均返回3IDs（BOS1+2新tokens），均含EOS2。这是真实原生prepare/generate结果，没有以cache推导代替实际LLM输入。实际视觉区逐byte等于projected features依最终mask的空间顺序；非视觉区按同一image marker重建对照。

源先完整运行16题并通过旧真实native1/1的projected/pixels/global/dense/visualsim/importance/selected32/mask，以及旧同AST GPU helper的完整gain trace，再允许beta2。历史8910的原文本在唯一(questionID,fullprompt)键上相同；旧运行未保存输出IDs或prepared embeddings，不声称拥有这些历史锚点或完整重建历史runtime。新两臂实际IDs与embeddings均保存。

原generate/prepare/encode/cross数学体不变；greedy仅在原mask赋值后、argmax前保存total_gain对象引用，逆AST恢复整个原函数，实际执行code SHA已记录。每轮total_gain新分配且不被后续修改；所有tensor内容读取/stack/保存均在generate返回后。这是已披露的观测代码，不是无观测的效率测量。每次生成先保存NPZ和未验证attempt，再检查锚点，全部输出保留，无自动重试。

32新NPZ共269,020,894字节、480有限数组。实际torch2.1.2+cu118/CUDA11.8、SDPA、FP16 LLM/projector及FP32 CLIP，7CPU门和独立实现审查通过。根PTY98439实际返回0，controller2949970/ticks1333861741、native2949974/ticks1333861772均已在原boot身份下退出，GPU空。

|证据|SHA256|
|---|---|
|plan|`404ee2ad79241f91b8c392e69b3958fe096727e84fe9a1eaa6a2cd1a9d3c6605`|
|root allocation|`49408e6a4449448e00941c5d20c7f7b825cc823ff47e6b423b76809c2324903d`|
|独立CPU终端核验|`2342e6b9cae6cb19f2056ace0f24fe618ea1254f93413a78f5f51c93c114069f`|
|root实际退出证明|`2c5c74592cdd58e44c754f994936033ce6315d56f10cb5c4c8f3766eb9d8a1f9`|
|release|`849b8422c5ae842c036ecda58d2457f26de6a05fb2928de94dce668d8b38adc7`|

这是一项公开POPE shell alpha1与论文beta>1/已发表beta2消融支持的固定机制假设，不是确认的主表argv或已知bug修复。它没有与历史alpha.5/beta2拼成同环境2×2。当前只另行CPU准备唯一同alpha1/beta2全8910候选；仍需本面板独立peer、候选冻结/根与独立审读及根新GPU分配，没有自动队列。主台账仍11条完整，七个主要剪枝目标仍未复现。

独立终端peer另外读取全部32个NPZ，144项source byte缓存门、16旧文本唯一键、480有限数组、实际32视觉空间序列和非视觉区、全部原始IDs及根实际退出链均通过；报告SHA `0a7356219219b4d126cfc92f0e79cc4c1902a5b6f5b8658d4a643415253a487b`。唯一变化输出为原index6/qid2167，Yes `[1,3869,2]` → No `[1,1939,2]`，仍未读GT，不据此判改善。完整单点候选的前序门已满足，尚无新GPU分配或完整分数。
