# POPE固定16：q=.2→.3缓存机制对照

全部16个源锚点通过，16个q=.3变体均实际测量，0未测。局部聚合4/16、importance3/16、完整gain trace3/16发生数值变化；有序32-token选择、最终空间集合和派生FP16空间序列均0/16改变。最大差分别为3.255903720855713e-5、4.723668098449707e-5、0.00408935546875；共同选择历史始终相同。该固定面板没有提供扩大为全8910题q=.3准确率实验的依据，不能外推所有图像都无变化。

参数固定alpha.5/beta1/K32，唯一实参变化为原生entropy_keep_ratio从.2到.3，原floor/topk/min1规则实际将77个dense位置保留15→23。两臂保留原过滤、聚合、归一化、平滑、power和facility算术；没有改Eq6/Eq7/Eq10或切换ablation类。公开论文的q=.3结论来自LLaVA1.5-7B/K128五任务消融，未确认主表q，也没有q=.2消融行。

复用原始16图的真实FP32 global/dense77/visual similarities和FP16 projected features，在熵过滤前切入。原native encode锚点属于alpha1/beta1；本次目标alpha.5/beta1是同次运行保存的真实GPU source-helper结果，并非独立native encode或LLM prepare锚点。每图importance、有序32IDs、完整32步gain、最终mask及按原cache派生的FP16空间序列全部先逐字节匹配才执行变体。未加载模型、CLIP、FA2或LLM，未读GT或答案，未生成回答或计算准确率。最终序列是无损cache推导，不冒称实际捕获的LLM输入。

单次GPU worker使用原torch2.1.2+cu118/CUDA11.8环境，16个新NPZ共9,996,121字节。独立NumPy终端核验320个有限数组、全部字节锚点、32步argmax/唯一IDs/mask与空间序列、来源哈希及真实worker退出。根PTY76513实际返回0，controller2913858/ticks1333710845与worker2913862/ticks1333710874均在原boot身份下退出，GPU空。没有自动重试或新benchmark台账行。

|证据|SHA256|
|---|---|
|plan|`668b9a5f644eec4019cb24e7b368a076b558421678c9109ad7dc60da9b492a4d`|
|root allocation|`2a1a15c9df81ee636b235ce11a9ec5a26f3de786d8a624f97dbd995884c647e4`|
|独立终端核验|`3c092c8a26fd71cd75d49fb1ffd3728c3e456968216f6f1361efc4f817ba6a07`|
|root实际退出证明|`6803dea9866edcf463449372834e44db00467ee45d06145521fc83c7b1e6e52d`|

独立peer终审另以NumPy读取全部16缓存，80项source字节锚点及320有限数组、32步gain/mask/派生序列、根真实退出链全部通过；报告`peer_terminal_review.json` SHA `0acf481de241cea9765d50bcd06edeb82a07eaeb45ceb1abc18279ec80c26456`。q=.3实际23成员的前15项过滤值和顺序与source15成员逐字节相同；未保存逐文本聚合softmax权重，不从这些输出倒推确切舍入位置。

主台账仍11完整/0pending/0验证失败，七主要剪枝目标仍未复现，不把本机制阴性结果当整体目标完成。
