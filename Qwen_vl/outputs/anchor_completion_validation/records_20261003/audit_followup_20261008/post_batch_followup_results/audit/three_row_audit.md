# 用户指定60.72生成配置的VizWiz结果

更新：2026-10-10T02:06:02+08:00。用户明确指定采用产生60.72的原始recipe，统一用于两个LLaVA模型全部三个预算。评分采用作者确认的EADP发布评分器。

|模型|名义预算|EADP reported|AnchorZip（指定配置）|
|---|---:|---:|---:|
|v15|128|58.00|57.85|
|v15|64|59.50|58.99|
|v15|32|59.30|59.76|
|next|640|60.60|60.72|
|next|320|60.40|60.08|
|next|160|60.20|59.74|

配置：alpha0.5/beta2、lambda.25、temperature0、单beam、vicuna_v1、完整文本guidance、max_new_tokens1024；使用已有4319题完整原预测（stream-wait修复前）。NeXT640/320/160对应K128/64/32，v1.5预算为128/64/32。

60.72来自旧NeXT K128同一答案文件：官方LOO59.254457→EADP发布评分60.716215。全六格均重新核验题ID/题面/模型名/GT/SHA，并逐项调用未改动EADP main验证。

该选择整体保留原recipe结果，包含低于后续控制结果的格；没有逐格取最大值。alpha0/beta1公开默认配对和统计保存在author_confirmed_vizwiz_scores_20261010.json及本审计JSON的supplementary字段，不充当原recipe的同协议EADP基线。

原recipe本机EADP VizWiz基线缺失；主表相应reproduced格置空，reported行保持论文值。其他数据集所有数值及Avg保持原值。旧预测缺 contemporaneous runtime metadata，不能把原recipe叫作完成stream-wait修复后的运行；作者确认仅覆盖评分器。

原75+后续5 GPU臂均已完成，本次没有启动GPU。未登记CN/逐crop与旧Qwen来源边界继续保留。

逐题评分与来源：/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/post_batch_followup_20261009/registered_followup/selected_vizwiz_recipe_6072_20261010.json。
