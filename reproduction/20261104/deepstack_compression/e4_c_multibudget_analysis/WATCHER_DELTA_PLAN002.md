# 背景观察的预先补充口径

在 32 图计时开始和任何该轮计时结果读取之前，对原 `ANALYSIS_PLAN.md` 增加强制观察链：冻结的 `e4_c_multibudget_environment/plan001.json` 及独立预审、来源和监督；实际 `monitor_watch001` 必须 completed/0/组空，存在 stop001、成对的 snapshot/logging_complete、最终 stop_marker watch_end 与监督 health 窗口。缺失、失败、截断、时间命名空间不一致均是未知覆盖并使分析拒绝，不可记为零重叠。保留完整原始journal的1520条、240预热及1280测量，不基于任意重叠删除样本。

对每个job的 TTFT窗口与生成完成窗口分别标注：原模型监督器 health 调用窗、watcher 的 ps 查询窗、从查询开始到第一条日志 fsync 结束窗、watcher 监督器 health 调用窗，以及已记录的手动检查有界窗。各自以同一 monotonic clock/time namespace 做区间相交，既保存原始索引亦作分任务/协议/预算的计数。显式保留每个job的phase/order、任务/qid/group/K/协议/repeat、三个原始ns时间、token数量/EOS/截断和全部窗口标记；输出不保存答案内容。现有原始被验收journal及其哈希仍保留为可追溯来源。

watcher 的第二条日志fsync、停止标记写入、未记录检查、主机背景进程的残留效应均未知；即使没有已知窗直接重叠也不称样本干净。ps的CPU是进程累计平均，不是瞬时利用率。观察者首个查询必须不晚于模型监督的启动时间，watch_end不早于模型监督真实结束，且在相同时间命名空间；UTC对齐受系统时钟变更影响须披露。若手动小时检查没有可用的单调起止或时间命名空间，标为未知而非补零；从监督器健康日志核查每小时健康快照或明确缺口，不得隐去样本。若超出冻结计划的观察/额外活动发生，以有界记录如实报告；本口径仍为事后探索、不改变既有统计主指标或确认性定位。
