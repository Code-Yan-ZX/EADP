# 三锚低分辨率 K256 无 GT 身份诊断草案

本三锚来自已固定 Pixel NVML 九臂计划（DocVQA 16539、TextVQA 39421、AI2D 368100），不因答案、质量或新运行结果替换；它们此前已被观察，不代表总体。目标仅验证作者 fixed-res 路径中，局部将 1024 正方形 resize 改为 512 后真实 processor、视觉塔、三路 DeepStack 和 mRoPE 在自然生成中是否能稳定处理恰 256 个视觉 token。本轮不读取参考答案或评分，不从三题宣称低分辨率质量或速度，也不称与 EADP K256 等计算量。

使用完全相同 E1 图、原 message、作者 `_build_messages`、图像处理/expand2square 和生成 recipe；局部进程内只改变作者 `QWEN3_FIXED_RESOLUTION` 为 512 来调用原 `_processor_inputs`，finally 恢复为 1024，不改作者源码。原路径先按接受 E1 P3-D0 的四个 processor 张量及图像 span 验明；低分辨率输出必须实际 `image_grid_thw=[1,32,32]`、merge size 2、连续 256 个 image token、无 padding，删除不同长度的视觉占位符后完整非图像文本 token 前后相等。不得假设用户图本身为正方形：继续作者 expand2square 后 resize。实际 prefill 三轴位置与 delta 按低分辨率输入通过模型自身 `get_rope_index` 重算，不能沿用原 1024 的 P3；CPU 另独立按网格重建并检查。

GPU 前必须补齐冻结的生成 worker、受监督保留容器 launcher、实时权重/环境/输入身份、三路 DeepStack 与 prefill hooks/原生 KV/位置核验、generate 返回后原始 token 立即 fsync、失败日志和 stop→kill→inspect、上限/截止、formal 验收和独立审查；当前草案和处理器探针始终不授权 GPU。每题要求完整索引及 256 行的视觉主路和三路 DS、实际预填充 mask/position/delta、自然生成、已生成 token、逐步 cache 真实证据和 hooks 恢复；记录全部三题与未执行情况，不用 hook 成本作为性能。若 processor K不等256、前后文本不同、DS/位置/cache 不一致或任一控制失败，停止并保留原因，不作质量或计算等价结论。

先以CPU-only真实processor验证原E1+低分辨率四张量摘要及 3D mRoPE，无模型/GPU、无新生成；参考E1 mixed journal 仅用于身份，明确会解析既有DEV预测但不使用其值、不读GT。证明和源码固定后执行实际同镜像无GPU容器前检并独审；主控安排 GPU，不自行启动。
