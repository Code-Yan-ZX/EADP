# 有界实验监督器

`supervise.py` 仅启动和管理用户显式传入的一条命令；不查杀按名称匹配的其他任务。运行目录必须不存在，避免覆写已有实验。写入 `manifest.json`、`started.json`、worker stdout/stderr、`health.jsonl` 和 `terminal.json`。启动、每小时和终止时采集 GPU、内存、磁盘、受控进程组及指定进度文件元数据。时间戳为 UTC；间隔基于单调时钟。

使用示例（把路径、时间和命令替换为实际已经冻结的实验配置）：

```bash
python reproduction/20261104/deepstack_compression/monitor/supervise.py \
  --run-dir /media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/runs/example_monitor \
  --timeout 21600 --stale 1800 \
  --progress /media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/runs/example/predictions.jsonl \
  --disk /media/disk2 -- python -u your_worker.py
```

新进程以独立 session / process group 启动。超出硬时限或指定文件在 `--stale` 秒内没有 inode、大小或修改时间变化，向仍活着的受控进程组发送 TERM；leader 在 `--grace-seconds` 内未退出则发 KILL。首次进度检查从启动开始计时，加载模型长时无文件更新时须据实际启动与最长单题耗时设置足够宽的阈值，或者省略 `--stale` 仅启用硬超时；普通日志刷新的变化不代表有效题目进展，应监控实际逐题输出。可多次提供 `--progress`，其中任一变化会重置全局停滞计时。发生超时不会自动重试：检查完整性、确定可恢复协议后，另起独立 run 和日志。

每小时快照是监督器运行期间的设备/状态记录，不是研究人员重新分析或自动判断实验质量的保证。调度、机器停电、重启、监督器被 SIGKILL、驱动不可用等可能中断监督；应另配置对话级定时检查并查看机器。脚本尽力对收到的 INT/TERM/HUP 进行清理，但无法控制脱离进程组或 Docker daemon 管理的容器；尤其不要把 `docker run -d` 当作受控 worker：需要另行监督真实容器生命周期。若 worker leader 先退出但组内还有进程，监督器报告非空 `remaining_process_group` 并返回失败，不在 leader 退出后盲目向可能复用的组号发送信号。完成并不代表结果已通过实验验收，须另核题数、真实退出、输出与评分。

测试：`python -m unittest discover -s reproduction/20261104/deepstack_compression/monitor -p 'test_*.py' -v`，仅使用短 CPU 子进程。
