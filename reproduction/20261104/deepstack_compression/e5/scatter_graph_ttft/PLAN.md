# Scatter author-op graph: three-anchor model TTFT design

This is a fixed32, warm-cache, three-DEV-anchor diagnostic for Qwen3-VL-8B,
P3 positions, DeepStack on, K256. It uses the same fixed anchors and
randomized adjacent paired schedule as `graph_ttft_pilot_v3`: one warmup and
five measured pairs per task, 36 total jobs. The comparison is the unchanged
author greedy function versus a CUDA graph of the single-statement scatter
variant. The author file remains unchanged.

Before any GPU execution, require completed formal and independent evidence
that all 96 cases of the scatter 32-image component probe preserve the full
ordered selection and quota for K128/K256/K512. Recheck all source and evidence
hashes, archive hashes, actual supervised proof exits, E1 input/processor
identities, user deadline, and device exclusivity. No such component result
exists at preparation time; CPU/container preflight does not authorize GPU.

Each job records all 256 selected ordered indices, actual quota, complete fixed
32-token output and recipe. Any pairwise difference, missing/failed job, timeout
or hook restoration issue prevents an equal-output aggregate timing claim;
retain every observation and failure. The original selector is restored after
each arm. Model-level correctness requires later independent review, and this
small diagnostic cannot establish general identity or accuracy.

The first-token clock includes image loading, preprocessing, selector,
prefill and the synchronization after token selection/appending. Graph capture
setup is outside warm request times and is reported separately with PyTorch
allocated/reserved memory. These are not device-wide peak measurements; GPU
background and unwindowed host work may affect timings. A raw continuation is
fsynced only after generate returns, after the first-token boundary, so no
end-to-end generation latency claim is allowed. There is no GT/scoring.

GPU hard limit 3600 s, supervisor 3720 s, grace 90 s, progress stall 1200 s;
weight preflight and graph setup are outside the experimental measurement.
Three images and five measured pairs cannot establish a general speedup or
complete M1. Independent prelaunch and post-run reviews and root scheduling
are required.
