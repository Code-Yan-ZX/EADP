# Registered single-alpha POPE control

This control is prepared and not scheduled. It changes only alpha1→.5 against
the completed beta1/K32 wait-only run02. Canonical8910 questions, source weights,
greedy128 decoding, q.2, all source mathematics and the same three caller waits
remain fixed. It tests global/dense fusion contribution and does not identify
the paper's unpublished main-table argv. Every result is retained regardless
of score; no parameter retry is scheduled.

The historical survey found no alpha.5/beta1 POPE protocol among526 small
metadata candidates, including265 POPE-related files and135 parameter records.
It did not scan large per-question traces. The full baseline was independently
re-scored as85.87870689695657, rounding to paper85.9, on both preserved8910
prediction sets with zero answer differences.

`cpu_acceptance_checks.json` records10 successful CPU checks without a CUDA
context. The verifier requires all8910 ordered IDs/prompts, exact pinned
image/question alignment, loaded model/source/runtime identity, unchanged
frozen plan and payload/image stats, actual worker exit, and matching author
CLI/independent full metrics. Same-input consistency is reported regardless of
outcome. The worker only records model identity after native loading; it leaves
the native generation loop and generate method unwrapped.

Only after root explicitly allocates a slot, run the following manually; no
supervisor or automatic GPU queue has been created:

```sh
/home/dell/miniconda3/envs/llava_pruner/bin/python \
  /media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/alpha_control.py \
  run --gpu-authorized --allocation-note 'explicit root slot allocation'
```

For independent CPU verification after successful generation:

```sh
CUDA_VISIBLE_DEVICES='' /home/dell/miniconda3/envs/llava_pruner/bin/python \
  /media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/alpha_control.py verify
```
