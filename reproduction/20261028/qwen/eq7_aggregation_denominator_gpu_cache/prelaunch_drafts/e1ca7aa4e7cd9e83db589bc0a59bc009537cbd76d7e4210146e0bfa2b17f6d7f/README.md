# Fixed16 Eq7 denominator GPU feature-cache replay (prepared only)

Two arms only: source aggregation denominator .010001, versus literal Eq7 .01.
The fixed original run02 sixteen questions use q=.2, alpha=.5, beta=2 and K256.
Source AST inverse checks prove this is the sole numerical change; Eq6 and
Eq10 remain unchanged. The frozen CPU experiment's stdlib AST helpers are reused.

The numerical graph starts at the **actual saved native FP32 global/dense cross
similarities and visual similarity**. Real BF16 projected features/text are read
as uint16 bit patterns and viewed as BF16, never numerically cast from uint16.
No encoder, projector, 8B model, FA2, Transformers, GT, historical answers,
benchmark scoring or LLM generation is imported or executed. This is a pruning
graph replay, not a new native_get or end-to-end model evaluation.

Each question first executes unchanged source importance and greedy selection
on GPU. The observed and unobserved source greedy calls must agree. Importance,
all256 ordered IDs, the full256-step gain trace and the final spatial BF16
sequence must **all match the archived GPU cache bitwise** before the Eq7 arm
is even run. A failed source anchor is saved and explicitly unmeasured; it is
not retried or replaced with a CPU approximation. Every fixed question remains
in the report, including all source-anchor failures.

The recorded original runtime is torch2.8.0/CUDA12.8, default FP32, no autocast
and matmul TF32 false. The host author interpreter imports torch2.8.0+cu128 in
an empty CUDA environment without initialization. Original cuDNN TF32,
benchmark and deterministic flags were not recorded; present defaults are
disclosed rather than claimed as historical observations. They are not tuned
to close anchors. Exact source anchors decide whether each measurement exists.

The default controller command is a stdlib-only check:

```sh
CUDA_VISIBLE_DEVICES='' /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python -B run.py
```

The single GPU attempt requires a future explicit root allocation note bound to
the frozen plan, slot A, CUDA_VISIBLE_DEVICES0 and exact allowed other GPU PID,
starttick and boot identities. Both controller and worker recheck this before
the worker's first CUDA access; loading and this small cache context count
toward the global maximum of two. An exclusive claim prevents a second attempt.
There is no waiter, scheduler or automatic retry. This preparation does not
request CUDA availability, load a GPU tensor or grant an allocation.

After the GPU worker actually exits0 and disappears from/proc and nvidia-smi,
a separate NumPy-only process audits the complete sixteen records. It checks
source cache bytes, saved GPU array hashes/dtypes/finiteness, each256-step
argmax, uniqueness and masking, spatial BF16 reconstruction, source anchors and
explicit unmeasured records. Gain differences through the first divergent
choice are separated from later differences in the -1e9 selection sentinel.
The report does not invoke source pruning again or compute accuracy.

`execution/release.json` is written only after worker exit and this audit.
The root must still confirm the controller's actual execution handle returned
and the controller has exited. Any failure preserves all partial arrays/logs
and identity records. Audit success can include unmeasured questions; it never
means all variants measured or that the paper results were reproduced.
