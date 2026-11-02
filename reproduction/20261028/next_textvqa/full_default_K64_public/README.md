Public-default NeXT TextVQA CLI64 coverage
========================================

This prepares one complete 5000-question run at alpha0.5, beta1, CLI64,
nominal320. No model, GPU, scheduler or queue starts during preparation/check.
Root must review READY/plan and supply a new allocation note before `run.py run`.
All earlier K64 beta2 (55.410), K128 beta1 and K32 beta1/2 results stay intact.
This fills a published-default coverage gap; it is not a predicted repair or
proof of the unpublished paper argv. The full result is reported regardless
of its direction, without a score-based retry or parameter search.

`worker.py` compiles the frozen K128 worker with isolated output paths,
CLI64/nominal320 and matching verifier expectations. An inverse AST check
restores every original statement. One hook only retains native intermediate
references; all tensor copies/reductions occur after native generation returns.
Native three-wait repair, source, model, images, complete prompts, greedy128,
SDPA and FP16 remain under the existing complete identity guards.

The native importance-based strategy initializes five64-token crop quotas,
allocates proportionally with floor and minimum1, then uses each crop's valid
greedy prefix to build its mask. Actual total is observed, not forced to320.
Minimum-one clamping can also exceed nominal320 (a valid CPU source example
gives320+1+1+1+1=324); only actual mask/crop capacity is a hard bound.
Every row saves the five native quotas, every effective crop mask index and
ordered valid greedy prefix. The CPU verifier checks unique/in-range indices,
mask/prefix membership/count equality, actual retained return value, all5000
input token hashes, output token decoding and official/independent pointwise
TextVQA consensus. No discarded greedy suffix is saved.

After a single native attempt exits, the controller runs the CPU verifier and
fresh official CLI. It preserves the author score, independent exact score,
all raw answers and denominators, including any empty answer. Failure is
terminal and retained. `execution/release.json` certifies actual native exit;
the root-owned exec must separately certify the controller's actual return.
It does not indicate that the paper score or the overall goal was achieved.

Preparation: `llava_pruner/bin/python run.py prepare`

Read-only validation: `llava_pruner/bin/python run.py check`

Run only after allocation: `llava_pruner/bin/python run.py run --allocation-note ROOT_NOTE.json`

The allocation note names authorization `explicit_root_gpu_allocation`,
run_id `NeXT_public_default_K64_alpha05_beta1_wait`, the exact plan SHA,
GPU_slot A/B and the exact `allowed_other_gpu_pids`. Loading counts as a GPU
model; global maximum is2. No future allocation is implied by this document.
