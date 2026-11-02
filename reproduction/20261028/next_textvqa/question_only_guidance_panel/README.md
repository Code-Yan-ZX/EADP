# Fixed16 TextVQA guidance boundary probe — CPU preparation only

This tests one inactive author-source suggestion: split the CLIP `texts=` input
at `\nReference OCR token`, while the LLM receives its complete original OCR
prompt and answer instruction. The published code currently retains OCR for
CLIP, and the paper does not uniquely prescribe this alternative. This is an
input hypothesis supported by a source comment, not a confirmed bug fix or a
paper recipe. No GT, accuracy, benchmark score, candidate search or retry.

The exact old native16 uniform panel remains positions195,507,820,1132,1445,
1757,2070,2382,2695,3007,3320,3632,3945,4257,4570,4882, sixteen unique images.
Parameters remain CLI64/nominal320, alpha.5/beta1, greedy128, SDPA/FP16,
vicuna_v1 and the already declared three caller-stream waits. The original
fresh `load_pretrained_model` and complete native model are used. Native CLIP,
projector, entropy q.2, aggregation, normalization over all five crops,
smoothing, floor/min1 quotas and facility math are unchanged. No forced total320.

The existing16 caches lack original CLIP image_embeds. This probe performs real
native extraction, never synthesizes those vectors or reconstructs a partial
7B model. It saves the actual FP32 image_embeds and CLIP text states after each
generate for future reuse. The original NeXT embedded vision tower remains;
official base CLIP vision weights are not substituted.

The single worker runs all16 source generations first. Every source must match
old16 projected FP16/pixels and FP32 global/dense/visualsim/importance exactly,
then match new fullK64 complete input-ID hashes, actual quotas/masks/effective
prefixes, output IDs and decoded text at those same16 positions. Existing
predictions are used only for this source consistency gate, never sample choice
or target-score selection. One failure retains the attempted output and stops;
no variant generation can start until all16 source anchors pass.
Every completed attempted generation is first saved as raw NPZ+attempt JSON,
before any post-generation content/source equality assertions. A failed check
adds a failed_checks sidecar referencing that raw attempt and immediately stops;
the absence of a passed case record means the attempt has never been accepted.

Only then do16 question-only generations run with the same complete LLM prompt,
image path, model, visual math and generation arguments. Both actual native
projected features, image_embeds, visualsim and image tensors must be bitwise
identical. Native spatial input order is reconstructed against real prepared
embeddings; complete LLM IDs and the nonvisual prefix/suffix must remain bitwise
identical even if crop quotas/visual count change. Observers retain references
and call original bound methods once. No captured GPU tensor is cloned/read or
copied until native generate returns. Timings are diagnostic, not efficiency.

Outputs record actual IDs, masks, crop quotas/effective prefixes, full prepared
FP16 embeddings and feature arrays. Each arm's BOS/EOS/returned/new-token count
is independently checked from guarded model/generation configs. A max128 result
without EOS is only a bounded-horizon observation; equal text/IDs does not rule
out logits or later differences. Text is worker native-tokenizer decoding; the
independent CPU validator checks IDs/records but does not separately re-decode.
Fresh source captures are native; no cached-mask or feature substitution occurs.

Preparation commands (no GPU/model) are:

```sh
python probe.py prepare
CUDA_VISIBLE_DEVICES='' python cpu_acceptance.py
python probe.py check
```

Only the root may later allocate a GPU. There is no queue or watcher. The future
allocation note must contain authorization=explicit_root_gpu_allocation,
run_id, exact plan_sha256, CPU_acceptance_sha256, GPU_slot A/B, GPU_device0,
boot_id, single_attempt=true, automatic_retry=false and exact
allowed_other_gpu_processes=[{pid,start_ticks},...]. The note is checked against
actual nvidia-smi PIDs/proc identities, with fewer than2 total pre-existing
models; loading this worker counts as one. The native worker also verifies its
parent PID/startticks. The one exclusive launch.claim prevents a second attempt.

The root's future execution command is the registered environment Python:

```sh
/home/dell/miniconda3/envs/llava_pruner/bin/python probe.py run --allocation-note ROOT_NOTE.json
```

The controller waits for the single worker, requires actual worker/GPU exit,
then runs only the CPU array/identity validator. It records complete/release,
but the root must separately prove actual controller/exec exit. Any failure is
terminal with no retry. Old code, data, full predictions, source caches and
ledgers are never overwritten. New32 compressed feature/embedding caches may
occupy roughly1–2GB; they are research artifacts rather than model weights.
