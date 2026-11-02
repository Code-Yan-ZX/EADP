POPE historical wrapper: bounded source and provenance audit
==========================================================

`audit.py` performs only targeted CPU AST/hash/metadata checks. It does not
import torch, load a model, parse predictions, score a benchmark or start GPU.
`audit.json` preserves exact paths, hashes, source lines and explicit limits.

The inspectable old pope_loader_runtime.py has the same data/prompt/collation
functions as the fresh author model_vqa_loader.py. Its complete eval_model AST
becomes identical after only removing the recorded-disabled AnchorZip install
branch, the generate-return count logger, and observer metadata. The native
generation kwargs, transfer dtype and prompt logic are unchanged. The old and
current three-wait CLIP forward ASTs have the same d7dece035… hash.

The old loader helper does have generation-time SHA binding: the completed
protocol records24d712b8…, its baseline source map pins the same helper, and
the old worker checks that map before runpy execution. The launcher7405822…
and recovery helperd6a9cb6c… are also tied to the completed recovery record;
the recovery code verifies the worker hash immediately before launching it.
The actual old stdout records the installed CLIP patch hash. This is stronger
than merely observing those source files today.

That chain does not supply all missing historical facts. The old protocol map
does not bind the actual loaded builder/pruner/LLM/conversation/mm_utils core
identities, component dtypes, attention backend, complete package versions or
alternative-architecture environment settings. Their inspectable current
copies and the current run03 observations cannot reconstruct those at-run
facts. The real recovery launcher inherits os.environ; fresh llava_llama.py
lines28–38 can select another mixin through USE_LLAVA_ARCH_* switches. There
is no evidence here that any such switch or precision difference was actually
active. Do not convert an unobserved field into an asserted runtime mismatch.

Existing audits already cover8910 canonical questions, historical scoring,
500 images/8910 prompts under tested small-package pins, and2334 duplicate
input groups with0 answer conflicts. This pass does not repeat those scores.

The recommendation is to avoid a new full8910 run solely because the wrapper
has a different filename or the old beta2 score is closer to the paper. Full
runtime equality is not proven, so a contemporary beta-only bridge is not
declared inherently useless. No observed conflict currently makes it a required
repair. If contemporary reproducibility becomes necessary, an existing fixed
panel at the exact old recipe can compare raw outputs/counts without GT or
accuracy first; it cannot prove8910 equivalence. No panel plan is created here.
