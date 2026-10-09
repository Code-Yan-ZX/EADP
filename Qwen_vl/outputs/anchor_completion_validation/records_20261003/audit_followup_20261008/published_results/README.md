# Published repaired-protocol results

Only complete, independently scored comparison groups appear here. 
Read [paired_results.csv](paired_results.csv) for scores and each group manifest for prediction, input and source SHA256 evidence.

TextVQA uses all 5000 official validation questions. ScienceQA uses the 2017 image questions in the shipped CQM-A input. POPE uses the three-category mean F1 over 8910 predictions with the controlled CUDA stream wait.

For NeXT, K128/K64/K32 denotes the per-crop budget parameter (nominal 640/320/160 for five crops). Actual retained tokens are recorded per sample in the runtime evidence.

The continuation rows use a frozen stream-wait repair. Their protocol is shown in the CSV; original scores remain separate, regardless of which score is higher.

Legacy single-method audit arms are in [repaired_legacy_results.csv](repaired_legacy_results.csv); they have no paired EADP difference. MME retains its raw perception points. MMBCN retains the historical English instruction (`lang=en`); this is recorded separately from the dataset language.

| Group | EADP | AnchorZip | AZ − EADP | FULL | Evidence |
|---|---:|---:|---:|---:|---|
| legacy_v15_mme_FULL_streamwait |  |  |  | 1507.059 | [manifest](legacy_v15_mme_FULL_streamwait/manifest.json) |
| legacy_v15_mme_K128_streamwait | 1431.982 | 1445.555 | 13.573 |  | [manifest](legacy_v15_mme_K128_streamwait/manifest.json) |
| legacy_v15_pope_K128_streamwait | 87.520 | 87.455 | -0.065 |  | [manifest](legacy_v15_pope_K128_streamwait/manifest.json) |
| next_sqa_FULL |  |  |  | 67.675 | [manifest](next_sqa_FULL/manifest.json) |
| next_sqa_FULL_streamwait |  |  |  | 67.625 | [manifest](next_sqa_FULL_streamwait/manifest.json) |
| next_sqa_K128 | 67.675 | 68.071 | 0.397 |  | [manifest](next_sqa_K128/manifest.json) |
| next_sqa_K128_streamwait | 67.824 | 68.121 | 0.297 |  | [manifest](next_sqa_K128_streamwait/manifest.json) |
| next_sqa_K32 | 67.576 | 67.179 | -0.397 |  | [manifest](next_sqa_K32/manifest.json) |
| next_sqa_K32_streamwait | 67.476 | 67.129 | -0.347 |  | [manifest](next_sqa_K32_streamwait/manifest.json) |
| next_sqa_K64 | 67.229 | 67.576 | 0.347 |  | [manifest](next_sqa_K64/manifest.json) |
| next_sqa_K64_streamwait | 67.229 | 67.476 | 0.248 |  | [manifest](next_sqa_K64_streamwait/manifest.json) |
| next_textvqa_FULL |  |  |  | 60.370 | [manifest](next_textvqa_FULL/manifest.json) |
| next_textvqa_FULL_streamwait |  |  |  | 60.370 | [manifest](next_textvqa_FULL_streamwait/manifest.json) |
| next_textvqa_K128 | 57.958 | 57.588 | -0.370 |  | [manifest](next_textvqa_K128/manifest.json) |
| next_textvqa_K128_streamwait | 57.866 | 57.634 | -0.232 |  | [manifest](next_textvqa_K128_streamwait/manifest.json) |
| next_textvqa_K32 | 51.996 | 53.174 | 1.178 |  | [manifest](next_textvqa_K32/manifest.json) |
| next_textvqa_K32_streamwait | 52.086 | 53.262 | 1.176 |  | [manifest](next_textvqa_K32_streamwait/manifest.json) |
| next_textvqa_K64 | 55.352 | 56.026 | 0.674 |  | [manifest](next_textvqa_K64/manifest.json) |
| next_textvqa_K64_streamwait | 55.410 | 56.292 | 0.882 |  | [manifest](next_textvqa_K64_streamwait/manifest.json) |
| v15_pope_K32_streamwait | 84.073 | 83.084 | -0.989 |  | [manifest](v15_pope_K32_streamwait/manifest.json) |
| v15_sqa_FULL |  |  |  | 69.509 | [manifest](v15_sqa_FULL/manifest.json) |
| v15_sqa_K128 | 69.559 | 69.212 | -0.347 |  | [manifest](v15_sqa_K128/manifest.json) |
| v15_sqa_K32 | 68.765 | 69.509 | 0.744 |  | [manifest](v15_sqa_K32/manifest.json) |
| v15_sqa_K64 | 68.815 | 69.113 | 0.297 |  | [manifest](v15_sqa_K64/manifest.json) |
| v15_textvqa_FULL |  |  |  | 58.226 | [manifest](v15_textvqa_FULL/manifest.json) |
| v15_textvqa_K128 | 56.390 | 56.604 | 0.214 |  | [manifest](v15_textvqa_K128/manifest.json) |
| v15_textvqa_K32 | 52.532 | 52.542 | 0.010 |  | [manifest](v15_textvqa_K32/manifest.json) |
| v15_textvqa_K64 | 54.918 | 54.978 | 0.060 |  | [manifest](v15_textvqa_K64/manifest.json) |

EADP K32 beta1 script-default sensitivity control: 50.966% over 5000 questions, with the same stream wait. This is a separate EADP control; its parameters are not verified to be the paper configuration. [manifest](next_textvqa_K32_default_beta1/manifest.json).
