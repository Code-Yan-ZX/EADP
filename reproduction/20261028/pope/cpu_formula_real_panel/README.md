# Real-feature CPU formula controls, 2026-10-28

Two independent formula controls reuse the original16 image/prompt pairs. Both keep the native77 text candidates, q=.2, entropy/scoring temperatures and smoothing fixed. This is an importance diagnostic; no labels, generated answers or benchmark accuracy are used.

| Fixed recipe | Single change | Images with changed importance | Maximum absolute change | Facility keep-set result |
|---|---|---:|---:|---|
| alpha1.0_beta1.0 | Eq10_remove_numerator_epsilon_only | 16/16 | 9.775161743e-06 | Unmeasured |
| alpha1.0_beta1.0 | Eq6_linear_quantile_ties_only | 0/16 | 0 | Unmeasured |
| alpha0.5_beta2.0 | Eq10_remove_numerator_epsilon_only | 16/16 | 1.680850983e-05 | Unmeasured |
| alpha0.5_beta2.0 | Eq6_linear_quantile_ties_only | 5/16 | 6.586313248e-06 | Unmeasured |

Eq10 removes only the numerator1e-6; its denominator1e-6 remains. The32 fused maps (16images ×2recipes) are all nondegenerate, so the synthetic all1-versus-all0 boundary was not observed.

Eq6 uses torch.quantile(q=.2, interpolation=linear), retaining every entropy <= threshold including ties. All16 images retain16 text positions versus the original floor-topk15. The newly included position has at most0.000060362585 aggregation weight. At alpha1 the dense contribution is zero, so Eq6 changes no importance values. Under alpha.5/beta2, five images have changes no larger than6.5863e-6.

The sets of32 highest-importance indices are unchanged in this panel, but those indices are not facility-location selections. Native LLaVA computes facility similarity from FP16 mm_projector outputs. This torch2.1.2 CPU runtime rejects native FP16 linear (addmm_impl_cpu_ not implemented for Half). No FP32 projector simulation or new GPU work was substituted, and actual facility keep-set differences remain unmeasured. Small importance changes can still affect a near-tied GPU greedy gain; this panel does not exclude that.

Features were extracted once again for the same16 images using the native FP32 CLIP loaders and source FP32→FP16→FP32 image input rounding. Source top15 text indices and their entropies exactly match the prior CPU panel. The cross-modal arrays are cached in cross_modal_intermediates.npz with a SHA manifest, so the final analysis was regenerated from cache without another feature pass. CPU features are not claimed bit-identical to GPU intermediates.

These observations provide no recovered paper recipe or benchmark improvement and do not support scheduling algorithm variants by themselves. The frozen run03 plan and all its source/weight/image guards remain unchanged; plan SHA256 is bf40e511c27cf920dbd287d61126c405346f97fc37728cff3254af608aef5a70.

Full per-image records, feature manifest, declared quantile convention, runtime limitation and script SHA are in real_formula_comparison.json.
