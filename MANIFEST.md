# MANIFEST — SAGE experiment bridge (2026-09-28)

| File | What it is |
|------|-----------|
| `refine-logs/EXPERIMENT_PLAN.md` | pre-registered plan (decisions 1–15) |
| `refine-logs/EXPERIMENT_TRACKER.md` | run-by-run status table |
| `idea-stage/docs/research_contract.md` | focused idea contract for session recovery |
| `Qwen_vl/scripts/discovery/sage_common.py` | edge support, features, frozen knobs |
| `Qwen_vl/scripts/discovery/sage_plan.py` | fresh locked confirmation split (720) |
| `Qwen_vl/scripts/discovery/sage_label.py` | paired Δ labeling, fit+val |
| `Qwen_vl/scripts/discovery/sage_critic.py` | phi build + critic/unary training |
| `Qwen_vl/scripts/discovery/sage_live.py` | live SAGEPruner/unary deployment machinery |
| `Qwen_vl/scripts/discovery/sage_calibrate.py` | val calibration, tau sweep, TTFT constraint |
| `Qwen_vl/scripts/discovery/sage_deploy.py` | confirmation arms B2/B1/SAGE/UNARY/RND/PERM/HINDSIGHT |
| `Qwen_vl/scripts/discovery/sage_analyze.py` | paired stats + pre-registered verdict |
| `Qwen_vl/outputs/discovery/sage_plan.json` | locked fresh split |
| `Qwen_vl/outputs/discovery/sage_labels_*.json` | labels (fit / val / sanity) |
| `Qwen_vl/outputs/discovery/sage_qbank_*.npz` | captured mean instruction embeddings |
| `Qwen_vl/outputs/discovery/sage_phi_*.npz` | critic feature matrices |
| `Qwen_vl/outputs/discovery/sage_critic_g*_w*.pt`, `sage_unary_g*_w*.pt` | frozen critics |
| `Qwen_vl/outputs/discovery/sage_calib.json` | frozen (g*, width*, tau*) + TTFT bound |
| `Qwen_vl/outputs/discovery/sage_conf_*.json|npz` | confirmation arm records |
| `Qwen_vl/outputs/discovery/sage_verdict.json` | pre-registered verdict |

Note: outputs land in `Qwen_vl/outputs/discovery/` (the project's convention),
not `refine-logs/`; the tracker and results summary in `refine-logs/` point at
them.
