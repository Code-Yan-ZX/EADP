# EXPERIMENT TRACKER — SAGE

| Run | Milestone | System / arm | Panel | Status | Notes |
|-----|-----------|--------------|-------|--------|-------|
| R001 | M0 | edge-generator gates | bank fit (offline) | DONE | E(x) ≤512, valid, deterministic, tie rules |
| R002 | M0 | end-to-end labeling x3 instances | bank fit | DONE | 96 edges, 0 errors |
| R003 | M1 | y(S0) baseline generations | fit+val 300 | DONE | 0 errors |
| R004 | M1 | edge labels 8/img x g{1,2,4,8} | fit 240 | DONE | 7680 edges, 0 errors, 68 min |
| R005 | M1 | edge labels 8/img x g{1,2,4,8} | val 60 | DONE | 1920 edges, 0 errors |
| R006 | M2 | critic training g{1,2,4,8} x w{256,1024} | fit | DONE | set critic overfits; unary ≥ set on val RMSE |
| R007 | M3 | val deployment + tau sweep + TTFT | val 180 | DONE | frozen g8/w256/τ.207 (val +0.64), unary g8/w256/τ.107 (val +1.77); TTFT all within 318.9 ms |
| R008 | M4 | B2 identity arm | fresh 720 | DONE | macro 63.62, TTFT 293.8 ms |
| R009 | M4 | B1 facility arm | fresh 720 | DONE (rerun) | first run silently ran block8 (install() hardwires BASE_SELECTOR); fixed; real B1 −0.3, CI incl. 0 |
| R010 | M4 | SAGE (frozen critic+tau) | fresh 720 | DONE | **−1.30 [−2.3, −0.4]**, rate 43.8%, TTFT 307.9 ms |
| R011 | M4 | RND matched-rate random edge | fresh 720 | DONE (rerun) | −1.7 (first run failed on edge-index bug; fixed via key_edge_counts) |
| R012 | M4 | UNARY control | fresh 720 | DONE (rerun) | first run ran set path (family= arg missing); fixed; −1.0 |
| R013 | M4 | PERM label-permutation control | fresh 720 | DONE | **degenerate by construction** (argmax of permuted list = same element ⇒ ≡ SAGE); kept as record |
| R013b | M4 | PERM-X cross-image rotation (amendment) | fresh 720 | DONE | −0.9 [−2.0, +0.1]; replaces degenerate PERM for C4 |
| R014 | M4 | HINDSIGHT ceiling (4 edges/img) | fresh 720 | DONE (rerun) | +5.2 macro lower bound, active on 8.2% of images |
| R015 | M5 | paired stats + verdict | — | DONE | **REFUTED**: C1/C2 fail, C3 pass, C4/C5 pass vacuously |

Artefacts: `Qwen_vl/outputs/discovery/sage_*` (labels, phi, critics, calib,
per-arm records, verdict); results in `refine-logs/EXPERIMENT_RESULTS.md`.
M10 routing control: BLOCKED (no definition anywhere in repo).
