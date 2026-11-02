# Fresh official TeX / archived table cross-check

PASS: all99 paper cells match exactly as Decimal values across freshly downloaded officialv1 TeX, archivedv1 HTML, its saved table JSON, and the existing comparison.json. This includes90 task cells and9 literal reported-average cells. Task order and all nominal budget blocks match.

| Model | Source EADP lines by nominal budget | Cells matched | Source TeX SHA256 |
|---|---|---:|---|
| v15 | 128: line23, 64: line36, 32: line45 | 33/33 | `75f8d559e4596878b0a9ca2ace43dc5fd49e4c15448a7ff4a7cea1073096f3fe` |
| next | 640: line24, 320: line37, 160: line47 | 33/33 | `3eb0705313c5ce254b366268cf70d082506f78960092d40840bce0d6bd1ab560` |
| qwen | 512: line15, 256: line20, 128: line25 | 33/33 | `9e1aef1c2289e874e98a9a176c8cecc1b5df30512eb1a1d9b3fac2c644189aae` |

LLaVA1.5 budgets are128/64/32; NeXT nominal budgets640/320/160; Qwen3 budgets512/256/128. The LLaVA source captions specify Avg9 excludingVizWiz; Qwen specifies Avg10. These are paper-reported averages, never local averages. No local values or averages were filled, recalculated, or overwritten.

crosscheck.json retains every one of the99 identities/values, exact source member/fileSHA/line/column, raw EADP and budget lines, caption/header, and archived-source fingerprints. The three local .tex copies are byte-identical members of the newly downloaded official archive. Existing comparison.json and archivedHTML hashes are unchanged. No GPU, current experiment, fixed plan, or method audit was altered.
