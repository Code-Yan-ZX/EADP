# arXiv current-release check, 2026-10-28

The current [official abstract](https://arxiv.org/abs/2607.02484) lists only v1,
submitted 2026-07-02 17:50:57 UTC. Its HTML link still points to
[2607.02484v1](https://arxiv.org/html/2607.02484v1); no v2 or separate ancillary
link is listed.

The page's [official TeX source](https://arxiv.org/src/2607.02484) was freshly
downloaded: 2,043,245 bytes, SHA256
`7477093f610f243dbaaf6faf6a40175b34e43d793bdbfa01663436b89fe48e28`.
Content-Disposition explicitly names `arXiv-2607.02484v1.tar.gz`.
Its 34 entries contain the paper, tables, included supplementary TeX, figures
and document-building files. There is no `anc/` directory or executable,
benchmark environment, or per-task main-table command file. `00README.json`
specifies pdflatex/TeXLive2025, which describes document compilation.
The bounded command/environment-string and hidden TeX-comment checks found
no new reproduction recipe. The only research-code URL is the already audited
author GitHub repository; no new recipe link was followed.

The source HTTP Last-Modified date is packaging metadata, not evidence of a
new submitted version. The full retrieved abstract, source archive, file
inventory, response headers, URLs and hashes are preserved locally in
`release_audit.json`. Existing comparisons remain tied to paper v1. No GPU,
fixed experiment, author contact, or issue posting occurred.
