# JAAMAS manuscript sources

Current target: Autonomous Agents and Multi-Agent Systems, Regular Paper. The official Springer Nature class and bibliography style remain unchanged. `jaamas_main.tex` and `jaamas_supplement.tex` are the editable main/supplement; information sheet, cover letter and declaration-copy sources are adjacent.

The main paper includes the frozen-policy readiness uncertainty and hard-action fallback results. The supplement provides S1–S7, with controls, seeds, costs, exact expectation formulas and failure boundaries. The original KBS edition is retained under [archive/kbs_20261126](archive/kbs_20261126/README.md); its dated validation records do not describe this edition.

Run `python scripts/package_jaamas.py` from the repository root to compile all documents and refresh [submission files](../submission/jaamas/README.md), source ZIP and checksums. Requirements: Python, pdflatex, BibTeX, Poppler pdfinfo. To build sources only, run `python paper/build.py`. Current build: main27pages, supplement17pages, Information Sheet2pages, cover letter1page; abstract216words and5keywords. Build checks reject unresolved references, missing characters, overfull boxes and an Information Sheet outside1–2pages.

Scientific evidence: [robustness report](../experiments/deadline_comm/deployment_robustness/results/RESULTS.md), [earlier results](../docs/RESULTS.md). The submission checklist distinguishes prepared files from author approval and journal submission.
