# JAAMAS manuscript sources

Current target: Autonomous Agents and Multi-Agent Systems, Regular Paper. The official Springer Nature class and bibliography style remain unchanged. `jaamas_main.tex` and `jaamas_supplement.tex` are the editable main/supplement; information sheet, cover letter and declaration-copy sources are adjacent.

The main paper includes the frozen-policy readiness uncertainty and hard-action fallback results. The supplement provides S1–S7, with controls, seeds, costs, exact expectation formulas and failure boundaries. The original KBS edition is retained under [archive/kbs_20261126](archive/kbs_20261126/README.md); its dated validation records do not describe this edition.

Run `python scripts/package_jaamas.py` from the repository root to compile all documents and refresh [submission files](../submission/jaamas/README.md), source ZIP and checksums. Requirements: Python, pdflatex, BibTeX, Poppler pdfinfo. To build sources only, run `python paper/build.py`. Current build: main28pages, supplement17pages, Information Sheet2pages, cover letter1page; abstract216words and5keywords. Build checks reject unresolved references, missing characters, overfull boxes and an Information Sheet outside1–2pages.

Scientific evidence: [robustness report](../experiments/deadline_comm/deployment_robustness/results/RESULTS.md), [earlier results](../docs/RESULTS.md). The submission checklist distinguishes prepared files from author approval and journal submission.

Review revision (2026-11-26): primary hard-hold miss rates and their scope are explicit in Section 5.8; Introduction/Information Sheet clarify identification findings and the relationship to journal-level metric reviews; Methods records actual AI assistance; the supplement cover names the journal. Figure 1 has 10pt axis labels and 9pt ticks/legend at 119mm width. `python paper/plot_diagnostics.py` uses frozen `figure_inputs.json` and records its numeric primitives; it launches no experiment. Source ZIP includes both files. [Selected table reproduction](../docs/TABLE_REPRODUCTION.md) checks printed numbers against saved arrays/action traces.
