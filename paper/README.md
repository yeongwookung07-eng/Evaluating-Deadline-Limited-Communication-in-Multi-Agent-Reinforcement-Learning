# JAAMAS submission edition

**Manuscript proof corrections, 2026-11-28.** Section 4 discloses how OpenAI Codex assisted with software, validation, plotting and manuscript preparation. Table 2 covers the five study cohorts, including the training gates and comparison conditions for Terran and native Predator–Prey. Section 5.6 cites Figure 2 for normal-access accuracy, Table 9 for source-aware accuracy and misses, and Figure 3 for seed-level gains. The cross-references to supplementary Sections S8 and S9 were checked against this edition. Six author surnames in the numbered reference lists were verified against the primary papers. [Main PDF](jaamas_main.pdf) · [Supplement](jaamas_supplement.pdf) · [Proof checks](editorial_proof_validation_20261128.json).

**Sequential sender studies, 2026-11-28.** Section 5.8 and supplementary Section S9 report two completed CPU evaluations on separately trained three-seed banks. Terran substitutions changed immediate joint actions at 22/30 decisions and realized suffix returns at 27/30. Round robin minus uniform was +0.352 [-0.345, 1.050] in native return units. Native Predator–Prey changed realized return at 22/139 sampled decisions, with a factual-minus-uniform episode contrast of +0.831 [−1.428, 3.090]. Its learning control passed and the zero-payload comparison was unresolved. The checks cover three frozen policies per task, common fresh reset panels and one prespecified Terran decision time. Confidence intervals are descriptive over training seeds, conditional on the reset panel. [Main PDF](jaamas_main.pdf) · [Supplement](jaamas_supplement.pdf) · [Standalone raw auditor](audit_sender_transfers.py) · [Audit records](sequential_sender_raw_audit_20261128.json). Earlier entries below document prior editions.

**Current 15-page review edition (2026-11-28).** Main text 8 pt, 15 pages including references and author details. Supplement 19 pages. The revised paper foregrounds the matched Hallway counterexample, specifies the queue identity conditions and the MNIST denominator, and bounds the receiver effect by seed, supervision and deadline regime. The public literature comparison adds Gao et al. (2025, 2026). Main has 211 abstract words, 9 tables, 4 figures and 31 references. [Internal mock review](jaamas_mock_review_ko_20261128.pdf) and [validation](review_15page_validation_20261128.json). Prior edition: `pre_15page_review_20261128.zip`. Earlier dated entries are historical.

**2026-11-28 author layout:** final-page author names are followed by role/email bullet points, with a separate shared-affiliation bullet. All author details retained; main 20 pages; no build warnings.

**2026-11-28 Funding wording:** main, title page, standalone and portal declarations now state: “This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.” Main remains 20 pages; clean build.

**2026-11-28 data and closing sections:** data paragraph describes Hallway/broadcast trajectories, MNIST inputs and supplementary SMACv2 checks; standalone and portal declarations synchronized. Section 6 is Discussion and limitations; Section 7 is Conclusion. Results, tables, equations and figures unchanged; main 20 pages, clean build.

**2026-11-28 Figures 1–4 size:** all four main result images now display at 0.85\linewidth, preserving aspect ratios and centering. Captions, prose, source plot assets and results unchanged. Main 20 pages; no build warnings.

**2026-11-28 Figure 1 size:** display width reduced from 1.0 to 0.85\linewidth; centered, aspect ratio preserved. Caption, prose and plot asset unchanged. Main 20 pages; clean build.

**2026-11-28 experiment setup:** Added Section 4.1 Experiment setup summarizing task roles and matching. Existing subsections are now 4.2–4.5; their text is unchanged. Main 20 pages, no build warnings. Snapshot: `pre_experiment_setup_20261128.zip`.

**2026-11-28 Problem Definition update:** Section 3 now defines the analytical problem (3.1), communication/receiver state (3.2), and computation/deadlines (3.3). Sections 4–6, declarations and experimental results unchanged. Author information is compacted on the final page with all details retained. Main 20 pages; no build warnings. Snapshot: `pre_problem_definition_20261128.zip`; validation: `problem_definition_validation_20261128.json`.

**2026-11-28 structure update:** Section 3 is Preliminaries (problem/policies, reception/computation/deadlines); Section 4 is Evaluation framework and experimental design (protocol, paired interventions, tasks, timing/uncertainty); Section 5 Results unchanged. All math/table bodies/figure blocks preserved. Main 20 pages; final page combines the end of references and author information. Snapshot: `pre_preliminaries_restructure_20261128.zip`; validation: `preliminaries_validation_20261128.json`.

**2026-11-28 subsection update:** Related Work now has numbered subsections 2.1 Learned communication and efficiency; 2.2 Scheduling rules and time-constrained decisions; 2.3 Evaluation of receiver capabilities and evaluation gap. The former gap paragraph is retained under 2.3. Prose/results unchanged; main 20 pages; no build warnings.

**2026-11-28 Related Work rewrite:** four thematic paragraphs describe communication learning/efficiency, scheduling rules/timing, receiver-capability evaluation, and the evaluation gap addressed by our controls. Only Section 2 changed; the abstract, results, tables and figures remain unchanged. Main 20 pages; no build warnings. Snapshot: `pre_related_work_narrative_20261128.zip`; validation: `related_work_validation_20261128.json`.

**2026-11-28:** Section 2 renamed to `Related work`; section content unchanged. Main remains 20 pages; build has no reference/layout warnings.

**2026-11-27 abstract update:** 236-word narrative abstract now follows background → problem → analytical approach → findings → implications. Only the abstract changed; main remains 20 pages. Validation: `narrative_abstract_validation_20261127.json`.

**Latest revision — 2026-11-27:** completed literature-grounded analysis. Main **20 pages including references, declarations and final author information**; supplement 19 pages. Page 1: title, Yeong-woo Kung / Young-guk Ha, Abstract; no Highlights or contact block. Full author information is on page 20; separate `Title_Page.pdf` supplies submission metadata required by the journal. Text left-aligned; figures/tables centered. All 9 main tables, 4 main figures, numerical tables/equations/figure assets and 29 bibliography entries preserved. No new experiment or training. Evidence mapping: `literature_grounding_20261127.json`; validation: `literature_rewrite_validation_20261127.json`. Earlier dated descriptions below are historical.

**Historical alignment update (2026-11-27):** all manuscript prose and front matter, Highlights and captions are left-aligned (ragged right); figures and tables remain centered. Main 20 pages, supplement 19 pages. Content, table cells, equations and plotted values are unchanged. `journal-reading-layout.sty` applies the shared presentation rules. Snapshot: `pre_left_text_alignment_20261127.zip`; validation: `left_alignment_validation_20261127.json`. Earlier alignment/page-count descriptions below are historical.

**Latest update — Highlights front page (2026-11-27):** page 1 contains five research highlights; title/authors/abstract begin on page 2. Total main length remains 20 pages, with 9 tables and 4 figures. Supplement S1–S8 now has 18 pages, 11 tables and 5 figures; the unchanged detailed diagnostic figure moved to S8. Validation: `highlights_frontpage_validation_20261127.json`. Earlier counts below describe preceding editions.

**Current edition — 2026-11-27:** main **20 pages including references and declarations**, supplement **17 pages (S1–S8)**. Main: 9 tables, 5 result figures and 29 references. Same 155 × 235 mm centered layout and 9.5 pt type. Introduction now states the problem and three findings; methods/design are consolidated, repeated interpretation shortened, and two tables/two plots plus two accounting equations relocated to S8. All 20 table bodies, 12 display-math blocks and 9 original plotted assets are preserved across the pair. Main scientific prose is edited, not byte-identical. Sources/bibliography entries remain unchanged; citation numbers follow their new order. Validation: `twenty_page_validation_20261127.json`; relocation: `twenty_page_relocation_20261127.json`. All preceding edition descriptions below are historical.

**Current layout:** author-requested reference-paper proportions, 155 × 235 mm with equal 16-mm side margins. Main 31 pages; supplement 15 pages. Scientific text/results remain unchanged; headings use sentence case. Current verification: `reference_layout_validation_20261126.json`. Earlier edition descriptions below are historical.

## Current copyedit and reference check — 2026-11-26

Corrected the weighting-section reference and the sender-sensitivity figure reference; replaced manual main-section numbers with labels. Standardized labeled/Student-$t$/$p$ spelling and repaired an ambiguous plural pronoun. Funding, interests and author roles retain the confirmed wording. Main, standalone and portal declarations now use the same order and wording; clarified that MNIST download instructions belong to the repository. No scientific table cells, equations, numerical results, figures or bibliography entries changed. Earlier dated status paragraphs below are historical. Current status supersedes their figure counts and declaration descriptions.

Snapshot: `pre_copyedit_20261126.zip`. Build: `python experiments/deadline_comm/publication/jaamas/build.py`. Validation: `copyedit_validation_20261126.json`.

Current edition includes the completed frozen-policy readiness-uncertainty and hard-action-fallback studies, retaining all earlier evidence and limitations. The official Springer Nature class/style remain unchanged. Earlier25/14-page documents and their exact preparation records are preserved in `pre_deployment_robustness_20261126.zip`. The historical KBS source is unchanged.

## Upload files and format

The standardized upload files are in [the separate submission folder](../../../../JAAMAS_submission_20261126/README.md) and the public worktree's [submission directory](../../../../github_analysis/submission/jaamas/README.md):

- `Main_Manuscript.pdf`:20 pages, 236-word abstract, five keywords, references and declarations; full author information on the final page.
- `Title_Page.pdf`:separate one-page title, author, affiliation and corresponding-contact metadata.
- `Manuscript_Source.zip`:editable main/supplement,figures,bibliography,official class/style and build helper. Primary source `jaamas_main.tex`.
- `Information_Sheet.pdf`:2pages, main claim, concrete evidence, closest related work and prior publication.
- `Cover_Letter.pdf`:1page.
- `Supplementary.pdf`:17pages,sectionsS1–S7.
- `Statements_and_Declarations.pdf`:1page, convenience copy of declarations already inside the main manuscript; `DECLARATIONS.md` supplies portal text.
- `Reviewer_Archive.zip`:optional full code/data/checkpoint/result archive in the local submission folder; generated by `github_analysis/scripts/package_jaamas.py --reviewer`. It is not committed as a large duplicate of the GitHub repository.
- `PERMISSIONS.md`:provenance/conditional permission record. No fictitious permission grant.

Word is an alternative to this chosen LaTeX workflow, so no Word manuscript is supplied. Highlights and Graphical Abstract are omitted from the active checklist. Existing KBS materials remain preserved.

## Evidence and status

The [follow-up report](../../deployment_robustness/results/RESULTS.md) records+9.25pp primary noisy-readiness admission gain and+5.63pp normal hard-hold receiver gain, together with error/cost reversals and zero-gain all-late conditions. The original scientific table bodies remain intact; new analyses are in the main subsection `sec:deployment_robustness` and supplementaryS7. No new training or evaluation was performed for this packaging task.

Author-confirmed no funding,no competing interests,no prior conference/journal publication and the existing contribution roles are retained. At the author's latest request, Methods now discloses OpenAI Codex assistance with research software, validation, aggregation and manuscript preparation; no unknown model version or unverified human approval is invented. Final co-author approval and exclusive consideration remain portal declarations for the submitting author; no submission is asserted.

## Build and validation

Build local documents: `python experiments/deadline_comm/publication/jaamas/build.py`. Build/package the public version: `cd github_analysis && python scripts/package_jaamas.py`. All five PDFs compile without unresolved references,missing characters or overfull boxes. A fresh extraction of `Manuscript_Source.zip` rebuilds all five PDFs with identical extracted text. See `jaamas_submission_validation_20261126.json` and `build_validation.json`. Original dated `content_validation.json` applies only to the preserved pre-extension edition.

The public checkout verifies5,741assets and375checkpoints, all earlier paired/probe/replay evidence and the new1,382,400MNIST example cells plus18,000broadcast episodes.63tests pass,1artifact-dependent early-smoke test skips. No GPU or training. Public publish status is recorded in `docs/GITHUB_ANALYSIS_UPLOAD_20261124.md`; local readiness does not assert a successful push.

Official guidance checked on2026-11-26: https://link.springer.com/journal/10458/submission-guidelines . Official template provenance remains in `provenance.json`. The user's supplied checklist and the guide agree on the Information Sheet,abstract/keywords,editable source and declaration requirements. This record does not certify every portal field.

## Review corrections and clean installation — 2026-11-26

The current revision adds primary hard-hold miss fractions (0.100% original; 0.058% adapted) and the rare-overrun interpretation in Section 5.8; sharpens the Introduction/Information Sheet mechanism claim; relates it to Peters et al.'s 2025 JAAMAS metric survey; discloses actual AI assistance in Methods; and adds the journal name to the supplement cover. Figure 1 now uses 10pt axes and 9pt ticks/legend at the final 119mm width, from identical frozen values. `plot_diagnostics.py` regenerates it. The preceding edition is preserved in `pre_review_fixes_20261126.zip` and `JAAMAS_submission_20261126_before_review/` at repository root. No scientific experiment or training was added.

An independently downloaded public source archive was installed in an isolated Python 3.13.12 venv with CPU-only PyTorch. All 5,741 asset hashes and 375 checkpoint loads pass; 63 tests pass, one unavailable early-smoke fixture skips. The new `github_analysis/experiments/reproduce_paper_tables.py` recomputes 22 printed rows across four tables and verifies the actual LaTeX values, with exact coverage and exclusions in `github_analysis/docs/TABLE_REPRODUCTION.md`. See that directory's clean-install validation record. This is saved-data reaggregation, not retraining or an assertion of hardware-time invariance.

The revised reviewer package also passed both commands after a fresh extraction into another directory in the isolated CPU venv. The final editable source ZIP independently rebuilds all five PDFs with identical text; 11 main and 9 supplementary table blocks remain unchanged. That preceding review edition has 28/17/2/1/1 pages with no layout or reference warnings.

## Current problem-and-rationale rewrite — 2026-11-26

Rewritten at the author's request using the KBS manuscript and completed experiment records only. Abstract/Introduction now lead with why a cost–performance outcome needs an explanation to guide which component to change. Section 3 states the research question and the paired intervention quantity, explains why each existing control addresses a specific ambiguity, and places the four-stage protocol before implementation and results. Each analysis subsection connects its intervention to that question; Discussion/Conclusion explain what is supported and what remains unresolved. Information Sheet and cover letter are aligned.

Main 31 pages, supplement 17 pages, Information Sheet 2 pages, cover/declarations 1 page each; abstract 232 words, 5 keywords. All ten non-protocol main tables, the entire supplementary source, displayed mathematical expressions, result-section numeric tokens, four figures and bibliography are unchanged. The protocol table is moved and rewritten as questions/controls. Existing measured/fixed readiness, supervised head feedback, state/episode weighting, null memory-return findings, failed learning/replay gates and hard-fallback limitations remain explicit. No new source, experiment, training, model or numerical result.

[KBS-to-claim evidence map](problem_rationale_evidence_map.json) and [validation](problem_rationale_rewrite_validation.json) document the scope. The preceding28-page edition is preserved in `pre_problem_rationale_rewrite_20261126.zip` and the root folder `JAAMAS_submission_20261126_before_problem_rewrite/`.

2026-11-26 author-requested edit: removed the AI-use paragraph from the active manuscript. The preceding source/PDF are preserved locally. Scientific results are unchanged.

## Findings-led revision — 2026-11-26

Applied the author’s supplied editorial direction to the latest JAAMAS source: a findings-led 218-word abstract, sequential questions in the Introduction, explicit Related Work positioning, renamed framework/results sections, transitions between analyses, and an interpretation-focused Discussion and Conclusion. Six new vector figures visualize the four-stage framework, delivery identity, equal-cost admission, sender sensitivity versus advantage, receiver decoding versus acting, and hard-fallback cost boundaries. Existing two main figures and all 11 table bodies are retained. Current main: 34 pages; supplement: 17 pages.

`python plot_findings.py` regenerates the six figures from the printed validated table values and explicitly identified prose results; `findings_figure_inputs.json` records the values, units and sources. Probe accuracy is separated from acting accuracy; state sensitivity from episode-weighted advantage; fixed-readiness receiver results from the later hard-fallback cohort. `python build.py` builds all five documents. No new experiment, training, configuration, model, or reference. Scope and failed learning/replay gates remain explicit.

Validation: [findings_revision_validation.json](findings_revision_validation.json); all 11 table bodies, 7 displayed equation/align blocks, supplementary source, bibliography and official class/style are unchanged. All five builds pass without reference or overflow warnings. New-figure PDF pages were visually inspected. Previous source/PDF: `pre_findings_revision_20261126.zip`; previous submission folder: `JAAMAS_submission_20261126_before_findings_revision/` at repository root.

## Result-only figures — 2026-11-26

Removed the explanatory four-stage flowchart at the author’s request. Its explanatory caption remains as prose; figure references now use the existing protocol table. All experimental plots, table bodies, equations and other prose are preserved. The renderer and source-package list no longer generate or include the flowchart. Previous source/PDF and diagram are preserved in `pre_framework_removal_20261126.zip`.

Validation: `framework_removal_validation.json`; main 33 pages, supplement 17 pages, all five builds without layout/reference warnings. Submission PDF and editable source ZIP synchronized.

Table 10 layout correction (2026-11-26): removed inherited footnote-size text and compressed column spacing; the memory-removal table now spans the text width with normal-size cells. Caption wording and all data are unchanged. Snapshot: `pre_memory_table_layout_20261126.zip`.

Copyedit closure: main 33 pages, supplement 17 pages; all five documents compile without warnings. Tables (11 main, 9 supplement), equations, figures, numeric tokens, bibliography and citation order are preserved. An extracted submission source independently rebuilds all five PDFs with identical text. Main declaration page 30 and standalone page 1 visually checked. No external reference-metadata re-audit or new scientific run.

Figure 7 label placement (2026-11-26): removed the annotation arrow and centered “All candidates late” above the 500-microsecond point, on two lines with enough space to avoid the plotted curve. Numerical inputs, uncertainty intervals, caption and manuscript source are unchanged. Graph visually checked; five-document build passes. Snapshot: `pre_execution_label_20261126.zip`.

## Reference-paper layout — 2026-11-26

Author-requested presentation revision based on the supplied Liu et al., “Improving scalability of multi-agent deep reinforcement learning with suboptimal human knowledge,” JAAMAS 40, 2 (2026), DOI 10.1007/s10458-025-09729-1. Local reference: `docs/coordination_computation_discussion_20261112/s10458-025-09729-1 (1).pdf`. Its measured page is 155 × 235 mm, text width approximately 123 mm, body 9.5 pt. The new `journal-reading-layout.sty` applies a centered 123-mm text block with equal 16-mm side margins, Times-compatible body text and compact sans-serif headings, full-width abstract/captions, and centered floats. Main section headings use sentence case. Scientific prose, numbers, tables and plotted inputs are retained. Official class/style are preserved; this is an author-prepared reading layout, not a publisher-issued proof. Earlier layout descriptions are historical. Snapshot: `pre_reference_layout_20261126.zip`.

Reference-layout closure: every main/supplement figure is centered to within 0.02 PDF points, all text stays inside the side margins, and no LaTeX warnings remain. All scientific prose, 20 table bodies, equations, numerical plotting inputs and original figure PDFs are preserved. The dedicated style is included in portable source packages.

## 20-page editorial revision — 2026-11-27

Completed at the author’s request: 20 main-document pages including references/declarations, with supplement separate. Follow the supplied Liu et al. paper’s problem–contribution–method–experiment–conclusion organization. Condense repeated interpretation; relocate detailed controls and duplicate visualizations to Supplementary S8. Preserve all original numerical table cells, figure assets, equations, supervision disclosures and negative results across the document pair. No training or scientific experiment. Snapshot: `pre_20page_revision_20261127.zip`.

20-page validation: all references resolve and both documents compile without warnings; the unchanged figure data regenerate from the edited main tables. Saved-observation reaggregation reproduces 22 distinct printed rows across four tables and the primary deadline-miss/failure controls. This is an editorial revision with saved-data verification, not new training, new trajectories or new timing measurements. Previous 31-page main/15-page supplement: `pre_20page_revision_20261127.zip`.

## Highlights first page — 2026-11-27

At the author’s request, a dedicated first page gives five highlights covering the four-stage framework, Full–RR identity, unresolved sender advantage, equal-cost MNIST admission and conditional supervised receiver benefit. Title/authors/abstract follow on page 2. The full-page diagnostics plot is retained unchanged in Supplementary S8 to keep the main PDF at 20 pages. All numerical findings and diagram assets are preserved; the normal font size and geometry remain unchanged. Snapshot: `pre_highlights_frontpage_20261127.zip`.
