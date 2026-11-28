# Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning

**Delivery, Computation, and Control — JAAMAS Regular Paper preparation**

This repository contains the manuscript, submission documents, research code, generated data, trained checkpoints and recorded results. The study separates offered traffic, timely delivery, sender choice, inherited reception history, computation and receiver use. It reports conditional benefits and failure boundaries, not a generally superior scheduler.

The two sequential sender studies use three restored policies per task. Terran substitutions changed immediate joint actions at 22/30 decisions and realized suffix returns at 27/30. Round robin minus uniform was +0.352 [-0.345, 1.050] in native return units. Native Predator–Prey changed realized return at 22/139 sampled decisions, with a factual-minus-uniform episode contrast of +0.831 [−1.428, 3.090]. Its learning control passed and the zero-payload comparison was unresolved. [Raw audit and fresh-run commands](docs/SEQUENTIAL_SENDER_REPLAY.md).

## JAAMAS submission files

[Download document bundle](submission/jaamas/JAAMAS_Submission.zip) · [Submission checklist](submission/jaamas/README.md)

| File or item | Status | Purpose |
|---|---|---|
| [Main_Manuscript.pdf](submission/jaamas/Main_Manuscript.pdf) | Prepared | 15-page manuscript, 220-word abstract, five keywords |
| [Manuscript_Source.zip](submission/jaamas/Manuscript_Source.zip) | Prepared; required for this LaTeX submission | Editable main/supplement sources, bibliography, figures, class/style and build helper |
| [Information_Sheet.pdf](submission/jaamas/Information_Sheet.pdf) | Prepared; required | Two pages answering claim, evidence, closest work and prior publication |
| [Cover_Letter.pdf](submission/jaamas/Cover_Letter.pdf) | Prepared | Contribution and relevance to JAAMAS |
| [Supplementary.pdf](submission/jaamas/Supplementary.pdf) | Prepared | 22 pages, sections S1–S9, including three-seed sequential sender experiments |
| [Statements_and_Declarations.pdf](submission/jaamas/Statements_and_Declarations.pdf) | Included in manuscript; separate copy supplied | Funding, interests, contributions, data and code availability |
| Reviewer_Archive.zip | Optional; generate locally | Complete code, data, checkpoints and results; see command below |
| [Permissions record](submission/jaamas/PERMISSIONS.md) | Conditional | Provenance and any applicable permission evidence |
| Main_Manuscript.docx | Not supplied | Word is an alternative to the selected LaTeX workflow |
| Highlights / Graphical Abstract | Not included in active bundle | Not part of the supplied Regular Paper checklist |

Prepared does not mean submitted. Portal author-approval and exclusive-consideration declarations remain the submitting author's responsibility. The previous KBS edition is preserved in [paper/archive](paper/archive/kbs_20261126/README.md).

## Evidence and reproduction

Frozen-policy follow-up: readiness error SD100µs plus decision overhead100µs retains a +9.25pp equal-traffic admission advantage; larger error/cost reverses it. Normal broadcast retains a +5.63pp receiver-head gain with actual late-action hold, while an extra500µs or privileged source-aware overhead removes the gain. See [full results](experiments/deadline_comm/deployment_robustness/results/RESULTS.md), [한국어 결과](experiments/deadline_comm/deployment_robustness/results/RESULTS_KO.md) and [figure](experiments/deadline_comm/deployment_robustness/results/robustness.png).

From the root of a fresh clone or extracted GitHub source ZIP (Linux, Python 3.13):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-cpu-verified.txt
python experiments/verify_results.py --recompute --tests
python experiments/reproduce_paper_tables.py
```

The earlier pre-transfer CPU validation passed 5,741 asset hashes, 375 checkpoint loads and 63 tests (one documented skip). The last command writes four CSVs and `verification.json` under `results/local_recompute/paper_tables/`, and fails on a mismatch with the manuscript. It reproduces 22 distinct printed rows across four tables; the two receiver-probe rows are outside this export. See the [table/source map and exact coverage](docs/TABLE_REPRODUCTION.md).

Verification checks recorded observations without training. Raw MNIST and the StarCraft II runtime are separate dependencies. Follow [reproduction instructions](docs/REPRODUCE.md). Rebuild documents and optionally generate the large portable reviewer ZIP:

```bash
python scripts/package_jaamas.py
python scripts/package_jaamas.py --skip-build --reviewer
```

The reviewer ZIP is written to `dist/Reviewer_Archive.zip`, not committed as a large duplicate of the repository. A GitHub source download already contains its scientific inputs.

| Directory | Contents |
|---|---|
| [submission/jaamas](submission/jaamas/README.md) | Upload-ready files, checklist and declarations |
| [paper](paper/README.md) | Editable JAAMAS sources and preserved earlier edition |
| [experiments](experiments/README.md) | Evaluation code and configurations |
| [data](data/README.md) | Generated arrays; raw MNIST is separate |
| [results](results/README.md) | Recorded outcomes, traces and file-level manifest |
| [checkpoints](checkpoints/README.md) | Frozen model files |
| [models](models/README.md) | Shared model components |
| [tests](tests/README.md) | Scientific contract and regression checks |
| [docs](docs/README.md) | Research map, interpretation, references and setup |

Yeong-woo Kung: yeongwoo@konkuk.ac.kr. Corresponding author Young-guk Ha: ygha@konkuk.ac.kr.

Current manuscript revision: research need → attribution question → reasons for the existing controls → observed findings and limits. [Paper rationale and evidence map](paper/problem_rationale_evidence_map.json). This is an editorial rewrite based on the KBS source and completed work; numerical evidence is unchanged.
