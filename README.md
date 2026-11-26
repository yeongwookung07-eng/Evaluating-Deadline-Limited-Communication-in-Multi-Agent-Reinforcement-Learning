# Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning

**Delivery, Computation, and Control — JAAMAS Regular Paper preparation**

This repository contains the manuscript, submission documents, research code, generated data, trained checkpoints and recorded results. The study separates offered traffic, timely delivery, sender choice, inherited reception history, computation and receiver use. It reports conditional benefits and failure boundaries, not a generally superior scheduler.

## JAAMAS submission files

[Download document bundle](submission/jaamas/JAAMAS_Submission.zip) · [Submission checklist](submission/jaamas/README.md)

| File or item | Status | Purpose |
|---|---|---|
| [Main_Manuscript.pdf](submission/jaamas/Main_Manuscript.pdf) | Prepared | Springer Nature manuscript, 216-word abstract, five keywords |
| [Manuscript_Source.zip](submission/jaamas/Manuscript_Source.zip) | Prepared; required for this LaTeX submission | Editable main/supplement sources, bibliography, figures, class/style and build helper |
| [Information_Sheet.pdf](submission/jaamas/Information_Sheet.pdf) | Prepared; required | Two pages answering claim, evidence, closest work and prior publication |
| [Cover_Letter.pdf](submission/jaamas/Cover_Letter.pdf) | Prepared | Contribution and relevance to JAAMAS |
| [Supplementary.pdf](submission/jaamas/Supplementary.pdf) | Prepared | Sections S1–S7, including uncertainty and hard-action checks |
| [Statements_and_Declarations.pdf](submission/jaamas/Statements_and_Declarations.pdf) | Included in manuscript; separate copy supplied | Funding, interests, contributions, data and code availability |
| Reviewer_Archive.zip | Optional; generate locally | Complete code, data, checkpoints and results; see command below |
| [Permissions record](submission/jaamas/PERMISSIONS.md) | Conditional | Provenance and any applicable permission evidence |
| Main_Manuscript.docx | Not supplied | Word is an alternative to the selected LaTeX workflow |
| Highlights / Graphical Abstract | Not included in active bundle | Not part of the supplied Regular Paper checklist |

Prepared does not mean submitted. Portal author-approval and exclusive-consideration declarations remain the submitting author's responsibility. The previous KBS edition is preserved in [paper/archive](paper/archive/kbs_20261126/README.md).

## Evidence and reproduction

Frozen-policy follow-up: readiness error SD100µs plus decision overhead100µs retains a +9.25pp equal-traffic admission advantage; larger error/cost reverses it. Normal broadcast retains a +5.63pp receiver-head gain with actual late-action hold, while an extra500µs or privileged source-aware overhead removes the gain. See [full results](experiments/deadline_comm/deployment_robustness/results/RESULTS.md), [한국어 결과](experiments/deadline_comm/deployment_robustness/results/RESULTS_KO.md) and [figure](experiments/deadline_comm/deployment_robustness/results/robustness.png).

```bash
python -m pip install -r requirements.txt
python experiments/prepare_inputs.py
python experiments/verify_results.py --recompute --tests
```

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
