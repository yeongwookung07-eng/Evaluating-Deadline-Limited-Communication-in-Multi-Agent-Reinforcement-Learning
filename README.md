# Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning: Delivery, Computation, and Control

Research code accompanying the manuscript by **Yeong-woo Kung and Young-guk Ha**.
This study separates traffic admission, timely delivery, sender selection, initialization, CPU readiness, and acting-policy use under action deadlines and receiver caching.

[Paper (PDF)](experiments/deadline_comm/publication/kbs/kbs_main.pdf) · [LaTeX](experiments/deadline_comm/publication/kbs/kbs_main.tex) · [Reproduction](REPRODUCE.md) · [Evidence](EVIDENCE.md) · [Completeness audit](COMPLETENESS.md) · [Current claims](claim.md) · [Third-party attribution](THIRD_PARTY_NOTICES.md)

The contribution is a controlled evaluation framework and its implementation and analysis. QMIX and Hallway are existing components credited to their original sources. DAC is an evaluated heuristic, not a generally superior scheduler. The manuscript is prepared in KBS format; submission or acceptance is not asserted. Author declarations remain pending.

The four stages validate learning, account for delivery and computation, isolate sender choice through matched full-state replay, and test downstream information use. Oracle-assisted Hallway and Oracle-free broadcast are reported separately.

Retained findings include 400 Full–RR delivery-identical Hallway pairs with 80% less attempted traffic; weighting-sensitive sender effects; common-prefix and readiness controls; receiver-input probes; the zero-message public-clock Hallway solution; failed XOR/SMACv2 learning gates; and a limited SMACv2 technical smoke. These do not establish task-level message necessity or learned standard-benchmark generalization.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Source, settings, tests, manuscript and small recorded results are included. Four compressed raw logs and a safe editorial audit are now included; trained checkpoints, receiver NPZ inputs and remaining detailed traces are not included. Full experimental reproduction requires those inputs; no downloadable Release is claimed. See [한국어 구성 안내](GUIDE_KO.md) and [verification instructions](REPRODUCE.md). A successful unit test is not a full paper rerun.

| Component | Location | Status |
| --- | --- | --- |
| Harness, policy, transport, cache | `experiments/deadline_comm/` | Existing implementation |
| Sender and delivery controls | `selection_validation/`, `review_revision/`, `tmlr_revision/` within the experiment | Completed recorded studies |
| Robustness and policy-use analysis | `tmlr_revision/robustness/` | Existing evidence retained |
| SMACv2 | `standard_benchmarks/`, `smac_protocol/` | Failed learning gate; technical smoke only |
| Snapshot verification | `CHECKS.md` | CPU test scope and excluded inputs |

All commands run from the repository root. Preserved historical configurations can point at existing output directories: choose a new output directory before rerunning. Full training is not required for the unit checks.

| Work / date | Status | Entry, inputs, command and result |
| --- | --- | --- |
| Public evidence completeness / 2026-11-22 | Completed audit; full experiment reproduction remains incomplete | `python scripts/verify_editorial.py`; `evidence/editorial/manifest.json`; four historical logs reproduce all fields of `experiments/deadline_comm/tmlr_revision/results/editorial_audit.json`. No new experiment/config/training. See [audit](COMPLETENESS.md). |
