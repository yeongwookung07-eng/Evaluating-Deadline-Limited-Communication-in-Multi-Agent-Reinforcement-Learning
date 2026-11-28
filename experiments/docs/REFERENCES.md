# References and code attribution

Research authors: Yeong-woo Kung and Young-guk Ha.

| Source | Use in this study |
| --- | --- |
| Foerster et al., *Learning to Communicate with Deep Multi-Agent Reinforcement Learning* (2016), `foerster2016learning`, https://arxiv.org/abs/1605.06676 | Learned communication and no-communication comparisons |
| Sukhbaatar et al., *Learning Multiagent Communication with Backpropagation* (2016), `sukhbaatar2016learning`, https://arxiv.org/abs/1605.07736 | Learned communication and channel-access reference |
| Rashid et al., *QMIX* (ICML 2018), `rashid2018qmix`, https://proceedings.mlr.press/v80/rashid18a.html | Monotonic mixer for the learned reference |
| Zhang et al., *Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control* (2019), `zhang2019efficient`, https://arxiv.org/abs/1909.02682 | Traffic reduction and retained task performance |
| Zhang et al., *Succinct and Robust Multi-Agent Communication with Temporal Message Control* (2020), `zhang2020succinct`, https://arxiv.org/abs/2010.14391 | Temporal reuse, buffered updates and reduced communication |
| Kim et al., *Learning to Schedule Communication in Multi-Agent Reinforcement Learning* (2019), `kim2019schedule`, https://arxiv.org/abs/1902.01554 | Learned access versus RR/full comparisons |
| Lowe et al., *On the Pitfalls of Measuring Emergent Communication* (AAMAS 2019), `lowe2019pitfalls`, https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf | Signaling, listening and message interventions |
| Eccles et al., *Biases for Emergent Communication in Multi-agent Reinforcement Learning* (NeurIPS 2019), `eccles2019biases`, https://proceedings.neurips.cc/paper/2019/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html | Reconstructed MNIST-sum task and signaling/listening objectives |
| Yuan et al., *DACOM: Learning Delay-Aware Communication for Multi-Agent Reinforcement Learning* (AAAI 2023), `yuan2023dacom`, https://arxiv.org/abs/2212.01619 | Waiting and deadline-sensitive communication |
| Jung and Lee, *Optimization framework for splitting DNN inference jobs over computing networks* (Computer Networks 2023), `jung2023splitting`, https://doi.org/10.1016/j.comnet.2023.109814 | Joint computation, transfer and waiting costs |

The cited methods motivate comparisons and mechanisms. The controlled effects in this repository are on the specified local policies and tasks, not refutations of the original methods on their native tasks.

## Third-party source

The exact Hallway environment comes from [CASEC-MACO-benchmark](https://github.com/TonghanWang/CASEC-MACO-benchmark) at commit `60687c1a5a26fc17074d5c815761d2259bdbb70f`. Its Python source is retained unchanged under `experiments/deadline_comm/environments/upstream/CASEC-MACO-benchmark/src/envs/hallway.py`. The original Apache-2.0 license text is preserved as [LICENSE.md](../deadline_comm/environments/upstream/CASEC-MACO-benchmark/LICENSE.md). The adapter loads the original class AST with isolated compatibility dependencies; importing the entire upstream SC2 environment package is not required for Hallway.

SMACv2 is attributed to Ellis et al. (NeurIPS Datasets and Benchmarks 2023), https://github.com/oxwhirl/smacv2. Runtime, maps and simulator binaries are separate dependencies. The research code has no newly assigned blanket license; third-party notices remain effective.

## Complete manuscript reference map

All 29 active entries in [references.bib](https://github.com/yeongwookung07-eng/Evaluating-Deadline-Limited-Communication-in-Multi-Agent-Reinforcement-Learning/blob/296b29a9684176f5eac332794041ce52fe25b6c1/paper/references.bib) are cited in the main paper.

| Key | Role | Primary source |
| --- | --- | --- |
| `foerster2016learning` | Channel versus NoComm comparison; learned communication background. | https://arxiv.org/abs/1605.06676 |
| `sukhbaatar2016learning` | Communication aggregation and zero-communication comparison. | https://arxiv.org/abs/1605.07736 |
| `das2019tarmac` | Targeted learned communication; background, not a reproduced baseline. | https://proceedings.mlr.press/v97/das19a.html |
| `zhang2019efficient` | Communication reduction and its traffic/performance comparison. | https://proceedings.neurips.cc/paper_files/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html |
| `zhang2020succinct` | Temporal suppression and cached message reuse. | https://proceedings.neurips.cc/paper_files/paper/2020/hash/c82b013313066e0702d58dc70db033ca-Abstract.html |
| `kim2019schedule` | Learned scheduling against RR/full access and stored messages. | https://arxiv.org/abs/1902.01554 |
| `hu2023event` | Event-triggered communication under bandwidth constraints. | https://doi.org/10.1109/TNNLS.2021.3121546 |
| `yuan2023dacom` | Delay-aware waiting decisions. | https://ojs.aaai.org/index.php/AAAI/article/view/26389 |
| `wang2020imac` | Compact message representation and scheduling; motivates holding the encoder fixed in our admission study. | https://proceedings.mlr.press/v119/wang20i.html |
| `kaul2012real` | Age-of-information background; not the origin of our particular age heuristic. | https://www.winlab.rutgers.edu/~gruteser/papers/sanjitnew.pdf |
| `rashid2018qmix` | QMIX mixing actually used by the Hallway/broadcast learner. | https://proceedings.mlr.press/v80/rashid18a.html |
| `zhu2024survey` | Communication-design taxonomy (whom/what/when) that locates the evaluation scope. | https://doi.org/10.1007/s10458-023-09633-6 |
| `lowe2019pitfalls` | Signaling versus listening and behavioral message interventions. | https://ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf |
| `ayan2022semantics` | Task-oriented versus freshness-oriented scheduling; not a MARL benchmark reproduction. | https://doi.org/10.1109/ACCESS.2022.3218410 |
| `jung2023splitting` | Joint computation, transmission and waiting cost boundary. | https://doi.org/10.1016/j.comnet.2023.109814 |
| `eccles2019biases` | Actual MNIST-sum reconstruction, signaling/listening biases and communication-use diagnostics. | https://proceedings.neurips.cc/paper_files/paper/2019/file/fe5e7cb609bdbe6d62449d61849c38b0-Paper.pdf |
| `cho2014learning` | GRU recurrent architecture actually used. | https://aclanthology.org/D14-1179/ |
| `kingma2015adam` | Adam optimizer actually used. | https://arxiv.org/abs/1412.6980 |
| `hasselt2016deepdouble` | Online action selection and target-network evaluation in Double-Q targets. | https://arxiv.org/abs/1509.06461 |
| `williams1992simple` | REINFORCE learning updates used in the external task. | https://doi.org/10.1007/BF00992696 |
| `lecun1998gradient` | MNIST digit-image data provenance, distinct from the Eccles task definition. | https://bottou.org/papers/lecun-98h |
| `wang2022casec` | MACO release provides the actual Hallway class; does not imply use of the CASEC algorithm. | https://arxiv.org/abs/2106.02886 |
| `ellis2023smacv2` | Actual SMACv2 benchmark used in feasibility/learning/replay checks; no successful multi-seed deployment claim. | https://arxiv.org/abs/2212.07489 |
| `agarwal2021deep` | Finite-training-run uncertainty in RL evaluation; does not claim use of rliable, IQM or its full benchmark aggregation protocol. | https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html |
| `efron1987better` | Bootstrap confidence intervals and the retained BCa sensitivity analysis. | https://doi.org/10.1080/01621459.1987.10478410 |
| `holm1979simple` | Holm correction used on the specified post-hoc contrast families. | https://www.jstor.org/stable/4615733 |
| `schuirmann1987comparison` | Two one-sided tests used for exploratory equivalence; post-hoc margin remains explicit. | https://doi.org/10.1007/BF01068419 |
| `hoerl1970ridge` | Ridge regression used in the held-out linear decoder. | https://doi.org/10.1080/00401706.1970.10488634 |
| `alain2016understanding` | Separately trained linear probes as diagnostics of fixed representations; our ridge decoder is an adaptation, not a reproduction of their classifiers. | https://arxiv.org/abs/1610.01644 |

| `peters2025emergent` | Journal-level review of emergent-language evaluation metrics. Main Related Work and Information Sheet distinguish that taxonomy from controlled deadline/history/execution interventions; not a native deadline baseline or a novelty proof. | https://link.springer.com/article/10.1007/s10458-025-09691-y |

Publisher sections 6.6, 7.2.1 and 8 checked on 2026-11-26. Lowe et al. remains the closest direct message-use diagnostic; Zhu and Peters provide complementary journal-level analysis.
