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

The exact Hallway environment comes from [CASEC-MACO-benchmark](https://github.com/TonghanWang/CASEC-MACO-benchmark) at commit `60687c1a5a26fc17074d5c815761d2259bdbb70f`. Its Python source is retained unchanged under `external/upstreams/CASEC-MACO-benchmark/src/envs/hallway.py`. The original Apache-2.0 license text is preserved as [LICENSE.md](../external/upstreams/CASEC-MACO-benchmark/LICENSE.md). The adapter loads the original class AST with isolated compatibility dependencies; importing the entire upstream SC2 environment package is not required for Hallway.

SMACv2 is attributed to Ellis et al. (NeurIPS Datasets and Benchmarks 2023), https://github.com/oxwhirl/smacv2. Runtime, maps and simulator binaries are separate dependencies. The research code has not been assigned a new blanket license by this export; third-party notices remain effective.
