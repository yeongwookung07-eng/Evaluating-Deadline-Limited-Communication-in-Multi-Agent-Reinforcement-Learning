# Attribution and scope

Research manuscript authors: Yeong-woo Kung and Young-guk Ha. The evaluation design, communication harness, controls and analyses are this project's research contribution; underlying algorithms and benchmarks are credited separately.

* **QMIX**: Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning* (ICML 2018), bibliography key `rashid2018qmix`. `models/qmix` is the local implementation used here, not a claim of native baseline reproduction.
* **Hallway**: the exact `HallwayEnv` source is retained at `external/upstreams/CASEC-MACO-benchmark/src/envs/hallway.py`, from https://github.com/TonghanWang/CASEC-MACO-benchmark at commit `60687c1a5a26fc17074d5c815761d2259bdbb70f`. Its upstream Apache-2.0 LICENSE is included adjacent to the source tree. The adapter loads the original class AST with local compatibility dependencies. The upstream source is unchanged. This provenance describes the benchmark source; no CASEC algorithm superiority is claimed.
* **SMACv2**: Ellis et al., SMACv2 (NeurIPS Datasets and Benchmarks 2023); upstream https://github.com/oxwhirl/smacv2 pinned in the setup YAML. StarCraft II/runtime/maps are not included. Failed learning gates and technical-only results remain explicit.
* **Elsevier CAS**: `cas-dc.cls`, `cas-common.sty`, and `cas-model2-names.bst` retain their original license notices. `cas-common.sty` has the previously documented keyword-box `\hss` adjustment. The paper uses the supplied CAS template; this is not journal approval.

The manuscript bibliography is the reference record. Original research code has no newly assigned blanket license in this export; third-party notices remain effective. The owner can select a project license separately without relicensing third-party material.
