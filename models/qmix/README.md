# QMIX comparison slot

STATUS: CORE IMPLEMENTATION COMPLETE; 80-STEP PIPELINE SMOKE PASSED

`QMIX` uses shared recurrent agent utilities, decentralized greedy action selection,
and a two-layer centralized state-conditioned monotonic mixer. Absolute hypernetwork
weights enforce nonnegative dependence on each selected agent utility. The generic
recurrent double-Q learner supplies complete-episode replay and target updates.

The implementation was independently written against Rashid et al., *QMIX:
Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*,
ICML 2018, and cross-checked against the official PyMARL mixer structure. It is a
minimal local implementation, not a claim to reproduce published SMAC scores or
the full PyMARL training stack.

Experiment 021 ran the existing 80-step pipeline check: 20 episodes, 17 updates,
finite loss decreasing from 0.2336 to 0.0876, and nonzero parameter change. Final
return 0.125/4 and zero episode success are execution diagnostics only. Config:
`configs/experiments/qmix_smoke.yaml`; run:
`qmix_smoke_021_seed0_20260922T093508138432Z_0737b1a3`.

## Official archive integrity audit — 2026-11-16 KST

기존에는 로컬 재구현만 있었으며, 공식 PyMARL 52개 파일을 새로 수집했다. 공식 원격 archive를 다시 받아 파일 SHA-256으로 대조했다.
새 원본: [official/20261116_c971afdceb34/](official/20261116_c971afdceb34/), 고정 버전 `c971afdceb34635d31b778021b0ef90d7af51e86`.
[검증 manifest](official/20261116_c971afdceb34_manifest.json) · [전체 판정과 제한](../SOURCE_INTEGRITY_20261116.md).
기존 upstream과 로컬 구현/실험은 보존했다. 새 원본 설치·실행·학습·논문 점수 재현은 수행하지 않았다.
