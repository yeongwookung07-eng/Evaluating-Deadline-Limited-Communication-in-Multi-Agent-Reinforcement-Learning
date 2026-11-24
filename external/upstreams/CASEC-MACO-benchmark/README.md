# Upstream Hallway environment

The retained `src/envs/hallway.py` is the unchanged upstream Hallway source used by this analysis. Source: https://github.com/TonghanWang/CASEC-MACO-benchmark, commit `60687c1a5a26fc17074d5c815761d2259bdbb70f`. The original license is preserved verbatim in [LICENSE.md](LICENSE.md).

The experiment adapter loads `HallwayEnv` from its AST, supplies private NumPy/Python RNGs and preserves transition/reward behavior. The five-agent task has two groups that must coordinate within-group arrival while avoiding the simultaneous-group penalty. The current study uses it for controlled communication comparisons; no CASEC algorithm result is claimed.
