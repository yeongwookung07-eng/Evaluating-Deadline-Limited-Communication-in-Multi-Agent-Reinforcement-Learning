# external_biases_mnist_parallel.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/external_biases_mnist_parallel.yaml`.

SHA256: `7a6e61bcfc7cd2a3b80bd32f058232a30fcabd0e23dfab2dbb9c9ab316782a87`

```yaml
base_config: experiments/deadline_comm/configs/external_biases_mnist.yaml
output: experiments/deadline_comm/external_biases/runs/mnist_sum_parallel_20261123_001
source: experiments/deadline_comm/external_biases/runs/mnist_sum_20261123_001
handoff_update: 1500
milestones: [3000, 6000, 12000, 24000]
workers: 6
wall_seconds: 1800
handoff_wait_seconds: 900
reason: Prospective execution amendment to parallel independent CPU cells; preserve original model/objective/RNG and continue all seeds equally. The 12000/24000 stages occur only if all-seed validation gate fails earlier; no test-based selection or hyperparameter change.
gate: Every both-bias seed sampled validation reward >0.2 and expected validation reward > shuffled.
interpretation: Bounded paper-based reconstruction, not author code or original score reproduction. Failure or incompleteness blocks external deadline claims.
```
