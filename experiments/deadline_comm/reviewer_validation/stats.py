"""Paired bootstrap on independent seed means, never pseudoreplicated episodes."""
import numpy as np


def paired_bootstrap(left, right, resamples=10000, seed=0):
    if set(left)!=set(right) or not left:
        raise ValueError('Require identical nonempty training seed sets')
    keys=sorted(left)
    delta=np.array([float(left[k])-float(right[k]) for k in keys])
    if not np.isfinite(delta).all():raise ValueError('Nonfinite return')
    rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),size=(resamples,len(delta)))].mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(n_training_seeds=len(keys),seeds=keys,mean_difference=float(delta.mean()),
                ci95_low=float(lo),ci95_high=float(hi),paired_differences=delta.tolist(),
                bootstrap_resamples=resamples,inference='descriptive' if len(keys)<10 else 'conditional_seed_bootstrap',
                multiple_comparison_adjustment=False)
