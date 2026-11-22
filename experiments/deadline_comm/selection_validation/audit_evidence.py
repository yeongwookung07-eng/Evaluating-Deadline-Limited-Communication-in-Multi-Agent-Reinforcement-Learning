"""Read-only recomputation of completed evidence; never trains or changes inputs."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import numpy as np


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def holm(rows):
    previous = 0.0
    for rank, r in enumerate(sorted(rows, key=lambda x: x['p_exact'])):
        previous = max(previous, min(1.0, (len(rows)-rank)*r['p_exact']))
        r['p_holm'] = previous


def main(config):
    cfg = json.loads(Path(config).read_text())
    src, out = Path(cfg['source']), Path(cfg['output'])
    if out.exists():
        raise FileExistsError(f'Preserve prior audit: {out}')
    rows = read_csv(src/'by_seed.csv')
    aggregate = read_csv(src/'aggregate.csv')
    contrasts = read_csv(src/'paired_ci.csv')
    assert len(rows) == 720 and len(aggregate) == 72 and len(contrasts) == 60
    index = {}
    reset_sets = {}
    hashes = {}
    episodes_count = 0
    for r in rows:
        key = (r['family'], r['condition'], r['arm'], int(r['seed']))
        assert key not in index
        index[key] = float(r['mean_return'])
        ep = src/f"{r['family']}_seed{r['seed']}_{r['arm']}"/f"eval_{r['deployment']}_{r['condition']}"/'episodes.csv'
        episodes = read_csv(ep)
        assert len(episodes) == 50
        assert abs(statistics.mean(float(e['episode_return']) for e in episodes)-index[key]) < 1e-10
        resets = tuple(e['seed'] for e in episodes)
        assert len(set(resets)) == 50
        pairing_key = (r['family'], r['condition'], r['seed'])
        assert reset_sets.setdefault(pairing_key, resets) == resets
        episodes_count += len(episodes)
        hashes[str(ep)] = hashlib.sha256(ep.read_bytes()).hexdigest()
    for r in aggregate:
        vals = [index[(r['family'],r['condition'],r['arm'],s)] for s in cfg['seeds']]
        assert abs(statistics.mean(vals)-float(r['mean_return'])) < 1e-10
        assert abs(statistics.stdev(vals)-float(r['mean_return_sd'])) < 1e-10
    audited = []
    signs = np.array(list(itertools.product([-1,1], repeat=10)))
    for r in contrasts:
        delta = np.array([index[(r['family'],r['condition'],r['left'],s)]-index[(r['family'],r['condition'],r['right'],s)] for s in cfg['seeds']])
        assert np.allclose(delta, json.loads(r['paired_differences']), atol=1e-12)
        draws = delta[np.random.default_rng(0).integers(0,10,size=(10000,10))].mean(axis=1)
        lo, hi = np.quantile(draws,[.025,.975])
        assert np.allclose([delta.mean(),lo,hi],[float(r[k]) for k in ['mean_difference','ci95_low','ci95_high']],atol=1e-10)
        p = float(np.mean(np.abs((signs*delta).mean(axis=1)) >= abs(delta.mean())-1e-12))
        audited.append(dict(family=r['family'],condition=r['condition'],left=r['left'],right=r['right'],difference=float(delta.mean()),ci95=[float(lo),float(hi)],p_exact=p))
    utility = [r for r in audited if r['left']=='ideal_full' and r['right']=='no_comm' and r['family']=='scratch_'+r['condition']]
    selector = [r for r in audited if r['left']=='dac' and r['right']=='rr_cap']
    assert len(utility)==3 and len(selector)==12
    holm(utility)
    holm(selector)
    result = dict(status='completed',episode_files_verified=len(rows),episode_returns_verified=episodes_count,aggregates_verified=len(aggregate),bootstrap_contrasts_verified=len(contrasts),paired_reset_groups=len(reset_sets),utility=utility,selector=selector,limitations=cfg['limitations'],statistical_assumptions=cfg['statistics'])
    out.mkdir(parents=True)
    (out/'registration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main(sys.argv[1])
