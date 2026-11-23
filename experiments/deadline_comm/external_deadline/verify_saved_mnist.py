"""Independent saved-array arithmetic audit; no checkpoint inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(root):
    root=Path(root);c=json.loads((root/'registration.json').read_text());r=json.loads((root/'results.json').read_text());checks=0
    hashes=json.loads((root/'input_hashes.json').read_text())
    for path,digest in hashes['checkpoints'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for path,digest in hashes['data'].items():assert hashlib.sha256((Path(c['data'])/path).read_bytes()).hexdigest()==digest
    for seed in c['seeds']:
      with np.load(root/f'seed{seed}_episodes.npz',allow_pickle=False) as a:
        ps=a['speaker_probabilities'];lp=a['listener_by_message'];zero=a['zero_probabilities'];n=len(ps);y=a['target'];j=np.arange(n)
        np.testing.assert_allclose(ps.sum(-1),1,atol=1e-6);np.testing.assert_allclose(lp.sum(-1),1,atol=1e-6);np.testing.assert_allclose(zero.sum(-1),1,atol=1e-6)
        # Different summation expression from the evaluator's einsum.
        full=sum(ps[:,m,None]*lp[:,m,:] for m in range(20));value=full[j,y]
        np.testing.assert_allclose(value,a['full_expected'],atol=1e-12)
        shuffled=sum(ps[a['shuffle_permutation'],m,None]*lp[:,m,:] for m in range(20));np.testing.assert_allclose(shuffled[j,y],a['shuffle_expected'],atol=1e-12)
        for d in c['deadlines_us']:
          mask=a['delivered_'+str(d)+'us'];calc=a['ready_us']+220<=d;np.testing.assert_array_equal(mask,calc)
          offers=a['random_offers_'+str(d)+'us'];assert offers.sum()==mask.sum()
          factual=np.stack([full[i] if mask[i] else zero[i] for i in range(n)])
          random=np.stack([full[i] if offers[i] and mask[i] else zero[i] for i in range(n)])
          np.testing.assert_allclose(factual[j,y],a[f'expected_{d}us'],atol=1e-12);np.testing.assert_allclose(random[j,y],a[f'random_expected_{d}us'],atol=1e-12)
          row=next(x for x in r['deadline_records'] if x['seed']==seed and x['deadline_us']==d)
          assert row['admit_offered_bytes']==row['random_offered_bytes']==17*offers.mean()
          assert np.isclose(row['informed_minus_random'],(factual[j,y]-random[j,y]).mean(),atol=1e-12)
          assert row['max_full_admit_distribution_error']==0
          checks+=n
    result=dict(passed=True,matched_episode_deadline_checks=checks,seeds=c['seeds'],exact_marginal_verified=True,input_hashes_verified=True,equal_mean_offered_budget_verified=True,scope='Saved arrays independently recomputed; structural input identity, not new rollout evidence')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
