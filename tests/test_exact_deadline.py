import json
import numpy as np
from experiments.deadline_comm.external_deadline.mnist_deadline_attribution import expected_listener,influence,reception_mask,after_delivery,gate_status,random_admission,original_powers


def test_exact_marginal_matches_explicit_sum():
    speaker=np.array([[.25,.75],[.5,.5]])
    listener=np.array([[[1.,0.],[0.,1.]],[[.8,.2],[.1,.9]]])
    expected=np.array([[.25,.75],[.45,.55]])
    np.testing.assert_allclose(expected_listener(speaker,listener),expected)


def test_causal_responsiveness_does_not_imply_correctness():
    speaker=np.array([[.5,.5]])
    listener=np.array([[[1.,0.,0.],[0.,1.,0.]]])
    assert influence(speaker,listener)[0]>0
    assert expected_listener(speaker,listener)[0,2]==0


def test_no_response_no_influence():
    speaker=np.array([[.25,.75]])
    listener=np.array([[[.2,.8],[.2,.8]]])
    np.testing.assert_allclose(influence(speaker,listener),0,atol=1e-12)


def test_late_offer_omission_identity_and_restoration():
    ready=np.array([100.,300.,600.,900.]);mask=reception_mask(ready,500,17,100000,50)
    np.testing.assert_array_equal(mask,[True,False,False,False])
    full=np.tile([.8,.2],(4,1));zero=np.tile([.3,.7],(4,1))
    factual=after_delivery(mask,full,zero)
    # Explicit omitted-message receiver input: zero branch for all omitted offers.
    expected=np.array([[.8,.2],[.3,.7],[.3,.7],[.3,.7]])
    np.testing.assert_array_equal(factual,expected)
    assert np.isclose(full[:,0].mean()-factual[:,0].mean(),.375)
    assert reception_mask(np.zeros(4),250,17,100000,50).all()


def test_gate_fails_closed_for_missing_or_incomplete_manifest(tmp_path):
    assert not gate_status(tmp_path,[0,1,2])[0]
    (tmp_path/'summary.json').write_text(json.dumps({'status':'completed','validation_gate_pass':False}))
    assert not gate_status(tmp_path,[0,1,2])[0]


def test_gate_rechecks_all_registered_seeds(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'status':'completed','validation_gate_pass':True,'selected_updates':3000}))
    val=[{'seed':seed,'arm':'both','validation':{'sampled':.3,'expected':.4,'shuffled':.2}} for seed in range(3)]
    (tmp_path/'validation_3000.json').write_text(json.dumps(val))
    assert gate_status(tmp_path,[0,1,2])[0]
    val[2]['validation']['sampled']=.1
    (tmp_path/'validation_3000.json').write_text(json.dumps(val))
    assert not gate_status(tmp_path,[0,1,2])[0]


def test_random_admission_preserves_cost_but_does_not_hide_late_offers():
    timely=np.array([True]*100+[False]*100)
    offered=random_admission(timely,17)
    assert offered.sum()==timely.sum()
    received=offered & timely
    assert received.sum()<offered.sum()
    assert (offered & ~timely).sum()>0
    np.testing.assert_array_equal(offered,random_admission(timely,17))


def test_published_powers_for_perfect_two_symbols_per_digit():
    source=np.arange(10);listener=np.zeros(10,dtype=int)
    speaker=np.zeros((10,20));lp=np.zeros((10,20,19))
    for d in range(10): speaker[d,2*d:2*d+2]=.5
    for m in range(20): lp[:,m,m//2]=1
    result=original_powers(speaker,lp,source,listener)
    assert np.isclose(result['original_listener_power'],.9)
    assert np.isclose(result['original_speaker_power'],.9)
    np.testing.assert_allclose(expected_listener(speaker,lp)[np.arange(10),source],1)
