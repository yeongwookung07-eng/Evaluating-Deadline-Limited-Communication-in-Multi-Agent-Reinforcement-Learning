import pytest
from experiments.deadline_comm.transport import SerialLink

def test_arrival_rotation_cutoff_boundary():
    link=SerialLink(1e9,latency_ns=3,header_bytes=2)
    result=link.schedule({0:8,1:8,2:8},3,1,5,28)
    assert [r['sender'] for r in result.messages]==[1,2,0]
    assert [r['arrival_ns'] for r in result.messages]==[18,28,38]
    assert [r['accepted'] for r in result.messages]==[True,True,False]
    assert result.wait_ns==23
    assert result.wire_bytes==30

def test_zero_and_expired():
    link=SerialLink(1e9)
    assert link.schedule({},3,0,0,10).wait_ns==0
    r=link.schedule({0:8},3,0,20,10)
    assert r.wait_ns==0 and not r.messages[0]['accepted']

@pytest.mark.parametrize('kwargs',[{'bandwidth_bytes_s':0},{'bandwidth_bytes_s':1,'latency_ns':-1}])
def test_invalid(kwargs):
    with pytest.raises(ValueError): SerialLink(**kwargs)
