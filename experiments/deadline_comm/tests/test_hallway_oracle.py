from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.hallway_oracle_check import check_episode

GROUPS=[0,0,1,1,1]

def test_straight_can_fail_with_unequal_distances():
    env=Hallway(group_ids=GROUPS)
    row,_=check_episode(env,GROUPS,'straight',0,[1,2,1,2,3])
    assert row['episode_return']==0 and not row['success']
    row,trace=check_episode(env,GROUPS,'synchronized_staggered',0,[1,2,1,2,3])
    assert row['episode_return']==20 and row['success']
    assert [s['reward'] for s in trace if s['reward']>0]==[10,10]

def test_simultaneous_groups_penalized_but_staggered_succeeds():
    env=Hallway(group_ids=GROUPS)
    row,trace=check_episode(env,GROUPS,'straight',0,[1,1,1,1,1])
    assert trace[0]['reward']==-10 and row['episode_return']==-10
    assert not row['success']
    row,_=check_episode(env,GROUPS,'synchronized_staggered',0,[1,1,1,1,1])
    assert row['episode_return']==20 and row['success'] and row['length']==2

def test_straight_succeeds_when_groups_are_already_scheduled():
    row,_=check_episode(Hallway(group_ids=GROUPS),GROUPS,'straight',0,[1,1,2,2,2])
    assert row['episode_return']==20 and row['success']
