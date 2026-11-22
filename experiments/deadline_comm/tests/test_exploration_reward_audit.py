from experiments.deadline_comm.exploration_reward_audit import DiagnosticHallway,terminal_cause

def test_one_group_failure_does_not_end_episode():
    env=DiagnosticHallway(group_ids=[0,0,1,1,1]);env.reset(seed=0)
    env.env.state_n[:]=[1,2,3,3,3]
    _,reward,term,trunc,_=env.step([1,0,0,0,0])
    assert reward==0 and not term and not trunc
    assert env.events[-1]['async_failed_groups']==[0]
    for _ in range(env.episode_length-1):env.step([0]*5)
    assert terminal_cause(env)=='time_limit'

def test_both_group_failures_terminate():
    env=DiagnosticHallway(group_ids=[0,0,1,1,1]);env.reset(seed=0)
    env.env.state_n[:]=[1,2,1,2,3]
    _,_,term,trunc,_=env.step([1]*5)
    assert term and not trunc and terminal_cause(env)=='all_groups_failed'
    assert env.events[-1]['async_failed_groups']==[0,1]

def test_simultaneous_penalty_is_event_not_immediate_terminal():
    env=DiagnosticHallway(group_ids=[0,0,1,1,1]);env.reset(seed=0)
    env.env.state_n[:]=[1]*5
    _,reward,term,trunc,_=env.step([1]*5)
    assert reward==-10 and not term and not trunc
    assert env.events[-1]['simultaneous_group_penalty']
    for _ in range(env.episode_length-1):env.step([0]*5)
    assert terminal_cause(env)=='time_limit'
