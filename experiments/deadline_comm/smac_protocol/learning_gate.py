"""Common learning criterion across the recorded training seeds."""
def gate_pass(records,spec):
    return len(records)==len(spec['seeds']) and all(r['win_rate']>=spec['learning_gate_win_rate'] for r in records) and sum(r['win_rate'] for r in records)>0
