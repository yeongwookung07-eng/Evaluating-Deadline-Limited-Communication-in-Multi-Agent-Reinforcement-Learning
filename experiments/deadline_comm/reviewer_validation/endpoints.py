"""Delivery endpoint interventions; all upstream modules unchanged."""
from copy import deepcopy
from ..day2.followup_harness import Day2Harness as OriginalHarness


from ..transport import SerialLink

class IdealLink(SerialLink):
    def schedule(self,payload_sizes,n_agents,t,ready_ns,cutoff_ns):
        # Ideal delivery reference ignores reception cutoff; CPU deadline misses
        # are still recorded by the harness, not hidden by a larger D_max.
        return super().schedule(payload_sizes,n_agents,t,ready_ns,float('inf'))

class NeverSend:
    last=None
    def __init__(self,n):self.last=[None]*n
    def decide(self,*args,**kwargs):return False
    def record_attempt(self,*args):raise AssertionError('No-Comm attempted transmission')


class EndpointHarness(OriginalHarness):
    def __new__(cls,model,method,cfg,seed,measured=True):
        if cfg.get('selector'):
            from ..selection_validation.harness import SelectionHarness
            return SelectionHarness(model,cfg['selector'],cfg,seed,measured)
        return super().__new__(cls)

    def __init__(self,model,method,cfg,seed,measured=True):
        endpoint=cfg.get('endpoint')
        if endpoint in ('no_comm','ideal_full'):
            cfg=deepcopy(cfg); cfg['decay_lambda']=1.
            super().__init__(model,'full',cfg,seed,measured)
            if endpoint=='no_comm':self.gate=NeverSend(model.n_agents)
            else:
                # Exactly zero serialization/propagation delay, but real CPU work.
                self.link=IdealLink(float('inf'),0,self.link.header_bytes)
        else:super().__init__(model,method,cfg,seed,measured)
        self.endpoint=endpoint
    def step(self,*args):
        action,row,messages=super().step(*args)
        if self.endpoint:
            row['method']=self.endpoint
            for m in messages:m['method']=self.endpoint
        if self.endpoint=='no_comm':
            assert row['K_attempt']==row['K_delivered']==0
            assert (self.source<0).all()
        return action,row,messages
