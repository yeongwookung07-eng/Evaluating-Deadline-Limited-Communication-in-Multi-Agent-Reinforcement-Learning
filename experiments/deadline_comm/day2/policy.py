"""Parameter-compatible inference extension of the retained BC/QMIX policy."""
from ..policy import CommPolicy
from .dac import decay_messages

class DecayedPolicy(CommPolicy):
    decay_lambda=1.0
    def q(self,hidden,cache,source,t):
        # Existing valid-count denominator is retained, not decay renormalized.
        decayed=decay_messages(cache,source,t,self.decay_lambda) if self.decay_lambda!=1 else cache
        return super().q(hidden,decayed,source,t)
