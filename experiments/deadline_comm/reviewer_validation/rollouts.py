"""Scoped endpoint/content interventions, retaining validity/age features."""
from contextlib import ExitStack
from unittest.mock import patch
from ..day2 import followup_harness as hm
from .endpoints import EndpointHarness


def rollout(env,model,method,cfg,seed,measured=True):
    import torch
    original=model.q
    def altered(hidden,cache,source,t):
        if cfg.get('message_intervention')=='zero':cache=torch.zeros_like(cache)
        elif cfg.get('message_intervention')=='rotate':cache=cache.roll(1,dims=1)
        return original(hidden,cache,source,t)
    with ExitStack() as stack:
        stack.enter_context(patch.object(hm,'Day2Harness',EndpointHarness))
        if cfg.get('message_intervention'):stack.enter_context(patch.object(model,'q',altered))
        return hm.rollout(env,model,method,cfg,seed,measured)
