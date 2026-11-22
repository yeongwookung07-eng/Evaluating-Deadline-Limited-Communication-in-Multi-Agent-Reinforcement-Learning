"""Serial shared broadcast link. B in bytes/s; all times in ns.

All attempted packets are charged in full (including late packets); the link
is flushed at the step boundary. No retransmission or inter-step backlog.
"""
from dataclasses import dataclass
import math

@dataclass
class LinkResult:
    messages: list
    wait_ns: float
    wire_bytes: int

class SerialLink:
    def __init__(self, bandwidth_bytes_s, latency_ns=0, header_bytes=16):
        if bandwidth_bytes_s <= 0 or latency_ns < 0 or header_bytes < 0:
            raise ValueError("Invalid link settings")
        self.bandwidth = float(bandwidth_bytes_s)
        self.latency_ns, self.header_bytes = latency_ns, header_bytes
    def schedule(self, payload_sizes, n_agents, t, ready_ns, cutoff_ns):
        if n_agents < 1 or any(i < 0 or i >= n_agents or b < 0 for i,b in payload_sizes.items()):
            raise ValueError("Invalid senders/sizes")
        order = sorted(payload_sizes, key=lambda i: (i-t) % n_agents)
        service, rows, wire = 0., [], 0
        for j, i in enumerate(order):
            size = payload_sizes[i] + self.header_bytes
            start = service
            service += size / self.bandwidth * 1e9
            arrival = ready_ns + service + self.latency_ns
            accepted = arrival <= cutoff_ns
            rows.append(dict(sender=i, receiver_scope="broadcast_except_self", queue_order=j,
                payload_bytes=payload_sizes[i], wire_bytes=size, queue_ns=start,
                arrival_ns=arrival, accepted=accepted,
                drop_reason=None if accepted else "receive_cutoff"))
            wire += size
        wait = min(service + self.latency_ns, max(0., cutoff_ns-ready_ns)) if order else 0.
        return LinkResult(rows, wait, wire)
