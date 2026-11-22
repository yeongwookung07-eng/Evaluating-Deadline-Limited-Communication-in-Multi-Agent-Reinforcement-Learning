"""FIFO persistent service with stochastic erasures and bounded immediate ARQ.

Logical seconds, independent packet-keyed random losses, non-preemptive service.
Retries append to FIFO; propagation does not occupy the serializer. ACK/control
bytes are excluded explicitly. Receive cutoffs do not erase physical backlog.
An arrival after cutoff can be consumed at the next decision with original age.
"""
from collections import deque
import hashlib
import heapq
import math


class QueuedLink:
    def __init__(self,bandwidth_bytes_s,loss_probability=0.,max_retries=0,latency_s=0.,seed=0,queue_limit=10000):
        if bandwidth_bytes_s<=0 or not 0<=loss_probability<=1 or max_retries<0 or latency_s<0 or queue_limit<1:
            raise ValueError('Invalid network')
        self.bandwidth=bandwidth_bytes_s;self.loss=loss_probability;self.max_retries=max_retries
        self.latency=latency_s;self.seed=seed;self.limit=queue_limit
        self.time=0.;self.queue=deque();self.active=None;self.arrivals=[];self.serial=0
        self.wire=0;self.retransmitted_wire=0;self.failed=0;self.overflow=0
    def lost(self,p):
        key=f"{self.seed}:{p['sender']}:{p['source_step']}:{p['attempt']}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/2**64<self.loss
    def start(self):
        if self.active is None and self.queue:
            p=self.queue.popleft();self.active=(self.time+p['wire_bytes']/self.bandwidth,p)
            self.wire+=p['wire_bytes']
            if p['attempt']:self.retransmitted_wire+=p['wire_bytes']
    def advance(self,until):
        if until<self.time:raise ValueError('Time cannot move backwards')
        self.start()
        while self.active is not None and self.active[0]<=until:
            end,p=self.active;self.time=end;self.active=None
            if self.lost(p):
                if p['attempt']<self.max_retries:
                    retry=dict(p,attempt=p['attempt']+1)
                    if len(self.queue)<self.limit:self.queue.append(retry)
                    else:self.overflow+=1
                else:self.failed+=1
            else:
                self.serial+=1
                heapq.heappush(self.arrivals,(end+self.latency,self.serial,dict(p,arrival_s=end+self.latency)))
            self.start()
        self.time=until
        delivered=[]
        while self.arrivals and self.arrivals[0][0]<=until:delivered.append(heapq.heappop(self.arrivals)[2])
        return delivered
    def submit(self,sender,source_step,wire_bytes,payload=None):
        if wire_bytes<=0:raise ValueError('Empty packet')
        if len(self.queue)>=self.limit:self.overflow+=1;return False
        self.queue.append(dict(sender=sender,source_step=source_step,wire_bytes=wire_bytes,payload=payload,attempt=0))
        self.start();return True
    def snapshot(self):
        active_remaining=max(0.,self.active[0]-self.time)*self.bandwidth if self.active else 0.
        return dict(wire_bytes_started=self.wire,retransmitted_wire_bytes=self.retransmitted_wire,
                    backlog_packets=len(self.queue)+int(self.active is not None),
                    backlog_bytes=sum(p['wire_bytes'] for p in self.queue)+active_remaining,
                    failed_packets=self.failed,queue_overflow=self.overflow)
