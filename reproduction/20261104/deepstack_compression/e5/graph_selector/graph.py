"""Fixed-shape CUDA-graph experiment; single-stream, single-request prototype."""
import torch
from kernel import fused_greedy,validate


class GraphSelector:
    def __init__(self,sim,importance,k):
        validate(sim,importance,k)
        if sim.device.type!='cuda':raise ValueError('CUDA required')
        self.sim=torch.empty_like(sim);self.importance=torch.empty_like(importance);self.k=k
        self.sim.copy_(sim);self.importance.copy_(importance)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.order,_,_=fused_greedy(self.sim,self.importance,k)

    def __call__(self,sim,importance):
        validate(sim,importance,self.k)
        if sim.shape!=self.sim.shape or importance.shape!=self.importance.shape or sim.device!=self.sim.device:
            raise ValueError('graph inputs must match captured shape and device')
        self.sim.copy_(sim);self.importance.copy_(importance)
        self.graph.replay()
        return self.order
