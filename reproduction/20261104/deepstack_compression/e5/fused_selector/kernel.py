"""Experimental fused facility-gain prototype, not a deployed selector.

Floating-point reductions may differ from the author's PyTorch implementation.
All candidate output comparisons must remain observable, and a mismatch is a
failure of this prototype for that input, never silently replaced.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gain_kernel(sim, weight, coverage, selected, gains, N:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    valid=col<N
    s=tl.load(sim+row*N+col,mask=valid,other=0.0)
    c=tl.load(coverage+col,mask=valid,other=0.0)
    w=tl.load(weight+col,mask=valid,other=0.0)
    value=tl.sum((tl.maximum(s,c)-c)*w,axis=0)
    taken=tl.load(selected+row)
    tl.store(gains+row,tl.where(taken,-1.0e9,value))


def validate(sim, importance, k):
    if not torch.is_tensor(sim) or not torch.is_tensor(importance): raise ValueError('tensor inputs required')
    if (sim.ndim!=2 or sim.shape[0]!=sim.shape[1] or importance.shape!=(sim.shape[0],)
            or sim.dtype!=torch.float32 or importance.dtype!=torch.float32
            or not sim.is_contiguous() or not importance.is_contiguous() or sim.device!=importance.device
            or type(k) is not int or not 1<=k<=sim.shape[0]):
        raise ValueError('invalid shapes, dtype, layout, device or budget')


def fused_greedy(sim, importance, k):
    validate(sim,importance,k)
    if sim.device.type!='cuda': raise ValueError('CUDA required')
    n=sim.shape[0]
    coverage=torch.zeros(n,device=sim.device,dtype=torch.float32)
    selected=torch.zeros(n,device=sim.device,dtype=torch.bool)
    gains=torch.empty(n,device=sim.device,dtype=torch.float32)
    order=torch.empty(k,device=sim.device,dtype=torch.long)
    for t in range(k):
        _gain_kernel[(n,)](sim,importance,coverage,selected,gains,N=n,BLOCK=triton.next_power_of_2(n))
        best=torch.argmax(gains).reshape(1)
        order[t].copy_(best[0])
        row=torch.index_select(sim,0,best).squeeze(0)
        coverage=torch.maximum(coverage,row)
        selected.scatter_(0,best,True)
    return order,coverage,selected
