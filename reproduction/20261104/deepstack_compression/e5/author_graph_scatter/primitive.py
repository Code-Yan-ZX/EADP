"""Isolated capture hypotheses; CUDA entry only after worker admission."""
import torch

NAMES=('index_assign','scatter_assign','index_read','boolean_mask_assign')

def apply(name,mask,index,batch,similarity,gain):
    if name=='index_assign':
        mask[batch,index]=True
        return mask
    if name=='scatter_assign':
        mask.scatter_(1,index.unsqueeze(1),True)
        return mask
    if name=='index_read':
        return similarity[batch,index,:]
    if name=='boolean_mask_assign':
        gain[mask]=-1e9
        return gain
    raise ValueError('unknown primitive')

def run(name,emit):
    if name not in NAMES:raise ValueError('unknown primitive')
    mask=torch.zeros((1,1024),dtype=torch.bool,device='cuda')
    index=torch.tensor([17],dtype=torch.long,device='cuda');batch=torch.arange(1,device='cuda')
    sim=torch.arange(1024*1024,dtype=torch.float32,device='cuda').reshape(1,1024,1024)
    gain=torch.ones((1,1024),dtype=torch.float32,device='cuda')
    if name=='boolean_mask_assign':mask[0,17]=True
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=apply(name,mask,index,batch,sim,gain)
    graph.replay();torch.cuda.synchronize()
    if name in ('index_assign','scatter_assign'):
        expected=torch.zeros_like(mask);expected[0,17]=True
    elif name=='index_read':expected=sim[:,17,:]
    else:
        expected=torch.ones_like(gain);expected[0,17]=-1e9
    equal=bool(torch.equal(out,expected))
    emit({'event':'primitive_compared','primitive':name,'matched':equal})
    if not equal:raise ValueError('primitive result differs')
