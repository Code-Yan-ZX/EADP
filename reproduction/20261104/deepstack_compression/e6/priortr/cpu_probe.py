"""Load exact PriorTR strategy/DS source modules in isolated package namespaces.
Synthetic inputs only; does not import the full model package or use GPU/weights."""
import hashlib,importlib,importlib.metadata,json,sys,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]/'e5/PriorTR/image/Qwen3-VL/visual_token_pruning'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def package(name,path):
    pkg=types.ModuleType(name);pkg.__path__=[str(path)];sys.modules[name]=pkg
package('visual_token_pruning',ROOT);package('visual_token_pruning.strategy',ROOT/'strategy');package('visual_token_pruning.model',ROOT/'model')
config=importlib.import_module('visual_token_pruning.config'); strategy=importlib.import_module('visual_token_pruning.strategy.priortr'); ds=importlib.import_module('visual_token_pruning.model.deepstack_handler')
assert not torch.cuda.is_initialized()
settings=config.VTRConfig(enabled=True,strategy='priortr',prune_layer=3,keep_tokens=256)
assert settings.query_aggregation=='question' and settings.get_prune_layers()==[3]
assert settings.get_keep_count(1024,0)==256
# Deliberately different prior vs question attention to distinguish paths.
torch.manual_seed(20261104); seq=1030; start=2; end=start+1024
attn=torch.rand((1,2,seq,seq),device='cpu')+0.01
S=strategy.PriorTRStrategy().compute_scores(attn,(start,end),settings)
P=attn[:,:,end:-1,start:end].mean(2).mean(1).squeeze(0).float(); Q=attn[:,:,end,start:end].mean(1).squeeze(0).float();P=P/(P.sum()+1e-10);Q=Q/(Q.sum()+1e-10);expected=P*torch.log((P+1e-10)/(Q+1e-10))
assert torch.allclose(S,expected,rtol=0,atol=0)
ids=strategy.PriorTRStrategy().select_tokens(S,1024,settings)
assert ids.numel()==256 and torch.equal(ids,torch.sort(S.topk(256).indices).values)
# At score layer index 2, DS 0/1 already injected, DS 2 remains pending.
stream=[torch.arange(1024*4,dtype=torch.float32).reshape(1024,4)+i*100000 for i in range(3)]
mask=torch.zeros((1,seq),dtype=torch.bool);mask[:,start:end]=True
keep=torch.cat([torch.arange(start),ids+start,torch.arange(end,seq)])
out,newmask=ds.DeepStackSyncHandler(num_deepstack_layers=3).sync_after_pruning(stream,mask,ids,keep,current_layer_idx=2)
assert torch.equal(out[0],stream[0]) and torch.equal(out[1],stream[1]) and torch.equal(out[2],stream[2].index_select(0,ids))
assert torch.equal(newmask,mask.index_select(1,keep)) and int(newmask.sum())==256
paths=[ROOT/'config.py',ROOT/'strategy/base.py',ROOT/'strategy/priortr.py',ROOT/'model/deepstack_handler.py',Path(__file__)]
print(json.dumps({'status':'passed','torch_version':torch.__version__,'transformers_version':importlib.metadata.version('transformers'),'cuda_initialized':torch.cuda.is_initialized(),'num_visual_in':1024,'kept':len(ids),'prune_layer':3,'score_layer_zero_based':2,'ds_uninjected_index2_rows':out[2].shape[0],'scope':'synthetic strategy and DS handler source only; no full model/cache/decode validation', 'source_sha256':{str(p):digest(p) for p in paths}},indent=2))
