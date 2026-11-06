"""Per-image 1024-to-512 input identity and independent mRoPE check."""
import hashlib
def meta(torch,t):
 t=t.detach().contiguous().cpu()
 return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
def span(torch,inputs,image_token_id,count):
 ids=inputs['input_ids']
 if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('single input required')
 index=torch.where(ids[0]==image_token_id)[0]
 if index.numel()!=count:raise ValueError('visual count')
 start=int(index[0])
 if not torch.equal(index,torch.arange(start,start+count,device=index.device)):raise ValueError('visual span')
 return ids[0],start
def check(torch,original,low,config,ref,validate_original,rope_index):
 validate_original(torch,original,ref['processor'],config.image_token_id,ref['image_start'])
 if config.vision_config.spatial_merge_size!=2 or low['image_grid_thw'].tolist()!=[[1,32,32]] or not bool(torch.all(low['attention_mask']==1)):raise ValueError('low-resolution grid, merge or padding')
 old,os=span(torch,original,config.image_token_id,1024);new,ns=span(torch,low,config.image_token_id,256)
 if os!=ns or not torch.equal(old[:os].cpu(),new[:ns].cpu()) or not torch.equal(old[os+1024:].cpu(),new[ns+256:].cpu()):raise ValueError('nonvisual prompt changed')
 pos,delta=rope_index(low)
 prefix=torch.arange(ns).view(1,-1).expand(3,-1);h=torch.arange(16).view(-1,1).expand(16,16).reshape(-1);w=torch.arange(16).view(1,-1).expand(16,16).reshape(-1);visual=torch.stack((torch.zeros(256,dtype=torch.int64),h,w))+ns
 suffix_n=int(new.numel()-ns-256);suffix=torch.arange(ns+16,ns+16+suffix_n).view(1,-1).expand(3,-1);expected=torch.cat((prefix,visual,suffix),dim=1).unsqueeze(1)
 if not torch.equal(pos.detach().cpu(),expected) or int(delta.item())!=int(pos.max().item()+1-new.numel()):raise ValueError('mRoPE mismatch')
 return {'original_visual_tokens':1024,'low_visual_tokens':256,'grid':low['image_grid_thw'].tolist(),'image_start':ns,'text_prefix_suffix_identical':True,'low_processor':{k:meta(torch,low[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'position_ids':meta(torch,pos),'rope_delta':delta.tolist(),'position_formula_verified':True}
