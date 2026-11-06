"""Independent plain-Python Qwen3 one-image, no-padding 32x32 P3 mapping.

Valid only for one static image with a contiguous 1024-token image block, a
single immediately preceding vision-start token and no video. Input values and
actual prefill tensors must still be separately bound to the real processor.
"""
def reconstruct(ids,image_token,vision_start,video_token,selected):
 if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids):raise ValueError('invalid input ids')
 if any(type(x) is not int for x in (image_token,vision_start,video_token)) or video_token in ids:raise ValueError('video/invalid token config')
 locations=[i for i,x in enumerate(ids) if x==image_token]
 if len(locations)!=1024 or locations!=list(range(locations[0],locations[0]+1024)) or locations[0]<1 or ids[locations[0]-1]!=vision_start or ids.count(vision_start)!=1:raise ValueError('not one contiguous static image')
 if not isinstance(selected,list) or not selected or selected!=sorted(set(selected)) or any(type(x) is not int or not 0<=x<1024 for x in selected):raise ValueError('invalid selected visual indices')
 start=locations[0];n=len(ids);end=start+1024
 prefix=list(range(start));tail=list(range(start+32,start+32+n-end))
 visual_t=[start]*1024;visual_h=[start+i//32 for i in range(1024)];visual_w=[start+i%32 for i in range(1024)]
 axes=[prefix+v+tail for v in (visual_t,visual_h,visual_w)];kept=list(range(start))+[start+i for i in selected]+list(range(end,n));position=[[[axis[i] for i in kept]] for axis in axes]
 maximum=max(max(axis) for axis in axes);return {'original_sequence_length':n,'image_start':start,'position_ids':position,'original_max_position':maximum,'rope_delta':[[maximum+1-n]],'retained_ids':kept}
