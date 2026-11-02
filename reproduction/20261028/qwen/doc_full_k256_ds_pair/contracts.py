"""Pure shape and selection contracts, no tensor runtime required."""
def mask_span(input_ids, image_token_id, retained=256):
    positions=[i for i,x in enumerate(input_ids) if x==image_token_id]
    if len(positions)!=1024 or positions != list(range(positions[0],positions[0]+1024)):
        raise ValueError('expected one contiguous 1024-token image span')
    n=len(input_ids)-1024+retained
    return [positions[0]<=i<positions[0]+retained for i in range(n)]

def compare_selection(ordered_ids, spatial_ids):
    if len(ordered_ids)!=256 or len(spatial_ids)!=256 or len(set(ordered_ids))!=256:
        raise ValueError('expected 256 unique actual author selections')
    if any(type(x) is not int or not 0<=x<1024 for x in ordered_ids) or spatial_ids!=sorted(ordered_ids):
        raise ValueError('actual author IDs do not match valid spatial order')
    return True
