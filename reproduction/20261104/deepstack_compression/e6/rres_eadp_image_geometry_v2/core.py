import hashlib,io,math
from PIL import Image

def dimensions(data,expected):
 if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('original image SHA mismatch')
 with Image.open(io.BytesIO(data)) as im:w,h=im.size
 if type(w) is not int or type(h) is not int or w<=0 or h<=0:raise ValueError('invalid raw dimensions')
 return w,h
def groups(w,h):
 if type(w) is not int or type(h) is not int or w<=0 or h<=0:raise ValueError('dimensions')
 aspect='portrait' if 5*w<4*h else ('landscape' if 4*w>5*h else 'middle')
 return aspect,('small' if w*h<1000000 else 'large')
def unique(rows):
 out={}
 for x in rows:
  k=(x['task'],str(x['qid']),x['group_id'])
  if k in out:raise ValueError('duplicate key')
  out[k]=x
 return out
def summary(rows):
 n=len(rows);a=sum(x['rres'] for x in rows);b=sum(x['eadp'] for x in rows)
 result={'all':{'questions':n,'image_groups':len({x['group_id'] for x in rows}),'rres_mean_pct':100*a/n if n else None,'eadp_mean_pct':100*b/n if n else None,'difference_pp':100*(a-b)/n if n else None}}
 for typ,labels in (('aspect',('portrait','middle','landscape')),('area',('small','large'))):
  result[typ]={}
  for label in labels:
   xs=[x for x in rows if x[typ]==label]
   if any(not math.isfinite(x[k]) or not 0<=x[k]<=1 for x in xs for k in ('rres','eadp')):raise ValueError('score')
   a=sum(x['rres'] for x in xs);b=sum(x['eadp'] for x in xs);n=len(xs)
   result[typ][label]={'questions':n,'image_groups':len({x['group_id'] for x in xs}),'rres_mean_pct':100*a/n if n else None,'eadp_mean_pct':100*b/n if n else None,'difference_pp':100*(a-b)/n if n else None}
 return result
