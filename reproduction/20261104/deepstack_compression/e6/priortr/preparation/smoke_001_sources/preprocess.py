"""Author E1 fixed-resolution input semantics; no model loading."""
from pathlib import Path
from PIL import Image
from qwen_vl_utils import process_vision_info

def square(image,color):
 w,h=image.size
 if w==h:return image
 side=max(w,h);new=Image.new(image.mode,(side,side),color);new.paste(image,((side-w)//2,(side-h)//2));return new

def prepare(processor,message):
 content=[]
 for item in message:
  if item['type']=='image':
   image=item['value']
   if not image.startswith(('http://','https://','file://','data:image')):
    if not Path(image).is_file():raise ValueError('image path missing')
    image='file://'+image
   content.append({'type':'image','image':image,'min_pixels':1048576,'max_pixels':1048576})
  elif item['type']=='text':content.append({'type':'text','text':item['value']})
  else:raise ValueError('only single-image DEV input is supported')
 if sum(item['type']=='image' for item in content)!=1:raise ValueError('expected one image')
 messages=[{'role':'user','content':content}]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 images,videos,video_kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
 if videos is not None:raise ValueError('videos are out of scope')
 mean=getattr(processor.image_processor,'image_mean',[.5,.5,.5]);color=tuple(int(x*255) for x in mean)
 images=[square(im,color).resize((1024,1024)) for im in images]
 return processor(text=text,images=images,videos=None,video_metadata=None,do_resize=False,return_tensors='pt',**(video_kwargs or {}))
