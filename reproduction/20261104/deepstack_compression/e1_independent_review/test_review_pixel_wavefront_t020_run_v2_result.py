"""Synthetic CPU counterexamples for the independent six-arm reviewer."""
import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent/'review_pixel_wavefront_t020_run_v2_result.py';spec=importlib.util.spec_from_file_location('wave_review_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_p3_known_geometry_and_invalid_image(self):
  cfg={'image_token_id':9,'vision_start_token_id':8,'video_token_id':7};prefix=[3,8]+[9]*1024+[4,5];start,keep,pos,maxpos,delta=m.p3(prefix,[0,32,1023],cfg)
  self.assertEqual(start,2);self.assertEqual(keep,[0,1,2,34,1025,1026,1027]);self.assertEqual([p[0][2:5] for p in pos],[[2,2,2],[2,3,33],[2,2,33]])
  with self.assertRaises(ValueError):m.p3(prefix+[7],[0],cfg)
 def test_archived_tensor_bytes_and_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   old=m.O;m.O=Path(d);path=Path(d)/'arm.npz';emb=np.array([[[0x3f80,0x4000],[0x4040,0x4080],[0x40a0,0x40c0]]],dtype=np.uint16);mask=np.array([[False,True,False]],dtype=np.bool_);att=np.array([[1,1,1]],dtype=np.int64);pos=np.array([[[0,1,2]],[[0,1,2]],[[0,1,2]]],dtype=np.int64);log=np.array([[0x3f80]],dtype=np.uint16);ds=[np.array([[0x4000,0x4040]],dtype=np.uint16) for _ in range(3)];tensors={'prefill_embeddings':emb,'prefill_mask':att,'prefill_positions':pos,'visual_mask':mask,'prefill_logits':log,**{'injected_ds'+str(i):x for i,x in enumerate(ds)}};types={'prefill_embeddings':'torch.bfloat16','prefill_mask':'torch.int64','prefill_positions':'torch.int64','visual_mask':'torch.bool','prefill_logits':'torch.bfloat16',**{'injected_ds'+str(i):'torch.bfloat16' for i in range(3)}};raw={k:np.ascontiguousarray(v).view(np.uint8) for k,v in tensors.items()};np.savez(path,**raw);meta={k:{'dtype':types[k],'shape':list(v.shape),'sha256':hashlib.sha256(raw[k].tobytes()).hexdigest()} for k,v in tensors.items()};selected=np.ascontiguousarray(emb[mask]).view(np.uint8);r={'archive_path':str(path),'archive_sha256':m.sha(path),'prefill_tensors':meta,'prefill_embeddings':meta['prefill_embeddings'],'prefill_logits':meta['prefill_logits'],'injected_deepstack':[meta['injected_ds'+str(i)] for i in range(3)],'visual_mask':mask.tolist(),'prefill_position_ids':pos.tolist(),'prefill_mask':att.tolist(),'main_visual':{'sha256':hashlib.sha256(selected.tobytes()).hexdigest()}}
   try:
    self.assertEqual(m.checked_archive(r)[0],path)
    r['main_visual']['sha256']='0'*64
    with self.assertRaises(ValueError):m.checked_archive(r)
   finally:m.O=old
if __name__=='__main__':unittest.main()
