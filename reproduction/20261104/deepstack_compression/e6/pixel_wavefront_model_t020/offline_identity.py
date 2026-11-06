"""Draft CPU reconstruction of a future author's/wavefront's recorded pair.

This checks evidence content, not launch/worker provenance. The future formal
and independent verifiers must additionally bind supervisor and Docker exits,
frozen sources, actual arm schedule and independently accepted preflight.
"""
import hashlib,importlib.util,json,struct
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/pixel_wavefront_model_t020'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def ints_sha(values):return hashlib.sha256(struct.pack('<'+'q'*len(values),*values)).hexdigest()
def validate_arm(directory,anchor,expected,config):
 directory=Path(directory).resolve();r=read(directory/'record.json');raw=read(directory/'returned.json');ids=raw['full_ids'];length=raw['input_length']
 if (raw.get('task'),str(raw.get('qid')),raw.get('arm'))!=(anchor['task'],str(anchor['qid']),r.get('path')) or type(length) is not int or length!=r['original_sequence_length'] or not isinstance(ids,list) or ids[length:]!=r['output_ids'] or r['output_ids']!=expected['tokens'] or r['selected_indices']!=expected['selected_indices']:raise ValueError('returned sequence/reference mismatch')
 if r['processor']!=expected['processor'] or r['generation_scalars']!=expected['generation_kwargs'] or r['eos_ids']!=expected['eos_ids']:raise ValueError('processor or recipe mismatch')
 if ints_sha(ids[:length])!=r['processor']['input_ids']['sha256'] or ints_sha([1]*length)!=r['processor']['attention_mask']['sha256'] or ints_sha([1,64,64])!=r['processor']['image_grid_thw']['sha256']:raise ValueError('prompt/mask/grid bytes differ from processor identity')
 p3=load('_wavefront_plain_p3',D/'p3_reconstruct.py').reconstruct(ids[:length],config['image_token_id'],config['vision_start_token_id'],config['video_token_id'],r['selected_indices'])
 if p3['position_ids']!=r['prefill_position_ids'] or p3['rope_delta']!=r['rope_delta'] or p3['original_max_position']!=r['original_max_position']:raise ValueError('independent P3 reconstruction mismatch')
 archive=Path(r['archive_path']).resolve()
 if not archive.is_relative_to(directory) or archive.name!='actual_prefill.npz':raise ValueError('archive outside arm')
 load('_wavefront_archive_contract',P/'archive_contract.py').validate_archive(r)
 events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
 if [e.get('event') for e in events]!=['generation_started','selector_returned','generation_returned'] or any(e.get('arm')!=r['path'] or e.get('task')!=anchor['task'] or str(e.get('qid'))!=str(anchor['qid']) for e in events) or events[1].get('indices')!=r['selected_indices']:raise ValueError('exposure/selector journal mismatch')
 return r
def validate_pair(author_dir,wavefront_dir,anchor,expected,config):
 author=validate_arm(author_dir,anchor,expected,config);wave=validate_arm(wavefront_dir,anchor,expected,config)
 load('_wavefront_pair_contract',P/'identity_contract.py').compare_pair(author,wave,anchor,expected['generation_kwargs'])
 return {'status':'passed','task':anchor['task'],'qid':str(anchor['qid']),'selected_count':len(author['selected_indices']),'generation_tokens':len(author['output_ids']),'KV_scope':'36-layer byte fingerprints, not archived KV tensors'}
if __name__=='__main__':raise SystemExit('locked: offline pair validator has no reviewed formal run')
