"""Write-once, pre-teacher freeze for the four-stream diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
import torch
import transformers
from experiment_common import (HERE, OUT, MAX_TOKENS, NativeEngine, common,
    message, random_keep, save_json, sha256)
from oracle import answer_token_ids
from multistream_run import ARMS, contrasts, source_paths

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.benchmark and not args.run: raise ValueError('--benchmark requires --run')
    output=OUT/'run_manifest.json'
    if output.exists() or (OUT/'predictions.json').exists():
        raise RuntimeError('refusing to overwrite run freeze or existing predictions')
    manifest=OUT/'data/manifest.json'; dev=OUT/'data/development_manifest.json'; gates=OUT/'gates.json'
    gate=json.loads(gates.read_text())
    if not gate.get('pass_all') or gate.get('cpu_only') or not gate.get('source_sha256'):
        raise RuntimeError('current CPU and GPU gates must pass first')
    if gate.get('environment',{}).get('manifest_sha256') != sha256(dev):
        raise RuntimeError('GPU gate used a different development manifest')
    if not isinstance(gate.get('checkpointing'),bool):
        raise RuntimeError('GPU gate lacks checkpointing configuration')
    for path,digest in gate['source_sha256'].items():
        if sha256(path)!=digest: raise RuntimeError('gated source changed: '+path)
    items=json.loads(manifest.read_text())['items']
    if len(items)!=32 or len({x['parent_id'] for x in items})!=32:
        raise ValueError('expected 32 frozen unique parent documents')
    eng=NativeEngine(common.load_model(common.BASELINE_MODEL,max_new_tokens=MAX_TOKENS))
    if eng.text.config._attn_implementation!='sdpa': raise RuntimeError('unexpected attention backend')
    snapshot_root=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct'
    revision=(snapshot_root/'refs/main').read_text().strip()
    snapshot=snapshot_root/'snapshots'/revision
    if not snapshot.is_dir(): raise RuntimeError('cached model revision missing')
    rendered={}
    for i,item in enumerate(items):
        if sha256(item['image_path'])!=item['image_sha256']: raise RuntimeError('image hash changed')
        prep=eng.prepare(message(item),'DocVQA_VAL')
        if prep['n_vis']!=1024: raise ValueError('unexpected visual token count')
        ids=prep['ids'][0].cpu().tolist()
        targets=answer_token_ids(eng.tok,item['answers'][0],device='cpu').tolist()
        seed,keep=random_keep(item['item_id'])
        rendered[item['item_id']]=dict(input_ids=ids,n_vis=1024,target_text=item['answers'][0],
            target_ids=targets,random_seed=seed,random_keep=keep.tolist(),
            rendered_text=eng.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            image_sha256=item['image_sha256'])
        print(f'prompt freeze {i+1}/32',flush=True)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
        '--format=csv,noheader'],check=True,text=True,capture_output=True).stdout.strip()
    checksums={name:sha256(snapshot/name) for name in ('config.json','model.safetensors.index.json',
        'preprocessor_config.json') if (snapshot/name).exists()}
    payload=dict(schema_version=1,manifest_sha256=sha256(manifest),development_manifest_sha256=sha256(dev),
        gates_sha256=sha256(gates),source_sha256={str(p):sha256(p) for p in source_paths()},
        model=dict(name=common.BASELINE_MODEL,revision=revision,snapshot=str(snapshot),
            metadata_sha256=checksums,weight_files=sorted(p.name for p in snapshot.glob('*.safetensors')),
            dtype=str(next(eng.model.parameters()).dtype)),
        environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
            cuda_runtime=torch.version.cuda,device=hardware,attention_backend=eng.text.config._attn_implementation,
            rotary_inv_freq=eng.text.rotary_emb.inv_freq.detach().cpu().tolist(),
            rotary_sections=list(eng.text.rotary_emb.mrope_section),
            rotary_attention_scaling=float(eng.text.rotary_emb.attention_scaling)),
        generation=dict(max_new_tokens=MAX_TOKENS,do_sample=False,eos_ids=sorted(eng.eos_ids),
            prompt_suffix='\nAnswer with the value only.'),checkpointing=gate['checkpointing'],
        planned_arms=list(ARMS),contrasts=contrasts(),rendered_inputs=rendered,
        pre_generation=True,predictions_observed=False)
    save_json(output,payload)
    print(f'wrote {output} sha256={sha256(output)}',flush=True)
    if args.run:
        from multistream_run import main as run_main
        run_main(eng)
    if args.benchmark:
        from multistream_benchmark import main as benchmark_main
        benchmark_main(eng)

if __name__=='__main__': main()
