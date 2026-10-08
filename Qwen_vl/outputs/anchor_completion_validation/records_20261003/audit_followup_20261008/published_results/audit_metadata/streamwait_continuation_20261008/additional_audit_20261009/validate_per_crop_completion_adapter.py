"""CPU mechanism gates, using AST-extracted frozen operators verbatim."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F

from per_crop_completion_adapter import make_per_crop_completion_stream

ROOT=Path('/media/disk2/YZX/research/EADP_amp')
OUT=Path(__file__).resolve().parent
PORT=ROOT/'LLaVA/llava/model/llava_arch_anchorzip.py'
AMP=ROOT/'Qwen_vl/scripts/anchor_merge_pilot/amp_common.py'


def extract(path, names, namespace):
    tree=ast.parse(path.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise RuntimeError('Frozen functions missing')
    module=ast.Module(body=functions,type_ignores=[])
    # The original function bodies and decorators remain untouched.
    exec(compile(module,str(path),'exec'),namespace)
    return {n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in functions}


def main():
    torch.set_num_threads(1)
    amp_namespace=dict(torch=torch,F=F,Optional=Optional)
    ast_hashes=extract(AMP,('group_sum','compute_assignment','merge_stream'),amp_namespace)
    synthetic_amp=types.ModuleType('amp_common')
    for name in ('group_sum','compute_assignment','merge_stream'):
        setattr(synthetic_amp,name,amp_namespace[name])
    if 'amp_common' in sys.modules:
        raise RuntimeError('Run this isolated CPU checker in a fresh process')
    sys.modules['amp_common']=synthetic_amp
    port_namespace=dict(torch=torch,LAM=.25)
    ast_hashes.update(extract(PORT,('_completion_stream',),port_namespace))
    original=port_namespace['_completion_stream']
    calls=[]

    def observed(feat,keep,unused):
        dropped,gid,_=amp_namespace['compute_assignment'](feat,keep)
        calls.append(dict(n_tokens=int(feat.shape[0]),local_keep=keep.tolist(),
                          dropped_indices=dropped.tolist(),assigned_anchor_indices=keep[gid].tolist()))
        return original(feat,keep,unused)

    adapter=make_per_crop_completion_stream(observed)
    gen=torch.Generator().manual_seed(20261009)
    feat=torch.randn(576,8,generator=gen).half()
    keep=torch.tensor([27,2,400,575],dtype=torch.long)
    before_feat=feat.clone();before_keep=keep.clone()
    single=adapter(feat,keep,None)
    gates=dict(single_crop_bit_exact=torch.equal(single,original(feat,keep,None)),
               caller_features_unchanged=torch.equal(feat,before_feat),
               caller_keep_unchanged=torch.equal(keep,before_keep))

    five=torch.cat([torch.tensor([1.,0.]).expand(576,2).clone(),
                    torch.tensor([0.,1.]).expand(576,2).clone(),
                    torch.tensor([-1.,0.]).expand(576,2).clone(),
                    torch.tensor([0.,-1.]).expand(576,2).clone(),
                    torch.tensor([1.,1.]).expand(576,2).clone()]).half()
    five[1:576]=torch.tensor([0.,1.]);five[577:1152]=torch.tensor([1.,0.])
    global_keep=torch.tensor([2304,0,1728,576,1152],dtype=torch.long)
    sorted_keep=global_keep.sort().values
    dropped,gid,_=amp_namespace['compute_assignment'](five,global_keep)
    legacy_cross=int(((dropped//576)!=(global_keep[gid]//576)).sum())
    legacy=original(five,global_keep,None)
    calls.clear()
    corrected=adapter(five,global_keep,None)
    crop_calls=list(calls)
    independent=torch.cat([original(five[c*576:(c+1)*576],torch.tensor([0]),None)
                           for c in range(5)])
    gates.update(five_crop_calls_only_576_tokens=len(crop_calls)==5 and all(c['n_tokens']==576 for c in crop_calls),
                 five_crop_local_indices_in_range=all(all(0<=i<576 for i in c['local_keep']+c['dropped_indices']+c['assigned_anchor_indices']) for c in crop_calls),
                 five_crop_independent_formula_bit_exact=torch.equal(corrected,independent),
                 counterexample_differs_from_cross_crop=not torch.equal(corrected,legacy),
                 selected_count_unchanged=int(corrected.shape[0])==int(global_keep.numel()),
                 global_gather_order_ascending=all(c['local_keep']==[0] for c in crop_calls))
    port_namespace['LAM']=0.
    gates['lambda0_bit_exact_gather']=torch.equal(adapter(five,global_keep,None),five[sorted_keep])
    port_namespace['LAM']=.25
    full_keep=torch.arange(five.shape[0],dtype=torch.long)
    gates['full_keep_no_dropped_bit_exact']=torch.equal(adapter(five,full_keep,None),five)
    gates['full_keep_same_as_original']=torch.equal(adapter(five,full_keep,None),original(five,full_keep,None))
    gates['empty_keep_legal']=adapter(five,torch.empty(0,dtype=torch.long),None).shape==(0,2)
    sparse_keep=torch.tensor([2304,0],dtype=torch.long)
    sparse_expected=torch.cat([original(five[:576],torch.tensor([0]),None),
                               original(five[2304:2880],torch.tensor([0]),None)])
    gates['unselected_crops_skip_legal']=torch.equal(adapter(five,sparse_keep,None),sparse_expected)
    gates['no_production_model_import']=not any(name=='llava' or name.startswith('llava.') for name in sys.modules)
    gates['no_cuda_initialization']=not torch.cuda._initialized
    assert all(gates.values()),gates
    report=dict(candidate='Per-crop Completion adapter, not installed or registered',
                lambda_completion=.25,crop_tokens=576,seed=20261009,gates=gates,
                legacy_counterexample_cross_crop_assignments=legacy_cross,
                legacy_output=legacy.tolist(),candidate_output=corrected.tolist(),
                candidate_vs_legacy_max_abs_difference=float((legacy.float()-corrected.float()).abs().max()),
                original_function_AST_sha256=ast_hashes,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (PORT,AMP,Path(__file__),OUT/'per_crop_completion_adapter.py')},
                scope='Only Completion grouping differs. Original functions perform all mathematics; RTG segment0, parameters, quota/facility/indices/gather order unchanged. No existing frozen file, queue, protocol, prediction, env or GPU touched.',
                causal_boundary='All gates are CPU synthetic mechanism checks, not measured downstream accuracy or a claim of improvement.')
    (OUT/'per_crop_completion_adapter_cpu_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=gates,legacy_cross=legacy_cross,max_difference=report['candidate_vs_legacy_max_abs_difference']),indent=2))


if __name__=='__main__':
    main()
