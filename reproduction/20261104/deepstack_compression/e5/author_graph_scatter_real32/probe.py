"""Capture once per K and compare 32 archived inputs, no standalone CLI."""
import sys,time
from pathlib import Path
import torch
import inventory

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
if Path(inventory.__file__).resolve()!=HERE/'inventory.py':raise ValueError('wrong inventory import')
SCATTER=BASE/'e5/author_graph_scatter';GRAPH=BASE/'e5/author_graph'
sys.path[:0]=[str(SCATTER),str(GRAPH)]
import transform,selector
if Path(transform.__file__).resolve()!=SCATTER/'transform.py' or Path(selector.__file__).resolve()!=GRAPH/'selector.py':
    raise ValueError('wrong actual transform or graph selector import')

def memory():
    return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

def load_case(item):
    path=Path(item['path'])
    if inventory.sha(path)!=item['sha256']:raise ValueError('case SHA changed before load')
    case=torch.load(path,map_location='cpu',weights_only=True)
    if inventory.sha(path)!=item['sha256']:raise ValueError('case SHA changed during load')
    imp=case['importance'];sim=case['similarity'];order=case['ordered'];k=item['budget']
    if (imp.shape!=(1,1024) or sim.shape!=(1,1024,1024) or order.shape!=(1,k)
        or imp.dtype!=torch.float32 or sim.dtype!=torch.float32 or order.dtype!=torch.int64
        or not torch.isfinite(imp).all() or not torch.isfinite(sim).all()):
        raise ValueError('case tensor contract failed')
    return case

def run_one(k,cases,emit):
    if k not in inventory.KS or len(cases)!=32 or [x['image_index'] for x in cases]!=list(range(32)) or any(x['budget']!=k for x in cases):
        raise ValueError('unexpected image/budget schedule')
    module=transform.load_author();candidate,metadata=transform.build(module);author=module._greed_select_impl
    data=load_case(cases[0]);imp=data['importance'].to('cuda');sim=data['similarity'].to('cuda')
    author(imp,sim,1);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=memory();start=time.perf_counter_ns();graph=selector.AuthorGraph(imp,sim,k,candidate)
    torch.cuda.synchronize();setup=(time.perf_counter_ns()-start)/1e6;after=memory()
    emit({'event':'capture_completed','k':k,'setup_wall_ms':setup,'memory_before':before,'memory_after':after,
          'transform':metadata,'scope':'pytorch allocator; not device-wide peak memory'})
    matched_all=True
    for item in cases:
        data=load_case(item);imp=data['importance'].to('cuda');sim=data['similarity'].to('cuda');saved=data['ordered']
        with torch.inference_mode():
            original,quota=author(imp,sim,k);torch.cuda.synchronize()
            start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter_ns();start_event.record();actual,actual_quota=graph(imp,sim);end_event.record()
            end_event.synchronize();wall_ms=(time.perf_counter_ns()-wall)/1e6
            graph_equal=bool(torch.equal(original,actual) and torch.equal(quota,actual_quota))
            author_matches_archive=bool(torch.equal(original.cpu(),saved))
            row={'event':'case_compared','k':k,'image_index':item['image_index'],'task':item['task'],'qid':item['qid'],
                 'group_id':item['group_id'],'case_sha256':item['sha256'],'graph_equals_author':graph_equal,
                 'author_equals_archive':author_matches_archive,'author_order':original.cpu().tolist(),
                 'graph_order':actual.cpu().tolist(),'author_quota':quota.cpu().tolist(),
                 'graph_quota':actual_quota.cpu().tolist(),'input_copy_plus_replay_device_ms':start_event.elapsed_time(end_event),
                 'host_wall_ms':wall_ms}
            start2=torch.cuda.Event(enable_timing=True);end2=torch.cuda.Event(enable_timing=True)
            start2.record();graph.graph.replay();end2.record();end2.synchronize()
            row['graph_only_device_ms']=start2.elapsed_time(end2)
            row['repeat_same_input_equal']=bool(torch.equal(original,actual) and torch.equal(quota,actual_quota))
            matched_all=matched_all and graph_equal and author_matches_archive and row['repeat_same_input_equal']
            emit(row)
    emit({'event':'budget_completed','k':k,'cases':32,'all_identity_equal':matched_all})
    return matched_all
