"""One-K actual author-graph probe. No CLI: reviewed worker must invoke it."""
import importlib.util,inspect,time
from pathlib import Path
import torch
import selector
if Path(selector.__file__).resolve()!=Path(__file__).resolve().parent/'selector.py':raise ValueError('wrong selector module')
AuthorGraph=selector.AuthorGraph


def load_author(path, expected_sha, sha):
    path=Path(path).resolve()
    if sha(path)!=expected_sha:raise ValueError('author source changed')
    spec=importlib.util.spec_from_file_location('fixed_eadp_author_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fn=module._greed_select_impl
    if Path(inspect.getfile(fn)).resolve()!=path:raise ValueError('wrong actual author function')
    return fn


def memory():
    return {'allocated_bytes':torch.cuda.memory_allocated(), 'reserved_bytes':torch.cuda.memory_reserved(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def run_one(k,case,author,emit):
    """Capture once; compare complete IDs/quotas and measure device boundaries.

    `emit` must durably persist each record; the enclosing worker owns timeout,
    exception handling and process isolation. All arguments require prior admission.
    """
    if type(k) is not int or k not in (1,16,128,256,512):raise ValueError('unexpected K')
    imp=case['importance'].to('cuda');sim=case['similarity'].to('cuda')
    author(imp,sim,1)  # initialize kernels before capture, not part of timing
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=memory();start=time.perf_counter_ns()
    graph=AuthorGraph(imp,sim,k,author)
    torch.cuda.synchronize();setup_ms=(time.perf_counter_ns()-start)/1e6;after=memory()
    emit({'event':'capture_completed','k':k,'setup_wall_ms':setup_ms,'memory_before':before,'memory_after':after,
          'scope':'pytorch allocator only; capture pool and baseline not isolated from total device use'})
    # A->B->A exposes missing copies and graph state leakage; ties/zero/random
    # retain shape, dtype and K. No output views survive the next replay.
    changed=(imp.roll(7,dims=1),sim.roll((7,7),dims=(1,2)))
    ties=(torch.ones_like(imp),torch.full_like(sim,.5))
    zero=(torch.zeros_like(imp),torch.zeros_like(sim))
    generator=torch.Generator(device='cpu').manual_seed(20261105)
    random_imp=torch.rand(imp.shape,dtype=torch.float32,generator=generator).to('cuda')
    random_sim=torch.rand(sim.shape,dtype=torch.float32,generator=generator).to('cuda')
    inputs=[('real_A',imp,sim),('changed_B',*changed),('real_A_again',imp,sim),
            ('tie',*ties),('zero',*zero),('seeded_random',random_imp,random_sim)]
    success=True
    for name,importance,similarity in inputs:
        with torch.inference_mode():
            reference,reference_quota=author(importance,similarity,k)
            torch.cuda.synchronize()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter_ns();begin.record();actual,quota=graph(importance,similarity);end.record()
            end.synchronize();wall_ms=(time.perf_counter_ns()-wall)/1e6
            matched=bool(torch.equal(reference,actual) and torch.equal(reference_quota,quota))
            row={'event':'replay_compared','k':k,'input':name,'matched':matched,
                 'author_order':reference.cpu().tolist(),'graph_order':actual.cpu().tolist(),
                 'author_quota':reference_quota.cpu().tolist(),'graph_quota':quota.cpu().tolist(),
                 'input_copy_plus_replay_device_ms':begin.elapsed_time(end),'host_wall_ms':wall_ms}
            # Replay only (same cached input); separate from input-copy + replay.
            begin2=torch.cuda.Event(enable_timing=True);end2=torch.cuda.Event(enable_timing=True)
            begin2.record();graph.graph.replay();end2.record();end2.synchronize()
            row['graph_only_device_ms']=begin2.elapsed_time(end2)
            row['repeat_same_input_equal']=bool(torch.equal(reference,actual) and torch.equal(reference_quota,quota))
            success=success and matched and row['repeat_same_input_equal']
            emit(row)
    emit({'event':'k_completed','k':k,'matched_all_inputs':success,'comparisons':len(inputs)})
    return success
