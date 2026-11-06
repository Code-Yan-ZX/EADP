"""CPU-only preparation checks for a future model worker; no GPU CLI."""
from pathlib import Path


def prepare_cpu(*,stage,mode,approved_sources,require_identity,prepare,load_runtime,build,sha):
    if stage!='smoke' or mode!='C-ON':raise ValueError('only first-mode smoke is prepared')
    if not isinstance(approved_sources,dict) or not approved_sources:raise ValueError('reviewed source map required')
    identity=require_identity()
    prepared=prepare(stage)
    actual=prepared.get('source_sha256')
    if not isinstance(actual,dict) or not actual or any(approved_sources.get(p)!=h or sha(p)!=h for p,h in actual.items()):
        raise ValueError('worker source not in reviewed source map')
    expected=build(prepared['rows'],prepared['excluded'],stage)
    if prepared['schedule']!=expected:raise ValueError('worker schedule independently rebuilt mismatch')
    if len(prepared['weights_before'])!=4:raise ValueError('weight identities incomplete')
    torch,native,wrapper,versions=load_runtime(prepared)
    if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
    if any(sha(p)!=h for p,h in actual.items()):raise ValueError('worker source changed during preflight')
    return {'identity':identity,'prepared':prepared,'torch':torch,'native':native,'wrapper_class':wrapper,
            'versions':versions,'expected_schedule':expected,'stage':stage,'mode':mode,
            'scope':'CPU preparation only; existing DEV prediction journal opened for input identity; no GPU/model/new prediction/GT'}


if __name__=='__main__':raise SystemExit('locked: no real GPU worker CLI exists')
