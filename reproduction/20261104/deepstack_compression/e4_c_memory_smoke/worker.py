"""Candidate actual worker bridge; no executable entry or admission authority."""
import json
from pathlib import Path
import worker_contract


def run(directory,nonce,*,admit,prepare,continue_run,sha):
    """A later independently reviewed entry must supply fixed, verified callbacks."""
    d=Path(directory).resolve();approval=admit()
    if (approval.get('scope')!='C-ON isolated memory smoke'
        or approval.get('gpu_authorized') is not True
        or approval.get('stage')!='smoke' or approval.get('mode')!='C-ON'):
        raise ValueError('real GPU smoke not independently admitted')
    launch=json.loads((d/'launch.json').read_text())
    worker_contract.validate(launch,directory=d,nonce=nonce,worker_sha=sha(__file__),
                             reviewed_source_sha=approval['source_sha256'],allocation_sha=approval['allocation_sha256'])
    def sources_unchanged():
        if any(sha(path)!=digest for path,digest in approval['source_sha256'].items()):
            raise ValueError('reviewed source changed')
    sources_unchanged()
    context=prepare(approval)
    if (context.get('stage')!='smoke' or context.get('mode')!='C-ON'
        or context['torch'].cuda.is_initialized() is not False):raise ValueError('worker pre-barrier state invalid')
    # The continuation must be the fixed reviewed worker_candidate implementation:
    # its PID-1 handshake checks the release before CUDA/model initialization.
    sources_unchanged()
    result=continue_run(context,d,nonce,approval)
    sources_unchanged()
    return result


if __name__=='__main__':raise SystemExit('locked: independently reviewed GPU admission/entry missing')
