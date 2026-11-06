"""CPU-checkable launch contract, never grants actual GPU permission."""
import re
from pathlib import Path

def validate(launch,*,directory,nonce,worker_sha,reviewed_source_sha,allocation_sha):
    d=Path(directory).resolve()
    if not re.fullmatch(r'[0-9a-f]{32}',nonce):raise ValueError('nonce invalid')
    if not isinstance(reviewed_source_sha,dict) or not reviewed_source_sha:raise ValueError('reviewed sources missing')
    expected={'stage':'smoke','mode':'C-ON','run_dir':str(d),'nonce':nonce,
              'worker_sha256':worker_sha,'source_sha256':reviewed_source_sha,
              'allocation_sha256':allocation_sha,'gpu_authorized':True}
    if not re.fullmatch(r'[0-9a-f]{64}',str(worker_sha)) or not re.fullmatch(r'[0-9a-f]{64}',str(allocation_sha)):
        raise ValueError('worker/allocation digest invalid')
    if any(launch.get(k)!=v for k,v in expected.items()):raise ValueError('worker launch/approval identity mismatch')
    return {'stage':'smoke','mode':'C-ON','nonce':nonce,'scope':'validated fields only; no permission by itself'}

if __name__=='__main__':raise SystemExit('locked: contract alone cannot authorize GPU')
