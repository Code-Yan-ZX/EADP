"""CPU-only post-run checks for nine isolated Pixel memory jobs.

Not an admission; no GPU/model calls. Samples are MiB-rounded and can miss peaks.
"""
import hashlib

def verify_identity(job, identity, raw, reference, prompt_bytes):
    if identity.get('status') != 'passed' or identity.get('job') != job or raw.get('job') != job:
        raise ValueError('job identity mismatch')
    for got, expected in (('processor','processor'),('selected_indices','selected_indices'),('output_ids','tokens'),('eos_ids','eos_ids'),('generation_kwargs','generation_kwargs')):
        if identity.get(got) != reference.get(expected):
            raise ValueError('same-arm reference mismatch: '+got)
    if identity.get('GT_opened') is not False or identity.get('scoring_performed') is not False:
        raise ValueError('unexpected scoring')
    full=raw.get('full_sequence_ids'); n=raw.get('accepted_input_length')
    info=reference['processor']['input_ids']
    if not isinstance(full,list) or any(type(x) is not int for x in full) or type(n) is not int or info.get('dtype') != 'torch.int64' or info.get('shape') != [1,n] or n<1 or full[n:] != reference['tokens']:
        raise ValueError('raw sequence mismatch')
    if hashlib.sha256(prompt_bytes(full[:n])).hexdigest()!=info.get('sha256'):
        raise ValueError('raw prompt mismatch')
    return {'prompt_length':n,'generated_length':len(full)-n,'selected_count':len(reference['selected_indices'])}

def stages(events):
    wanted=('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end')
    found={}
    for e in events:
        if e.get('event') in wanted:
            if e['event'] in found:raise ValueError('duplicate stage')
            found[e['event']]=e
    if set(found)!=set(wanted):raise ValueError('missing stage')
    times=[found[x].get('monotonic_ns') for x in wanted]
    if any(type(t) is not int or t<=0 for t in times) or any(a>=b for a,b in zip(times,times[1:])):
        raise ValueError('invalid stage order')
    return found,dict(release=times[0],**dict(zip(wanted[1:],times[1:])))

def check_handshake(job,expected,ready,release,stopped,first_sample):
    for k in ('container_id','nonce','mode'):
        if ready.get(k)!=expected.get(k) or release.get(k)!=expected.get(k):raise ValueError('barrier identity')
    if expected.get('mode')!=job['arm'] or ready.get('cuda_initialized') is not False or ready.get('namespace_pid')!=1 or release.get('status')!='release':raise ValueError('pre-CUDA handshake')
    if type(release.get('first_sample_ns')) is not int or first_sample.get('query_end_ns',0)>release['first_sample_ns']:
        raise ValueError('first sample after release')
    for k in ('container_id','nonce','host_pid','start_ticks','boot_id'):
        if expected.get(k)!=stopped.get(k):raise ValueError('stopped identity')


def check_journal(journal,jobs,final):
    if len(jobs)!=9 or len(journal)!=18 or final!={'status':'passed','jobs':9}:raise ValueError('nine-job inventory')
    for i,j in enumerate(jobs):
        if j.get('order')!=i or journal[2*i].get('event')!='started' or journal[2*i+1].get('event')!='completed' or journal[2*i].get('job')!=j or journal[2*i+1].get('job')!=j:raise ValueError('nine-job order or completion')
