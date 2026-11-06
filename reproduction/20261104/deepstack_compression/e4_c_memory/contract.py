"""CPU-checkable handshake and sampled NVML proof; no GPU launch authority.

The sampler describes the whole isolated worker process, never individual
jobs. nvidia-smi reports MiB-rounded values and can miss transient peaks.
"""
import re

MODES=('C-ON','C-DROP','C-SKIP')
def identity(value):
 if not isinstance(value,dict) or value.get('mode') not in MODES or not re.fullmatch(r'[0-9a-f]{32}',str(value.get('nonce',''))):raise ValueError('bad mode/nonce')
 if not re.fullmatch(r'[0-9a-f]{64}',str(value.get('container_id',''))) or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',str(value.get('gpu_uuid',''))):raise ValueError('bad container/GPU identity')
 if type(value.get('host_pid')) is not int or value['host_pid']<=0 or type(value.get('start_ticks')) is not int or value['start_ticks']<=0 or type(value.get('max_sample_gap_ns')) is not int or value['max_sample_gap_ns']<=0 or not value.get('boot_id'):raise ValueError('bad host process identity')
 return value

def admit_release(expected,ready,sampler_started,first_sample):
 """A controller may release worker only after correctly identified sampling."""
 identity(expected)
 if not isinstance(ready,dict) or any(ready.get(k)!=expected[k] for k in ('mode','nonce','container_id')) or ready.get('cuda_initialized') is not False or ready.get('namespace_pid')!=1:raise ValueError('worker ready mismatch')
 if not isinstance(sampler_started,dict) or any(sampler_started.get(k)!=expected[k] for k in ('mode','nonce','container_id','gpu_uuid','host_pid','start_ticks','boot_id')):raise ValueError('sampler identity mismatch')
 if sampler_started.get('status')!='sampling' or type(sampler_started.get('monotonic_ns')) is not int:raise ValueError('sampler not started')
 if any(first_sample.get(k)!=expected[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')) or first_sample.get('type')!='sample' or type(first_sample.get('query_start_ns')) is not int or type(first_sample.get('query_end_ns')) is not int or first_sample['query_start_ns']<sampler_started['monotonic_ns'] or first_sample['query_end_ns']<first_sample['query_start_ns'] or first_sample['query_end_ns']-first_sample['query_start_ns']>expected['max_sample_gap_ns'] or first_sample.get('monotonic_ns')!=first_sample['query_end_ns'] or first_sample.get('other_gpu_processes')!=[] or first_sample.get('used_bytes') is not None:raise ValueError('first sample invalid, target initialized, or GPU not exclusive')
 return {'status':'release','mode':expected['mode'],'nonce':expected['nonce'],'container_id':expected['container_id'],'first_sample_ns':first_sample['monotonic_ns']}

def audit(expected,release,samples,sampler_terminal,worker_end_ns):
 """Require continuous recorded process samples and a completed sampler."""
 identity(expected)
 if release.get('status')!='release' or any(release.get(k)!=expected[k] for k in ('mode','nonce','container_id')):raise ValueError('release mismatch')
 if sampler_terminal.get('reason')!='completed' or sampler_terminal.get('worker_returncode')!=0 or sampler_terminal.get('remaining_process_group')!=[]:raise ValueError('sampler did not exit cleanly')
 if not isinstance(samples,list) or not samples or any(any(s.get(k)!=expected[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')) or s.get('type')!='sample' or type(s.get('query_start_ns')) is not int or type(s.get('query_end_ns')) is not int or s['query_start_ns']>s['query_end_ns'] or s['query_end_ns']-s['query_start_ns']>expected['max_sample_gap_ns'] or s.get('monotonic_ns')!=s['query_end_ns'] or s.get('other_gpu_processes')!=[] for s in samples):raise ValueError('bad sample or concurrent GPU process')
 times=[s['monotonic_ns'] for s in samples]
 if times!=sorted(set(times)) or any(b['query_start_ns']<a['query_end_ns'] or b['query_end_ns']-a['query_start_ns']>expected['max_sample_gap_ns'] for a,b in zip(samples,samples[1:])) or release.get('first_sample_ns')!=times[0] or type(worker_end_ns) is not int or worker_end_ns<times[0] or samples[-1]['query_start_ns']<worker_end_ns:raise ValueError('sampling does not cover worker lifetime')
 used=[s.get('used_bytes') for s in samples if s.get('used_bytes') is not None]
 if not used or any(type(n) is not int or n<0 or n%1048576 for n in used):raise ValueError('target process memory missing or malformed')
 return {'status':'passed','mode':expected['mode'],'host_pid':expected['host_pid'],'start_ticks':expected['start_ticks'],'gpu_uuid':expected['gpu_uuid'],
         'samples':len(samples),'nvml_process_sampled_peak_bytes':max(used),'scope':'whole isolated process, MiB-rounded sampled maximum; transient peaks can be missed'}
