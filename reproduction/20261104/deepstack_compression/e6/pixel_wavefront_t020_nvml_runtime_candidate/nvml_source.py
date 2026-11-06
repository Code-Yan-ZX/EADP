"""Sanitized device/process memory query source, callable only under future admission."""
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

def mib(value):
    if value is None or value.strip() in ('N/A','Not Supported','[Not Found]'):return None
    match=re.fullmatch(r'\s*(\d+)\s+MiB\s*',value)
    if not match:raise ValueError('unrecognized memory unit/value')
    return int(match.group(1))*1048576

def parse_xml(text,uuid):
    root=ET.fromstring(text);gpus=[g for g in root.findall('gpu') if g.findtext('uuid')==uuid]
    if len(gpus)!=1:raise ValueError('expected one UUID-matched GPU')
    gpu=gpus[0];fb=gpu.find('fb_memory_usage');proc=gpu.find('processes')
    if fb is None or proc is None:raise ValueError('device memory or processes unavailable')
    memory={key:mib(fb.findtext(key)) for key in ('total','used','free','reserved')}
    processes=[];seen=set()
    for item in proc.findall('process_info'):
        pid=item.findtext('pid');kind=(item.findtext('type') or '').strip()
        if not pid or not pid.strip().isdecimal() or int(pid)<=0 or kind not in ('C','G','C+G'):
            raise ValueError('GPU process identity/type unavailable')
        pid=int(pid)
        if pid in seen:raise ValueError('duplicate GPU process PID')
        seen.add(pid)
        processes.append({'pid':pid,'type':kind,'used_bytes':mib(item.findtext('used_memory'))})
    return {'gpu_uuid':uuid,'device_memory_bytes':memory,'processes':processes}

def source(expected,metadata_stream,*,run=subprocess.run,clock=time.monotonic_ns):
    """Return a replacement for sample_only's subprocess.run, not a GPU launcher."""
    uuid=expected['gpu_uuid']
    def query(argv,**kwargs):
        if argv!=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']:
            raise ValueError('unexpected sampler query')
        start=clock()
        try:
            result=run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True,timeout=10)
            meta=parse_xml(result.stdout,uuid)
        except Exception as exc:
            # Do not log subprocess output, command lines or exception strings.
            metadata_stream.write(json.dumps({'status':'failed','query_start_ns':start,'query_end_ns':clock(),
                                             'error_type':type(exc).__name__})+'\n');metadata_stream.flush();os.fsync(metadata_stream.fileno())
            raise RuntimeError('device query failed: '+type(exc).__name__) from None
        meta.update(status='completed',query_start_ns=start,query_end_ns=clock())
        metadata_stream.write(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n');metadata_stream.flush();os.fsync(metadata_stream.fileno())
        compute=[]
        for p in meta['processes']:
            if p['type'] in ('C','C+G'):
                if p['used_bytes'] is None:raise ValueError('compute process memory unavailable')
                compute.append(f"{uuid}, {p['pid']}, {p['used_bytes']//1048576}")
        # Only the existing sampler's CSV interface receives this sanitized data.
        class Result: pass
        wrapped=Result();wrapped.stdout='\n'.join(compute);return wrapped
    return query

if __name__=='__main__':raise SystemExit('locked: query source cannot authorize GPU work')
