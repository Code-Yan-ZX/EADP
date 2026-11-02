"""Author Python CPU subprocess in an isolated container without NVIDIA devices."""
import os
from pathlib import Path
import common as c
IMAGE='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
ROOT=c.HERE.parents[3]

def command(args,name,write_output=False,cidfile=None):
    mounts=[(Path('/home/dell/miniconda3'),True),(ROOT,not write_output),
            (Path('/media/disk2/YZX/doct/FastV'),True),
            (Path('/home/dell/.cache/huggingface'),True),
            (Path('/data/models/huggingface'),True)]
    cmd=['docker','run','--rm','--name',name,'--network','none','--cap-drop','ALL','--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}']
    if cidfile is not None:cmd+=['--cidfile',str(cidfile)]
    for path,ro in mounts:
        if path.exists():cmd+=['--mount',f'type=bind,src={path},dst={path}'+(',readonly' if ro else '')]
    for k,v in dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HOME='/tmp').items():cmd+=['--env',f'{k}={v}']
    return cmd+['--workdir',str(c.HERE),'--entrypoint',c.PYTHON,IMAGE,'-B',*args]
