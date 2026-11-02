"""Preserve original rows while restoring frozen dataset order; no labels."""
import json
from pathlib import Path

def reorder(source,target,items,arm):
 slots=[None]*len(items)
 with Path(source).open(encoding='utf8') as f:
  for line in f:
   value=json.loads(line);pos=value['position']
   if type(pos) is not int or not 0<=pos<len(items) or slots[pos] is not None:raise ValueError('duplicate or invalid position')
   item=items[pos]
   if item['position']!=pos or item['index']!=value['index'] or value['arm']!=arm:raise ValueError('sample identity mismatch')
   slots[pos]=line
 if any(line is None for line in slots):raise ValueError('incomplete predictions')
 with Path(target).open('x',encoding='utf8') as f:
  for line in slots:f.write(line)
 return len(slots)
