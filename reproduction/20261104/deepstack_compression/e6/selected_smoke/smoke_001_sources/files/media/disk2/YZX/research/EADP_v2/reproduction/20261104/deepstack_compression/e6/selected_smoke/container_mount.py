"""Make canonical audited model metadata paths available read-only in both stages."""
MOUNT='type=bind,src=/data/models/huggingface/hub,dst=/data/models/huggingface/hub,readonly'
def add(command):
 result=list(command)
 mounts=[result[i+1] for i,x in enumerate(result[:-1]) if x=='--mount']
 if any('dst=/data/models/huggingface/hub' in x for x in mounts):raise ValueError('duplicate audited model mount')
 i=result.index('--workdir');result[i:i]=['--mount',MOUNT]
 return result
def verify_actual(command,inspected):
 from pathlib import Path
 expected=[]
 for i,x in enumerate(command[:-1]):
  if x!='--mount':continue
  parts=command[i+1].split(',');values=dict(y.split('=',1) for y in parts if '=' in y)
  if values.get('type')!='bind' or not values.get('src') or not values.get('dst'):raise ValueError('unsupported mount')
  expected.append((str(Path(values['src']).resolve()),values['dst'],'readonly' not in parts))
 if len({x[1] for x in expected})!=len(expected):raise ValueError('overlapping mount destination')
 mounts=inspected.get('Mounts') or []
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in mounts if x.get('Type')=='bind']
 if (len(mounts)!=len(expected) or sorted(actual)!=sorted(expected) or
     sum(x[1]=='/data/models/huggingface/hub' and x[2] is False for x in actual)!=1):
  raise ValueError('actual container mount inventory differs')
 return len(actual)
