"""Independent CPU inspection of M3 proposals; no labels, model or GPU."""
import collections, datetime, hashlib, importlib.util, json, pathlib, random
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('m3_frozen',D/'proposals.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sample=json.loads((D/'sample_001.json').read_text()); entries=sample['examples']; assert len(entries)==128
counts=collections.Counter(x['qid'] for x in entries); duplicates=[x for x,n in counts.items() if n>1]
# rank and pool reconstruction from a deterministic, tie-heavy independent input.
rng=random.Random(115); streams=[[rng.randrange(17) for _ in range(1024)] for _ in range(4)]; selected=rng.sample(range(1024),256); selected_set=set(selected); rejected=sorted(set(range(1024))-selected_set)
expected=[]
for stream in streams:
 order=sorted(range(1024),key=lambda i:(stream[i],-i)); position={v:n for n,v in enumerate(order)}; expected.append([position[i]/1023 for i in range(1024)])
assert all(m.rank(streams[i])==expected[i] for i in range(4));p=m.proposals('independent42',selected,streams)
main=sorted(rejected,key=lambda i:(-expected[0][i],i))[:64];multi=sorted(rejected,key=lambda i:(-sum(a[i] for a in expected)/4,i))[:64]
assert p['pools']['MAIN']==main and p['pools']['MULTI']==multi
seed=lambda *fields:int.from_bytes(hashlib.sha256('|'.join(map(str,('20261104',*fields))).encode()).digest()[:8],'big')
assert p['pools']['RANDOM']==random.Random(seed('independent42','RANDOM','POOL')).sample(rejected,64)
for r in (8,16):
 b=p['by_r'][str(r)]; assert b['removed']==selected[-r:] and b['n_proposals_per_source']==8
 for source,sets in b['sources'].items():
  assert sets[0]==sorted(p['pools'][source][:r]) and len({tuple(x) for x in sets})==8
  for j in range(1,8):assert sets[j]==sorted(random.Random(seed('independent42',source,r,j)).sample(p['pools'][source],r))
  for chosen in sets:assert len(m.replacement(selected,chosen,r))==256 and set(m.replacement(selected,chosen,r))==set(selected[:-r])|set(chosen)
assert p==m.proposals('independent42',selected,streams)
terminal=json.loads((D/'monitor_proposal002/terminal.json').read_text());started=json.loads((D/'monitor_proposal002/started.json').read_text());assert terminal['reason']=='completed' and terminal['worker_returncode']==0 and not terminal['remaining_process_group'] and terminal['worker_pid']==started['worker_pid']
assert 'Ran 3 tests' in (D/'monitor_proposal002/stderr.log').read_text()
proof={'status':'passed','scope':'CPU proposal contract and deterministic synthetic reconstruction only; no actual utility, teacher, natural generation, model identity, costs or final inference approval', 'source_sha256':sha(D/'proposals.py'),'test_source_sha256':sha(D/'test_proposals.py'),'sample_sha256':sha(D/'sample_001.json'),'started_sha256':sha(D/'monitor_proposal002/started.json'),'terminal_sha256':sha(D/'monitor_proposal002/terminal.json'),'reviewer_sha256':sha(__file__),'n_sample':len(entries),'duplicate_qids_across_samples':duplicates,'checks':['four percentile ranks with ties','64 rejected pool each','random pool seed serialization','last r removal','two r x three sources x eight deterministic candidates','256 unique sorted replacements','equal candidate counts'], 'limits':['seed byte encoding fixed here before teacher use','sample frozen after E4 DEV score disclosure','real runner must save Python version, proposals and source hashes','candidate count reduction and retry exhaustion not forced in independent synthetic run','teacher canonical answer, inputs, synchronized main/DS/P3, loss and budgets need separate implementation and validation'], 'GT_opened':False,'predictions_opened':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out=D/'proposals_independent_review.json';assert not out.exists();out.write_text(json.dumps(proof,indent=2)+'\n'); print('passed',sha(out),'duplicate qids',duplicates)
