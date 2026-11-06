"""Lazy accepted DEV references. Never opened during preflight or import.

On first request per task, hash and scan the TSV for frozen qid offsets without
CSV-decoding answers; then parse only the requested, already baseline-verified
row. All offsets and source bytes are rechecked for identity. No alternate
reference is selected by model loss or generated answer.
"""
import ast,csv,hashlib
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
class LazyTruth:
    def __init__(self,sources,sample):
        self.sources=sources;self.allowed={(r['task'],str(r['qid'])) for r in sample};self.index={}
    def _index(self,task):
        csv.field_size_limit(100_000_000)
        desc=self.sources[task];path=Path(desc['path'])
        if sha(path)!=desc['sha256']:raise ValueError('reference source changed')
        wanted={qid.encode() for t,qid in self.allowed if t==task};offsets={}
        with path.open('rb') as f:
            header=next(csv.reader([f.readline().decode('utf8')],delimiter='\t'));index=header.index('index')
            if index not in (0,len(header)-1):raise ValueError('unsupported qid column')
            while True:
                offset=f.tell();raw=f.readline()
                if not raw:break
                key=raw.split(b'\t',1)[0] if index==0 else raw.rstrip(b'\r\n').rsplit(b'\t',1)[-1]
                if key[:1]==b'"' and key[-1:]==b'"':key=key[1:-1]
                if key in wanted:
                    if key in offsets:raise ValueError('duplicate reference qid')
                    offsets[key]=offset
        if set(offsets)!=wanted or sha(path)!=desc['sha256']:raise ValueError('reference inventory/source mismatch')
        st=path.stat();self.index[task]=(path,header,offsets,(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns))
    def __call__(self,row):
        key=(row['task'],str(row['qid']))
        if key not in self.allowed:raise ValueError('reference not in frozen sample')
        task,qid=key
        if task not in self.index:self._index(task)
        path,header,offsets,identity=self.index[task];st=path.stat()
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=identity:raise ValueError('reference source changed after indexing')
        with path.open('rb') as f:f.seek(offsets[qid.encode()]);raw=f.readline()
        fields=next(csv.reader([raw.decode('utf8')],delimiter='\t'))
        if len(fields)!=len(header):raise ValueError('incomplete selected row')
        record=dict(zip(header,fields))
        if record['index']!=qid:raise ValueError('reference row mismatch')
        answer=record['answer']
        if task=='AI2D_TEST':return {'answer':answer}
        try:value=ast.literal_eval(answer)
        except (ValueError,SyntaxError):value=answer
        values=[str(x) for x in value] if isinstance(value,list) else [str(value)]
        if not values:raise ValueError('empty first reference')
        return {'answers':values}
