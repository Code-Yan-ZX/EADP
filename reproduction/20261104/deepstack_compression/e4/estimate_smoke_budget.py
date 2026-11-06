"""Summarize smoke archive-completion intervals; never reads GT or scores.

Filesystem mtimes are proxies for sequential end-of-array writes, not measured
model latency. The first arm is excluded because there is no prior arm archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--run-terminal',type=Path,required=True);p.add_argument('--verify-terminal',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    proof=json.loads(a.acceptance.read_text());term=json.loads(a.run_terminal.read_text());vt=json.loads(a.verify_terminal.read_text())
    if (proof.get('status')!='passed' or proof.get('all_arms_validated')!=256 or proof.get('forced_arms_validated')!=8 or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or
        any(x.get('reason')!='completed' or x.get('worker_returncode')!=0 or x.get('remaining_process_group')!=[] for x in (term,vt))):
        raise ValueError('successful real execution and supervised acceptance required')
    records_path=run/'records.jsonl'
    if proof['files_sha256'].get(str(records_path))!=sha(records_path):raise ValueError('accepted records changed')
    rows=[json.loads(line) for line in records_path.read_text().splitlines()]
    if len(rows)!=256:raise ValueError('unexpected natural arm count')
    grouped={(k,on):[] for k in (128,256,512,1024) for on in (False,True)};last=None;times=[];sizes=[]
    for row in rows:
        path=Path(row['arrays']['path']);stat=path.stat();stamp=stat.st_mtime_ns
        if last is not None:
            elapsed=(stamp-last)/1e9
            if elapsed<0:raise ValueError('archive timestamps are not monotonic')
            grouped[(row['budget'],row['ds_on'])].append(elapsed)
        last=stamp;times.append(stamp);sizes.append(stat.st_size)
    def quantile(values,q):
        data=sorted(values);x=(len(data)-1)*q;i=int(x);j=min(i+1,len(data)-1)
        return data[i]+(data[j]-data[i])*(x-i)
    per={}
    for (budget,on),values in grouped.items():
        if len(values)<31:raise ValueError('insufficient timing proxies')
        per[f'K{budget}-D{int(on)}']={'n':len(values),'mean_seconds':statistics.mean(values),
            'p50_seconds':quantile(values,.5),'p90_seconds':quantile(values,.9),'max_seconds':max(values)}
    means=sum(v['mean_seconds'] for v in per.values())*384
    p50=sum(v['p50_seconds'] for v in per.values())*384;p90=sum(v['p90_seconds'] for v in per.values())*384
    result={'status':'completed','scope':'smoke diagnostic generation plus full tensor archiving; not model performance',
        'run_elapsed_seconds':term['elapsed_seconds'],'verification_elapsed_seconds':vt['elapsed_seconds'],
        'natural_archive_bytes':sum(sizes),'natural_first_to_last_seconds':(times[-1]-times[0])/1e9,'per_arm':per,
        'full_384_arm_interval_extrapolation_seconds':{'sum_means':means,'sum_p50':p50,'sum_p90':p90},
        'limitations':['Archive mtime differences include generation, synchronization, copying, serialization and filesystem effects; mtimes are mutable and are not direct latency measurements.',
                       'The first arm lacks a preceding archive and is omitted. Percentile sums are heuristic, not a confidence interval.',
                       'Full-panel summary-only writes have different overhead; smoke is only a rough planning reference.',
                       'No GT or scores were read.'],
        'acceptance_sha256':sha(a.acceptance),'records_sha256':sha(records_path),'estimator_sha256':sha(__file__)}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'completed','arms':per,'full_seconds':result['full_384_arm_interval_extrapolation_seconds']}))

if __name__=='__main__':main()
