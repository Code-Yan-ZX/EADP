"""Three-anchor descriptive report; requires complete formal and independent review.

No GT, statistical significance, full-device memory claim, or sample exclusion.
"""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];PILOT=BASE/'e6/pixel_natural_timing_pilot_v4';ACCEPT=BASE/'e6/pixel_natural_timing_accept_v4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def summarize(jobs,records,annotations):
 if len(jobs)!=9 or len(records)!=9 or len(annotations)!=18:raise ValueError('incomplete panel')
 bykey={(x['task'],str(x['qid']),x['arm'],x['phase']):x for x in annotations}
 if len(bykey)!=18:raise ValueError('duplicate annotation')
 out=[]
 for job,rows in zip(jobs,records):
  if [x.get('event') for x in rows]!=['model_loading','model_ready','started','raw_generation','completed','started','raw_generation','completed','job_complete'] or any(x.get('job')!=job for x in rows):raise ValueError('job event mismatch')
  rec=rows[7];result=rec['result'];a=bykey[(job['task'],str(job['qid']),job['arm'],'measured')];b=result['boundaries'];mem=rec['memory'];k=result['dynamic_k']
  if (a['t0_ns']!=b['t0_before_image_read_ns'] or a['first_token_ns']!=b['first_token_selected_ns'] or a['generation_end_ns']!=b['generation_complete_ns'] or a.get('dynamic_k')!=k or a.get('health_overlap_status','').find('unknown')<0):raise ValueError('formal annotation mismatch')
  if type(k) is not int or not 1<=k<=1024 or b['first_token_selected_ns']<b['t0_before_image_read_ns']:raise ValueError('invalid measurement')
  out.append({'task':job['task'],'qid':str(job['qid']),'group_id':job['group_id'],'arm':job['arm'],'k':k,
      'ttft_ms':(b['first_token_selected_ns']-b['t0_before_image_read_ns'])/1e6,
      'setup_seconds':rows[1]['setup_wall_ns']/1e9,
      'peak_allocated_gib':mem['max_allocated_bytes']/2**30,'peak_reserved_gib':mem['max_reserved_bytes']/2**30,
      'output_tokens':len(result['tokens']),'eos':result['eos'],'truncated':result['truncated']})
 if len({(x['task'],x['qid'],x['arm']) for x in out})!=9:raise ValueError('duplicate measured job')
 byanchor={}
 for x in out:byanchor.setdefault((x['task'],x['qid'],x['group_id']),{})[x['arm']]=x
 if len(byanchor)!=3 or any(set(x)!={'pixel_t0','pixel_t045','eadp_k256'} for x in byanchor.values()):raise ValueError('unpaired anchors')
 for arms in byanchor.values():
  baseline=arms['eadp_k256']['ttft_ms']
  if baseline<=0:raise ValueError('zero baseline')
  for x in arms.values():x['ttft_relative_to_eadp']=x['ttft_ms']/baseline
 return out

def render(rows,formal_sha,review_sha,background_sha):
 labels={'pixel_t0':'Pixel τ=0','pixel_t045':'Pixel τ=0.45','eadp_k256':'EADP K256'}
 ordered=sorted(rows,key=lambda x:(x['task'],['eadp_k256','pixel_t0','pixel_t045'].index(x['arm'])))
 text=['# 三锚自然首 token 时延试验','',
  '三张已冻结 DEV 图像，每个方法、每图一次测量；下表是单次观察，不代表总体速度，也不构成显著性结论。每臂独立进程，每图三臂随机配对，测量前各有一次同输入预热。', '',
  '| 任务 / 题号 | 方法 | 实际 K | 首 token (ms) | 相对同题 EADP | 模型加载准备 (s) | PyTorch allocated 峰值 (GiB) | PyTorch reserved 峰值 (GiB) |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for x in ordered:
  text.append(f"| {x['task']} / {x['qid']} | {labels[x['arm']]} | {x['k']} | {x['ttft_ms']:.2f} | {x['ttft_relative_to_eadp']:.3f}× | {x['setup_seconds']:.2f} | {x['peak_allocated_gib']:.2f} | {x['peak_reserved_gib']:.2f} |")
 text += ['', '首 token 区间从图像读取与预处理前开始，至已选 token 追加后的同步结束，包含选择器、实际视觉路径和语言模型预填充；模型加载和预热另列。Pixel 在视觉 blocks 前按输入裁剪，EADP 保留完整视觉路径后选择；K 是保留数量，不代表计算量。',
 '', '实际 K 由测量路径的 Python 包装器逐次读取，其开销包含在时延内，尚未单独量化；EADP 轻量适配也有开销。上述峰值仅是整段生成期间 PyTorch 分配器的 allocated/reserved，包含模型常驻，不是 NVML 进程峰值或整卡峰值。各臂自然输出均与本方法已验收参考一致。',
 '', '健康查询确曾发生，但监督日志未记录查询的单调时钟窗口或时钟命名空间，各测次与查询直接重叠未知。另有约15秒一次的 ps/落盘及外部 Java 活动，采样非连续、CPU比例是进程终身均值，不能归因到单个测次；所有测次均保留。',
 '', '正式验收器在GPU运行结束后、解释结果前冻结；已验证9容器及18次生成，并经独立终审。所用参考仅为已接受 DEV 记录，未读取真值或评分。',
 '', f'正式证据 SHA256：`{formal_sha}`；独立复核 SHA256：`{review_sha}`；背景汇总 SHA256：`{background_sha}`。']
 return '\n'.join(text)+'\n'

def terminal_ok(start,terminal):
 return terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and terminal.get('remaining_process_group')==[] and start.get('worker_pid')==terminal.get('worker_pid') and type(start.get('worker_pid')) is int

def admission(plan):
 for path,h in plan['sha256'].items():
  if sha(path)!=h:raise ValueError('report source/proof mismatch')
 formal=read(ACCEPT/'verification002.json');fs=read(ACCEPT/'monitor_verify002/started.json');ft=read(ACCEPT/'monitor_verify002/terminal.json')
 review_path=BASE/'e1_independent_review/pixel_natural_timing_v4_result_review.json';reviewer=BASE/'e1_independent_review/review_pixel_natural_timing_v4_result.py'
 rm=BASE/'e1_independent_review/monitor_pixel_natural_timing_v4_result';review=read(review_path);rs=read(rm/'started.json');rt=read(rm/'terminal.json')
 runstart=PILOT/'monitor_pilot001/started.json';runterm=PILOT/'monitor_pilot001/terminal.json';background=BASE/'e6/pixel_natural_timing_v4_environment/background_summary002.json'
 if (formal.get('status')!='passed' or formal.get('jobs')!=9 or formal.get('generations')!=18 or formal.get('measured')!=9 or formal.get('errors')!=[] or not terminal_ok(fs,ft) or
     review.get('status')!='passed' or review.get('independent_review') is not True or review.get('jobs')!=9 or review.get('generations')!=18 or review.get('measured')!=9 or review.get('errors')!=[] or review.get('GT_opened') is not False or review.get('scoring_performed') is not False or not terminal_ok(rs,rt) or
     review.get('formal_sha256')!=sha(ACCEPT/'verification002.json') or review.get('formal_started_sha256')!=sha(ACCEPT/'monitor_verify002/started.json') or review.get('formal_terminal_sha256')!=sha(ACCEPT/'monitor_verify002/terminal.json') or
     review.get('run_started_sha256')!=sha(runstart) or review.get('run_terminal_sha256')!=sha(runterm) or review.get('reviewer_sha256')!=sha(reviewer) or review.get('background_sha256')!=sha(background)):raise ValueError('formal or independent acceptance failed')
 cmd=rs.get('command',[]);cwd=Path(rs.get('cwd','/'));resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if resolved.count(str(reviewer.resolve()))!=1 or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or resolved[cmd.index('--out')+1]!=str(review_path.resolve()):raise ValueError('independent reviewer command mismatch')
 return formal,review

def main():
 plan=read(HERE/'plan001.json');formal,review=admission(plan)
 pilot_plan=read(PILOT/'plan001.json');jobs=pilot_plan['jobs'];records=[lines(PILOT/'run_pilot001'/('job_%03d'%j['order'])/'observations/journal.jsonl') for j in jobs]
 rows=summarize(jobs,records,formal['annotations']);bg=BASE/'e6/pixel_natural_timing_v4_environment/background_summary002.json';proof=ACCEPT/'verification002.json';ind=BASE/'e1_independent_review/pixel_natural_timing_v4_result_review.json'
 result=render(rows,sha(proof),sha(ind),sha(bg));out=HERE/'RESULT.md'
 with out.open('x') as f:f.write(result);f.flush();os.fsync(f.fileno())
 meta={'status':'completed','measured':len(rows),'task_groups':3,'GT_opened':False,'scoring_performed':False,'all_samples_retained':True,'formal_sha256':sha(proof),'independent_review_sha256':sha(ind),'report_sha256':sha(out),'plan_sha256':sha(HERE/'plan001.json')}
 with (HERE/'report001.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':raise SystemExit(main())
