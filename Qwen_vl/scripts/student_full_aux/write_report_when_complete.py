#!/usr/bin/env python3
"""Write the full-run report only once automatic postprocessing is complete.

Consumes no partial predictions. Uses the postprocess success artifact, checks
its hashes, rechecks DB snapshot, and independently rescans 32 x 7 scores.
"""
import argparse, importlib.util, json, os, sqlite3, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]; ROOT=QWEN.parent; OUT=QWEN/'outputs/student_full'
FREEZE=OUT/'run_freeze.json'; DATABASE=OUT/'predictions.sqlite3'
TARGET=ROOT/'docs/student_full_results.md'


def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def pct(x):return f'{100*x:.3f}'


def write_report(status_path):
    helper=mod('full_postprocess_check',HERE/'postprocess_when_complete.py')
    status=json.loads(status_path.read_text())
    if status.get('status')!='complete':raise ValueError('postprocessing is not complete')
    freeze=helper.integrity();fhash=helper.sha(FREEZE)
    if status['freeze_sha256']!=fhash:raise ValueError('postprocess freeze mismatch')
    files=status['outputs']
    for key in ('analysis','sensitivity','report'):
        if helper.sha(files[key+'_path'])!=files[key+'_sha256']:raise ValueError(key+' output hash mismatch')
    digest=helper.snapshot_digest(DATABASE,fhash)
    if digest!=files['ordered_db_rows_sha256']:raise ValueError('database changed since postprocessing')
    a=json.loads(Path(files['analysis_path']).read_text()); s=json.loads(Path(files['sensitivity_path']).read_text())
    if not a['complete'] or not s['complete'] or s['ordered_db_rows_sha256']!=digest:
        raise ValueError('incomplete or inconsistent analyses')
    if set(a['arms'])!=set(helper.ARMS) or len(a['question_scores'])!=5349 or len(a['paired_changes'])!=13 or len(s['comparisons'])!=13:
        raise ValueError('unexpected result coverage')
    if any(a['arms'][arm]['success']!=5349 for arm in helper.ARMS):raise ValueError('missing predictions')
    # Independent source TSV and official scoring path cross-check. It starts
    # only after complete status and all hashes/coverage have been confirmed.
    score=mod('official_full_score',QWEN/'scripts/student_full/score_full.py')
    recs=score.load_manifest(OUT/'docvqa_val_manifest.json',Path(freeze['original_tsv_path']))
    byrow={r['row_index']:r for r in recs}; qrows={r['row_index']:r for r in a['question_scores']}
    con=sqlite3.connect('file:'+str(DATABASE.resolve())+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    sampled=con.execute('SELECT row_index,arm,prediction FROM results WHERE row_index<32 ORDER BY row_index,arm').fetchall()
    if len(sampled)!=32*7:raise ValueError('32-row scoring audit incomplete')
    for i,arm,prediction in sampled:
        if abs(score.anls(byrow[i]['answer'],prediction)-qrows[i]['anls'][arm])>1e-12:
            raise ValueError(f'official scoring discrepancy row {i} arm {arm}')
    sessions=con.execute('SELECT started_unix,finished_unix,wall_seconds,new_success FROM sessions ORDER BY started_unix').fetchall()
    errors=con.execute('SELECT count(*) FROM error_events').fetchone()[0];con.close()
    gate=json.loads((OUT/'dev_gate.json').read_text());audit=json.loads((OUT/'docvqa_val_data_audit.json').read_text())
    if not gate['pass_all'] or not all(x['passed'] for x in gate['checks']) or not audit['complete']:
        raise ValueError('gate or data audit did not pass')
    if any(v[k] for v in audit['student_data_overlap'].values() for k in ('overlap_parent_ids','overlap_image_sha256','overlap_rgb_pixel_sha256','overlap_question_ids')):
        raise ValueError('student split overlap')
    completed_session_seconds=sum(float(x[2]) for x in sessions if x[2] is not None)
    labels={'full':'完整 1024','b2':'B2：EADP 分数 + block8','b1':'B1：EADP 分数 + facility',
            'rres512':'R-res512','magnitude_v__seed_11':'学生 seed 11',
            'magnitude_v__seed_23':'学生 seed 23','magnitude_v__seed_37':'学生 seed 37'}
    lines=['# DocVQA VAL 全量评测结果','',
      '**状态：全量 5,349 问 × 7 臂均已完成。** 主指标为 DocVQA ANLS；三种种子分别报告，所有题目包括截断输出都保留在主分母。', '',
      '本轮是原生 Qwen3-VL-8B、1024→256 的同题对照。学生在推理时只接收图像与问题；magnitude V 表示训练阶段教师标签来自主视觉流，学生实际仍使用主视觉流和三路 DeepStack。', '',
      '## 全量结果','', '| 方法 | ANLS / 100 | 严格字符串准确率 / % | 截断 |', '| --- | ---: | ---: | ---: |']
    for arm in helper.ARMS:
        r=a['arms'][arm];lines.append(f"| {labels[arm]} | {pct(r['anls'])} | {pct(r['literal_exact'])} | {r['truncated']}/5349 |")
    seed=a['magnitude_v_seeds_mean_and_range'];lines += ['',f"三个学生种子的 ANLS 均值为 **{pct(seed['mean'])}**，范围为 **{pct(seed['range'][0])}–{pct(seed['range'][1])}**；没有按本次结果选 seed。",'',
        '## 配对比较','', '下表差值为“后者 − 前者”，单位为 ANLS 百分点。使用 10,000 次配对 bootstrap，按 938 个原始父文档重采样，保留被抽中文档的全部问题并按问题数加权。提升、下降与持平指单题 ANLS 的严格变化，不是二值答对或答错。', '',
        '| 比较 | 差值 | 95% 区间 | 提升 / 下降 / 持平 |', '| --- | ---: | ---: | ---: |']
    for name,r in a['paired_changes'].items():
        lo,hi=r['ci95'];lines.append(f"| {name} | {100*r['delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {r['improved']} / {r['worsened']} / {r['tied']} |")
    lines += ['', '## 截断敏感性','', '以下为双方回答均未截断的题目；各比较的题数可能不同。这是次要分析，不能替代保留所有 5,349 题的主结果。', '',
       '| 比较 | 保留题数 | 仅对照截断 / 仅方法截断 / 双方截断 | 差值 | 95% 区间 |', '| --- | ---: | ---: | ---: | ---: |']
    for name,r in s['comparisons'].items():
        p=r['paired']
        val='—' if p is None else f"{100*p['delta']:+.3f}"
        ci='—' if p is None else f"[{100*p['ci95'][0]:+.3f}, {100*p['ci95'][1]:+.3f}]"
        lines.append(f"| {name} | {r['retained']}/5349 | {r['excluded_ref_only']} / {r['excluded_target_only']} / {r['excluded_both']} | {val} | {ci} |")
    lines += ['', '## 数据、协议和审计','',
       '- 数据是本地 VLMEvalKit 原始 DocVQA_VAL 的全部 5,349 问，来自 938 个父文档、1,286 个页面路径。训练 fit 120、checkpoint 选择用 val 16，以及此前 test 32 和 dev 1 的父文档、题目 ID、原图字节和精确 RGB 像素与该验证集均无重叠。哈希无法排除有损重编码或预训练见过材料；本项目更早的实验也查看过部分 DocVQA_VAL。',
       '- 三个学生 checkpoint 均来自此前 TRAIN 独立 validation proxy 选点，均输入 V、三路 DeepStack 和问题；本次未用验证答案参与选 token。B1/B2 是本项目的原生 Qwen3 适配，不能称为对 EADP 论文原始设置的完整复现。',
       '- 提示词使用仓库中的 Qwen3 VQA 模板，greedy、正常 EOS、上限 128 个新 token。该模板与先前 32 题数值 pilot 的提示词不同，主指标也从 numeric exact 换成 ANLS，两个结果不可直接比较。',
       '- 七臂共用同一源图派生输入；图像特征只在同页、相同预处理和网格下复用。每题和每臂重新进行选集、原生三维位置与语言模型 prefill/KV。两道开发问题覆盖七臂的缓存对新鲜计算及视觉流一致性门禁通过。',
       f"- 运行冻结 SHA256 `{fhash}`；结果库有序快照 SHA256 `{digest}`。全量覆盖、checkpoint/源码/数据冻结、前后处理快照一致性均经校验；另从原始 TSV 按官方计分路径独立复核前 32 题 × 7 臂，共 **{len(sampled)}** 条。当前错误事件历史数为 {errors}；前处理记录的当前状态须为零错误才能完成。",
       '- 位置和选集只保留 SHA256，没有逐题保存选中索引；不能仅根据结果库独立重建全部选集或位置张量。图像输入和逻辑提示词另有可核对产物。',
       '', '## 时间和范围','',
       f"已完成写入会话累计 wall time **{completed_session_seconds:.1f} 秒**（{len(sessions)} 个会话；不含模型加载）；这是同页缓存下的批处理运行时间，不是单请求端到端延迟，也不表示跨页延迟分布。",
       '', '该结果只覆盖 DocVQA_VAL 和此模型、预算与提示词；尚不能推断 TextVQA、OCRBench、其他模型或压缩率上的效果。多项配对比较与先前 pilot 探索均限制推断；请结合点估计、区间与方法成本解释。', '',
       '## 产物','', f"- [完整逐题计分与全部配对比较]({files['analysis_path']})。",
       f"- [截断敏感性与分母]({files['sensitivity_path']})。",
       f"- [自动后处理状态]({status_path.resolve()})、[全量预注册协议]({ROOT/'docs/student_full_prereg.md'})。",
       f"- [提示词审计]({OUT/'prompt_sidecar.json'})；它重建逻辑消息，并非逐题记录的 token IDs。",'']
    # Exclusive creation: do not overwrite a concurrently-written report.
    fd=os.open(TARGET,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:f.write('\n'.join(lines));f.flush();os.fsync(f.fileno())
    print('report complete',TARGET,'sha256',helper.sha(TARGET),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True);p.add_argument('--wait',action='store_true');p.add_argument('--interval',type=int,default=60);args=p.parse_args()
    if args.interval<10:raise ValueError('poll interval must be >=10 seconds')
    if args.wait:
        while not args.status.exists():time.sleep(args.interval)
    if not args.status.exists():raise FileNotFoundError(args.status)
    write_report(args.status)

if __name__=='__main__':main()
