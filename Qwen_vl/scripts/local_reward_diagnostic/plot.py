"""Plot the complete registered reward diagnostic; CPU-only, no model imports.

Reads only the final analysis.json. Missing/undefined estimates are displayed as
NA rather than zero. Figures show an evidence coverage proxy, never QA accuracy.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Qwen_vl/outputs/local_reward_diagnostic'
METRIC_ORDER=['global_raw','global_normalized','span_mean','span_length_weighted','tail20']
LABELS={'global_raw':'Global, raw','global_normalized':'Global, normalized','span_mean':'Fragment mean',
        'span_length_weighted':'Length-weighted mean','tail20':'Lower tail 20%'}
TAIL='#b74736';CONTROL='#376b8c';FIXED='#717b87'


def label(name):
    return LABELS.get(name,name.replace('_',' ').capitalize())


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def boot(values,seed,draws):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return {'mean':None,'ci95':[None,None],'n_documents':0}
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n_documents':len(values)}


def fixed_action_estimates(data):
    """Use exactly the valid documents used by reward-selected mask estimates.

    Current analysis stores fixed-action means but not their intervals. Recover
    the same document Q values from yield.all_details and bootstrap documents;
    never use excluded/truncated-reference pages or independently sampled masks.
    """
    rates=data['action_rates']
    reward_choices=data['primary']['best_mask']['by_reward']
    first=next(iter(reward_choices.values()),{})
    valid=[r['doc_id'] for r in first.get('rows',[])]
    details={r['doc_id']:r for r in data['yield']['all_details']}
    output={}
    for arm,rate in rates.items():
        values=[]
        for doc in valid:
            q=details[doc]['arm_values'][arm]['Q']
            if not finite(q):
                raise ValueError(f'Missing valid-page Q for {doc}/{arm}')
            values.append(q)
        estimate=boot(values,data.get('bootstrap_seed',20261108),data.get('bootstrap_draws',10000))
        expected=rate['page_macro_Q']
        if expected is not None and (estimate['mean'] is None or abs(estimate['mean']-expected)>1e-10):
            raise ValueError(f'Fixed-mask mean disagrees with final analysis: {arm}')
        output[arm]=estimate
    return output


def point(ax,y,estimate,color):
    mean=estimate.get('mean');ci=estimate.get('ci95',[None,None])
    if not finite(mean):
        ax.text(.03,y,'Not estimable',va='center',color='#647384',fontsize=10)
        return
    if len(ci)==2 and all(finite(v) for v in ci):
        if not ci[0]-1e-12<=mean<=ci[1]+1e-12:
            raise ValueError('Estimate outside its reported interval')
        ax.errorbar(mean,y,xerr=[[max(0,mean-ci[0])],[max(0,ci[1]-mean)]],
                    fmt='o',color=color,ecolor=color,capsize=4,markersize=7,linewidth=2)
    else:
        ax.plot(mean,y,'o',color=color,markersize=7)
    ax.annotate(f'{mean:.1%}',(mean,y),xytext=(6,-12),textcoords='offset points',
                fontsize=9,color=color,ha='left' if mean<.90 else 'right')


def render(data,stem):
    if not data.get('complete') or data.get('yield',{}).get('ocr_outputs')!=192:
        raise ValueError('Refusing partial analysis: complete=true and 192 OCR outputs required')
    near=data['primary']['near_tie'];metrics=[m for m in METRIC_ORDER if m in near['concordance']]
    metrics += [m for m in near['concordance'] if m not in metrics]
    choices=data['primary']['best_mask']['by_reward'];fixed=fixed_action_estimates(data)
    verdict=data['verdict'];information_hold='INSUFFICIENT_INFORMATION' in verdict
    if information_hold:
        title='HOLD — insufficient informative documents or mask pairs'
    elif verdict.startswith('HOLD') or 'HOLD' in verdict:
        title='HOLD — '+verdict.replace('_',' ').lower().replace('hold ','')
    elif verdict.startswith('STOP'):
        title='STOP — no incremental lower-tail mechanism established'
    elif verdict=='SIGNAL_FOR_FURTHER_STUDY':
        title='Reward-ranking signal for further study — training effect untested'
    else:
        title='LayoutLite local-reward diagnostic — '+verdict.replace('_',' ').lower()
    title_color='#9a3d27' if information_hold or 'HOLD' in verdict else '#183247'
    nvalid=data['yield']['valid_pages'];ndoc=near['contributing_documents'];npair=near['discordant_pairs']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,(left,right)=plt.subplots(1,2,figsize=(15.7,8.8),gridspec_kw={'width_ratios':[1,1.28]})
    fig.subplots_adjust(left=.15,right=.98,top=.78,bottom=.23,wspace=.65)
    fig.suptitle(title,x=.5,y=.96,fontsize=17,fontweight='bold',color=title_color)
    subtitle=(f'32 frozen pages; {nvalid} valid pages. Near-global-tie ranking: {ndoc} documents / {npair} pairs '
              '(registered gates: ≥12 documents and ≥24 pairs).')
    fig.text(.5,.895,subtitle,ha='center',fontsize=11,color='#384858')
    fig.text(.5,.855,'Evidence coverage proxy; no training',ha='center',fontsize=13,fontweight='bold',color='#183247')
    for y,metric in enumerate(metrics):
        point(left,y,near['concordance'][metric],TAIL if metric=='tail20' else CONTROL)
    left.set_yticks(range(len(metrics)),[label(m) for m in metrics]);left.invert_yaxis()
    left.set_ylim(len(metrics)-.4,-.65)
    left.axvline(.5,color='#8d99a7',linestyle='--',linewidth=1,label='Neutral concordance = 50%')
    left.set_title('Registered near-global-tie ranking',loc='left',pad=18,fontsize=12,fontweight='bold')
    left.set_xlabel('Concordance with evidence coverage proxy\n(document macro; 95% CI; no training)',labelpad=12)
    fixed_names=list(fixed);selection_metrics=[m for m in metrics if m in choices]
    positions=list(range(len(fixed_names)))+[len(fixed_names)+1+i for i in range(len(selection_metrics))]
    names=[];estimates=[];colors=[]
    for index,arm in enumerate(fixed_names):
        desc='Top-K' if arm=='topk' else f'Gumbel {index}' if arm.startswith('gumbel_') else arm
        names.append('Fixed mask: '+desc);estimates.append(fixed[arm]);colors.append(FIXED)
    for metric in selection_metrics:
        names.append('Reward selects: '+label(metric));estimates.append(choices[metric]);colors.append(TAIL if metric=='tail20' else CONTROL)
    for y,estimate,color in zip(positions,estimates,colors):point(right,y,estimate,color)
    right.set_yticks(positions,names);right.invert_yaxis();right.set_ylim(max(positions)+.65,-.7)
    right.axhline(len(fixed_names)-.05,color='#d3dae1',linewidth=1)
    right.set_title('External coverage on the same five-mask action pool',loc='left',pad=18,fontsize=12,fontweight='bold')
    right.set_xlabel('Evidence coverage proxy\n(page macro; 95% CI; no training)',labelpad=12)
    for ax in [left,right]:
        ax.set_xlim(-.015,1.09);ax.set_xticks([0,.25,.5,.75,1.]);ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='x',color='#e6ebf0',linewidth=.8);ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,pad=8)
    y=data['yield']
    footer=(f"Full OCR covered {y['full_reference_covered_QA']}/{y['total_annotated_QA']} annotated answer targets; "
            f"full references truncated: {y['full_truncated']}; masked outputs truncated: {y['compressed_truncated']}.\n"
            f"Dashed line: neutral concordance = 50%. Intervals resample independent documents ({data.get('bootstrap_draws',10000):,} draws).\n"
            'All reward selections use the same five predeclared masks. '
            'Literal answer presence does not establish correct field binding or actual QA performance. '
            '\nNo reward or model was trained on this panel. These small-sample estimates do not establish a positive or negative training effect.')
    fig.text(.15,.085,footer,ha='left',va='top',fontsize=9,color='#465565',linespacing=1.55)
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix('.png'),dpi=180,facecolor='white',bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white',bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',default=str(OUT/'analysis.json'))
    ap.add_argument('--out-stem',default=str(OUT/'results'))
    args=ap.parse_args()
    data=json.loads(Path(args.analysis).read_text())
    render(data,args.out_stem)
    print(str(Path(args.out_stem).with_suffix('.png')))
    print(str(Path(args.out_stem).with_suffix('.pdf')))

if __name__=='__main__':main()
