"""Render the saved numerical snapshot as tables; no image editing or GPU use."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'latest_tables.json').read_text())
ll=['GQA','VizWiz','SQA_IMG','TextVQA','POPE','MME','MMB_EN','MMB_CN','MMVet','Avg7']
qw=['TextVQA','ChartQA','AI2D','OCRBench','HallusionBench','MME','MMB_EN','MMB_CN','DocVQA','InfoVQA','Avg10']
names={'v15':'LLaVA-1.5-7B','next':'LLaVA-NeXT-7B','qwen':'Qwen3-VL-8B'}
labels={'SQA_IMG':r'SQA$^{IMG}$','TextVQA':r'VQA$^{Text}$','POPE':'POPE-F1','MMB_EN':r'MMB$^{EN}$','MMB_CN':r'MMB$^{CN}$','HallusionBench':'HallBench','Avg7':r'Avg$_7$','Avg10':r'Avg$_{10}$'}

def fmt(c,k):
    if c is None or c['value'] is None or c['status']=='protocol_unresolved':return '—'
    s=str(Decimal(str(c['value'])).quantize(Decimal('1') if k=='OCRBench' else Decimal('.1'),rounding=ROUND_HALF_UP))
    if c['status']=='stale_pending_rerun':s+='†'
    if k=='VizWiz' and (c.get('validation_star') or c['status'] not in ['reported_attachment','reported_paper']):s+='*'
    return s

plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'axes.unicode_minus':False})
with PdfPages(OUT/'latest_tables.pdf') as pdf:
    for num,model in enumerate(['v15','next','qwen'],1):
        cols=qw if model=='qwen' else ll
        rows=[];kinds=[];last=None
        for row in d['rows']:
            if row['model']!=model:continue
            budget=row['budget']
            if budget!=last:
                desc='Full model' if not budget else f"Retain {budget*5 if model=='next' else budget} tokens"+(' (nominal)' if model=='next' else '')
                rows.append([desc]+['']*len(cols));kinds.append('budget');last=budget
            method=row['method']
            if method=='AnchorZip (ours)':method='AnchorZip (Ours)'
            rows.append([method]+[fmt(row['cells'].get(k),k) for k in cols]);kinds.append('ours' if method.startswith('AnchorZip') else 'regular')
        height=max(6.1,2.0+.248*(len(rows)+1))
        fig=plt.figure(figsize=(13.8,height),facecolor='white')
        ax=fig.add_axes([.025,1.03/height,.95,(height-1.88)/height]);ax.axis('off')
        tab=ax.table(cellText=rows,colLabels=['Method']+[labels.get(k,k) for k in cols],colWidths=[.23]+[.77/len(cols)]*len(cols),cellLoc='center',loc='center',bbox=[0,0,1,1])
        tab.auto_set_font_size(False);tab.set_fontsize(10.8)
        for (i,j),c in tab.get_celld().items():
            c.set_edgecolor('white');c.set_linewidth(.3);c.PAD=.035
            if i==0:c.set_facecolor('#edf0f4');c.set_text_props(weight='bold',fontsize=9.8);c.set_edgecolor('#8b929b');c.set_linewidth(.5)
            else:
                kind=kinds[i-1]
                if kind=='budget':c.set_facecolor('#dfe2e6');c.set_text_props(style='italic',fontsize=10.2)
                elif kind=='ours':c.set_facecolor('#ecf3ff');c.set_text_props(weight='bold')
            if j==0:c.set_text_props(ha='left')
        fig.text(.025,1-.30/height,f'Table {num}: {names[model]} — updated benchmark comparison',fontsize=16,weight='bold')
        fig.text(.025,1-.62/height,'Snapshot: '+d['snapshot_beijing'].replace('T',' ')+'   |   Reported / reproduced rows kept separate',fontsize=9,color='#505966')
        footer=['† Old complete result; repaired rerun pending.  — Missing or unresolved protocol.  * VizWiz validation.',
                'SQA: shipped CQM-A, 2017 image questions. POPE: mean F1 (2910 / 3000 / 3000). NeXT budgets are nominal.',
                'LLaVA MME: perception; Qwen MME: perception + cognition. Reported baselines are references, not our reproductions.']
        if model=='qwen':footer+=['Avg10 = (8 percentage columns + OCRBench/10 + MME/20)/10. Qwen rows are prior completed legacy runs.',
                                  'EADP K256 HallBench: scoring audited; generation-bank provenance unresolved. No local EADP K128/K512 comparison.']
        else:footer+=['Avg7 = (GQA + SQA-IMG + TextVQA + POPE-F1 + MME/20 + MMB-EN + MMB-CN)/7.',
                      'Local Avg7 remains blank while required MMB results/protocol checks are missing.']
        fig.text(.025,.12/height,'\n'.join(footer),fontsize=8.6,linespacing=1.4,va='bottom',color='#414a56')
        fig.savefig(OUT/f'table_{num}_{model}.png',dpi=150,facecolor='white')
        pdf.savefig(fig)
        plt.close(fig)
print('Rendered three PNG tables and latest_tables.pdf from '+d['snapshot_beijing'])
