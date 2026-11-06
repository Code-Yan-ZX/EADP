"""Exploratory DEV score vs retained visual tokens, not a compute Pareto plot."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B=Path(__file__).resolve().parent.parent;D=B/'e4_dev_figure'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
sources={'eadp':('e4_analysis/score_panel003/analysis.json','e1_independent_review/e4score003_review.json'),
         'pixel0':('e6/pixelprune_dev384_analysis/score001/analysis.json','e1_independent_review/pixel_dev384_scores_review.json'),
         'pixel045':('e6/pixelprune_t045_dev384_analysis/score001/analysis.json','e1_independent_review/pixel_t045_dev384_scores_review.json')}
data={};metadata={}
for name,(f,r) in sources.items():
 proof=read(B/r);assert proof['status']=='passed' and proof['analysis_sha256']==sha(B/f)
 data[name]=read(B/f);metadata[name]={'analysis_sha256':sha(B/f),'review_sha256':sha(B/r)}
tasks=[('DocVQA_VAL','DocVQA'),('TextVQA_VAL','TextVQA'),('AI2D_TEST','AI2D')]
fig,axes=plt.subplots(1,3,figsize=(14,6.2),sharex=True,sharey=True)
fig.subplots_adjust(left=.065,right=.99,bottom=.28,top=.86,wspace=.055)
for ax,(key,title) in zip(axes,tasks):
 e=data['eadp']['tasks'][key];ks=[128,256,512,1024];ys=[e['arm_means_pct'][f'K{k}-D1'] for k in ks]
 lo=[y-e['arm_percentile_95_ci_pct'][f'K{k}-D1'][0] for k,y in zip(ks,ys)];hi=[e['arm_percentile_95_ci_pct'][f'K{k}-D1'][1]-y for k,y in zip(ks,ys)]
 ax.errorbar(ks,ys,yerr=[lo,hi],color='#1f5f8b',marker='o',lw=1.8,capsize=2.5,markersize=4,zorder=2)
 for name,label,color,marker in [('pixel0','pixel_dynamic','#b55328','s'),('pixel045','pixel_t045','#7654a3','D')]:
  v=data[name]['tasks'][key];count=v['dynamic_k'];y=v['arm_means_pct'][label];low,high=v['arm_percentile_95_ci_pct'][label]
  ax.errorbar([count['median']],[y],xerr=[[count['median']-count['min']],[count['max']-count['median']]],yerr=[[y-low],[high-y]],fmt=marker,color=color,elinewidth=1.05,capsize=2.5,markersize=5,zorder=3)
 ax.set_title(f'{title}  ·  n=128',loc='left',fontweight='bold');ax.set_xlim(-10,1080);ax.set_ylim(34,100.5);ax.set_xticks([128,256,512,768,1024]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Retained visual tokens K')
axes[0].set_ylabel('DEV score (0–100)')
handles=[Line2D([0],[0],color='#1f5f8b',marker='o',label='EADP fixed K, DS on'),Line2D([0],[0],color='#b55328',marker='s',ls='none',label='PixelPrune threshold 0'),Line2D([0],[0],color='#7654a3',marker='D',ls='none',label='PixelPrune threshold 0.45')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.105))
fig.suptitle('Exploratory DEV quality and visual token counts',x=.065,y=.965,ha='left',fontweight='bold',fontsize=14)
fig.text(.065,.025,'Vertical bars: unadjusted 95% image-group bootstrap intervals. Pixel points: median K; horizontal bars: min–max.\nPixel prunes after patch embedding and before vision blocks; equal K does not imply equal compute. Threshold 0.45 was chosen after DEV inspection.',fontsize=8.5)
fig.savefig(D/'quality_vs_visual_tokens.png',dpi=180,bbox_inches='tight');fig.savefig(D/'quality_vs_visual_tokens.svg',bbox_inches='tight')
metadata.update({'scope':'DEV exploratory only; K not compute','script_sha256':sha(__file__),'png_sha256':sha(D/'quality_vs_visual_tokens.png'),'svg_sha256':sha(D/'quality_vs_visual_tokens.svg')})
(D/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('completed')
