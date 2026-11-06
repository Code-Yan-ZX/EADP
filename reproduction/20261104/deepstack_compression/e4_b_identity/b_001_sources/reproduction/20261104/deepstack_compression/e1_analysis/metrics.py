"""Author functions plus independently coded checks; no dataset is loaded on import."""
import ast
import copy
import logging
import re
import string
import types
from pathlib import Path
import numpy as np

ROOT=Path('/media/disk2/YZX/research/EADP_v2')
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval'
VQA=AUTHOR/'dataset/utils/vqa_eval.py'
MATCH=AUTHOR/'utils/matching_util.py'
MISC=AUTHOR/'smp/misc.py'
NATIVE_NA={'','#N/A','#N/A N/A','#NA','-1.#IND','-1.#QNAN','-NaN','-nan','1.#IND','1.#QNAN','<NA>','N/A','NA','NULL','NaN','None','n/a','nan','null'}

def _extract(path,names,namespace):
    module=ast.parse(path.read_text(encoding='utf8'))
    nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise ValueError(f'author functions absent: {path}')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

def author_functions():
    def istype(value,typ):
        if isinstance(value,typ): return True
        try:return isinstance(ast.literal_eval(value),typ)
        except (ValueError,SyntaxError,TypeError):return False
    n={'np':np,'istype':istype,'eval':ast.literal_eval,'listinstr':lambda pats,s:any(p in s for p in pats),
       'cp':copy,'os':types.SimpleNamespace(environ={}), 'string':string, 'get_logger':logging.getLogger}
    _extract(MISC,{'process_punctuation'},n)
    _extract(VQA,{'_process_digit_article','levenshtein_distance','anls_compute','process_answer','process_line','hit_calculate'},n)
    _extract(MATCH,{'can_infer_option','can_infer_text','can_infer'},n)
    return n

def native_prediction(raw):
    if not isinstance(raw,str):raise TypeError('raw generated answer must be a string')
    return 'nan' if raw in NATIVE_NA else raw

def _author_constants():
    tree=ast.parse(VQA.read_text(encoding='utf8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_process_digit_article')
    out={}
    for n in fn.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('articles','manualMap','contractions'):
            out[n.targets[0].id]=ast.literal_eval(n.value)
    if set(out)!={'articles','manualMap','contractions'}:raise ValueError('author normalizer constants missing')
    return out

def independent_text_normalize(text,constants):
    text=text.replace('\n',' ').replace('\t',' ').strip()
    initial=text; comma=bool(re.search(r'(\d),(\d)',initial))
    for ch in '; / [ ] " { } ( ) = + \\ _ - > < @ ` , ? !'.split():
        text=text.replace(ch,'' if ch+' ' in initial or ' '+ch in initial or comma else ' ')
    text=re.sub(r'(?<!\d)\.(?!\d)','',text)
    words=[]
    for word in text.lower().split():
        word=constants['manualMap'].get(word,word)
        if word not in constants['articles']:words.append(constants['contractions'].get(word,word))
    return ' '.join(words)

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ch in enumerate(a,1):
        current=[i]
        for j,other in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ch!=other)))
        previous=current
    return previous[-1]

def independent_doc(prediction,answers):
    vals=[]
    for truth in answers:
        a=' '.join(truth.lower().split());b=' '.join(prediction.lower().split())
        d=max(len(truth.upper()),len(prediction.upper()))
        score=1-edit_distance(a,b)/d if d else 1.0
        vals.append(score if score>=0.5 else 0.0)
    return max(vals)

def independent_text(prediction,answers,constants):
    truths=[independent_text_normalize(s,constants) for s in answers]
    n=len(truths)
    if not n:raise ValueError('empty reference list')
    hits=sum(x==independent_text_normalize(prediction,constants) for x in truths)
    return (hits*min(1,max(hits-1,0)/3)+(n-hits)*min(1,hits/3))/n

def independent_option(prediction,choices):
    # Independent re-expression of can_infer, including the text fallback.
    option=None
    if 'Failed to obtain answer via API' not in prediction:
        if any(s in prediction for s in ["Sorry, I can't help with images of people yet.","I can't process this file.","I'm sorry, but without the image provided",'Cannot determine the answer']):
            option='Z'
        else:
            words=re.sub(r'[.()\[\],:;!*#{}]',' ',prediction).split()
            found=[k for k in choices if k in words]
            if len(found)==1 and words.index(found[0])>len(words)-5:option=found[0]
            elif not found and 'Z' in words:option='Z'
    if option:return option
    text=prediction.lower()
    if len(text)>2*sum(len(str(v)) for v in choices.values()):return None
    found=[k for k,v in choices.items() if str(v).lower() in text]
    return found[0] if len(found)==1 else None

def score(task,raw,truth,author,constants,scoring_prediction=None):
    pred=native_prediction(raw) if scoring_prediction is None else scoring_prediction
    if not isinstance(pred,str):raise TypeError('scoring prediction must be a string')
    if task in ('DocVQA_VAL','TextVQA_VAL'):
        answers=truth['answers']
        method='anls' if task=='DocVQA_VAL' else 'vqa_score'
        line={'answer':repr(answers),'prediction':pred}
        result=author['process_line'](line,method=method)
        official=float(author['hit_calculate']([result],task)[0])
        independent=independent_doc(pred,answers) if method=='anls' else independent_text(pred,answers,constants)
        literal=independent_doc(raw,answers) if method=='anls' else independent_text(raw,answers,constants)
        return {'official':official,'independent':independent,'literal_answer_diagnostic':literal,'normalized_prediction':pred}
    choices=truth['choices'];correct=truth['answer']
    inferred=author['can_infer'](pred,copy.deepcopy(choices)) or 'Z'
    independent=independent_option(pred,choices) or 'Z'
    literal=independent_option(raw,choices) or 'Z'
    return {'official':float(inferred==correct),'independent':float(independent==correct),
            'literal_answer_diagnostic':float(literal==correct),'normalized_prediction':pred,
            'official_option':inferred,'independent_option':independent}
