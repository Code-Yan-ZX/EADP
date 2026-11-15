"""Small decision/invariant checks, using no experiment outcomes."""
from copy import deepcopy
from analyze import ARMS, METRICS, compare_pairs, diagnostic, verdict, selection
from reward_metrics import self_test


def main():
    self_test()
    pages=[]
    for i in range(12):
        arms={}
        for arm,q in zip(ARMS,[0.,.5,1.,0.,1.]):
            arms[arm]={"Q":q,"rewards":dict(global_raw=.95,global_normalized=.95-.001*q,
                span_mean=.9-.1*q,span_length_weighted=.9-.1*q,tail20=.5+.2*q)}
        pages.append(dict(doc_id=f"independent_doc{i}",arms=arms))
    result=diagnostic(pages)
    near=result["near_tie"]
    assert near["contributing_documents"]==12 and near["discordant_pairs"]>=24
    assert near["concordance"]["global_raw"]["mean"]==.5
    assert near["concordance"]["span_mean"]["mean"]==0
    assert near["concordance"]["tail20"]["mean"]==1
    assert verdict(result,{"reverse_alignment":result},True)=="SIGNAL_FOR_FURTHER_STUDY"
    assert verdict(result,{"reverse_alignment":result},True,False)=="INADEQUATE_INTERFACE"
    assert verdict(result,{"reverse_alignment":result},False)=="PENDING_INCOMPLETE_PANEL"
    few=diagnostic(pages[:2])
    assert verdict(few,{"reverse_alignment":few},True)=="HOLD_INSUFFICIENT_INFORMATION"
    alternate=deepcopy(result)
    alternate["near_tie"]["tail_increment"]["span_mean"]["mean"]=0
    assert verdict(result,{"reverse_alignment":alternate},True)=="ALIGNMENT_SENSITIVE_HOLD"
    tied=deepcopy(pages)
    for p in tied:
        for a in ARMS:p["arms"][a]["rewards"]["span_mean"]=p["arms"][a]["rewards"]["tail20"]
    equal=diagnostic(tied)
    assert verdict(equal,{"reverse_alignment":equal},True)=="STOP_SPECIAL_TAIL_MECHANISM"
    selected=selection(pages)
    assert selected["by_reward"]["global_raw"]["rows"][0]["selected"]==ARMS[0]
    assert selected["by_reward"]["tail20"]["rows"][0]["selected"]==ARMS[2]
    print("document independence, reward ties, bootstrap and decision gates: PASS")


if __name__=="__main__":main()
