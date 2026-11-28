import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"photometric_pairing"))
from measures import read,write,dump,bootstrap,quant,table_rates,preference,cases,result

def run():
    axes=("left","right","up","down");rows=read(HERE/"scores.csv")
    ids,m=preference(rows,read(HERE.parent/"translation/pairs_frozen.csv"),["O",*[f+"_"+a for f in ("ref","edge") for a in axes]])
    details=[];orig=[];ref=[];edge=[];diff=[];sets={}
    for pid in ids:
        o=m[pid,"O"];a=[];b=[]
        for axis in axes:
            r=m[pid,"ref_"+axis];e=m[pid,"edge_"+axis];a.append((o+r)/2);b.append((o+e)/2);diff.append(abs(r-e))
            details.append(dict(pair_id=pid,direction=axis,m0=o,m_ref=r,m_edge=e,ref_minus_edge=r-e,
                       ref_aggregate=a[-1],edge_aggregate=b[-1],ref_outcome=result(a[-1]),edge_outcome=result(b[-1])))
        orig.append([o]*4);ref.append(a);edge.append(b)
    write(HERE/"pair_directions.csv",details)
    o,r,e=map(np.array,(orig,ref,edge))
    for kind,base,cond in (("rescue",o[:,0]<0,lambda x:x>0),("harm",o[:,0]>0,lambda x:x<0),("new_ties",o[:,0]!=0,lambda x:x==0)):
        left=cond(r);right=cond(e);lr=left.any(1);er=right.any(1)
        sets[kind]={k:np.array(ids)[base&pred].tolist() for k,pred in dict(both=lr&er,ref_only=lr&~er,edge_only=~lr&er,
                same_direction=(left&right).any(1),ref_exclusive_direction=(left&~right).any(1),edge_exclusive_direction=(right&~left).any(1)).items()}
    pixels=read(HERE/"pixels.csv")
    with np.load(HERE/"input_masks.npz") as masks:
        assert len(masks)==len(pixels)==2048
        for p in pixels:assert np.unpackbits(masks[p["mask_key"]]).sum()/224**2==float(p["input_changed"])
    checks=read(HERE/"repeated.csv") if (HERE/"repeated.csv").exists() else []
    stats=dict(abs_margin_change=quant(diff),zero=int(np.sum(np.array(diff)==0)),positive=int(sum(x["ref_minus_edge"]>0 for x in details)),
        counts={k:[np.sum(x>0)/4,np.sum(x<0)/4,np.sum(x==0)/4] for k,x in (("ref",r),("edge",e))},
        diff_pp=bootstrap(100*((e>0).mean(1)-(r>0).mean(1))),O_to_ref=table_rates(o,r),O_to_edge=table_rates(o,e),
        ref_to_edge=table_rates(r,e),coverage=sets,examples=cases(ids,r,e),
        pixels={k:quant([float(p[k]) for p in pixels]) for k in ("source_changed","border_rgb_mae","source_mae","input_changed","input_changed_mae","input_mae")},
        repeated_decisions=len(checks)//6,repeated_scores=len(checks),repeat_disagreements=sum(x["original_score"]!=x["repeat_score"] for x in checks))
    dump(HERE/"summary.json",stats)
if __name__=="__main__":run()
