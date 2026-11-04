"""CPU audit of derived scores/errors; no raw images, gold keys, or model weights.
Run: python reconstruct.py
Optional new paired percentile intervals: python reconstruct.py --bootstrap --seed VALUE
Exact manuscript seeds remain available from the authors upon request.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CELLS=['internvl3_2b__mmstar','internvl3_2b__ai2d','qwen2_5_vl_7b__mmstar','qwen2_5_vl_7b__ai2d']
Q=[.05,.10,.20,.30]
def load(key):
    with (ROOT/(key+'_derived.csv')).open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    n=len(rows);source=np.array([int(r['source_row']) for r in rows]);g=np.array([int(r['image_group']) for r in rows])
    errors=np.array([[int(r[b+'_error']) for r in rows] for b in ['bf16','int4']])
    risks=np.array([[[float(r[b+'_risk_'+s]) for r in rows] for b in ['bf16','int4']] for s in ['max','entropy','margin']])
    assert len(np.unique(source))==n and np.isfinite(risks).all()
    return rows,source,g,errors,risks
def auc(group,e,w):
    p=np.bincount(group,weights=w*e);z=np.bincount(group,weights=w*(1-e),minlength=len(p))
    return float(np.sum(p*(z.sum()-np.cumsum(z)+.5*z))/(p.sum()*z.sum()))
def aurc(order,e,w):
    ex=np.repeat(order,w[order]);return float(np.mean(np.cumsum(e[ex])/np.arange(1,len(ex)+1)))
def queue(order,w,k):
    cumulative=np.cumsum(w[order]);v=np.zeros_like(w);v[order]=np.clip(k-cumulative+w[order],0,w[order]);assert v.sum()==k;return v
def interval(values):return list(np.quantile(values,[.025,.975]))
def bootstrap(index,key,seed,replicates,unit):
    rows,source,g,e,risks=load(key);n=len(source);G=int(g.max())+1
    high=[[np.lexsort((source,-risks[s,b])) for b in range(2)] for s in range(3)]
    low=[np.lexsort((source,risks[0,b])) for b in range(2)];rg=[np.unique(-risks[0,b],return_inverse=True)[1] for b in range(2)]
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,0 if unit=='row' else 1]))
    av,ar,nrs=[],[],[];vv=np.zeros((3,4,replicates));jj=np.zeros_like(vv)
    for rep in range(replicates):
        w=np.bincount(rng.integers(0,n,size=n),minlength=n) if unit=='row' else np.bincount(rng.integers(0,G,size=G),minlength=G)[g]
        nr=int(w.sum());nrs.append(nr)
        av.append(auc(rg[1],e[1],w)-auc(rg[0],e[1],w));ar.append(aurc(low[1],e[1],w)-aurc(low[0],e[1],w))
        for s in range(3):
            for qi,q in enumerate(Q):
                k=math.ceil(q*nr);a,b=[queue(high[s][backend],w,k) for backend in range(2)];vv[s,qi,rep]=100*((b-a)@e[1])/k
                inter=np.minimum(a,b).sum();jj[s,qi,rep]=inter/(2*k-inter)
    return {'cell':key,'unit':unit,'delta_auroc_ci':interval(av),'delta_aurc_ci':interval(ar),'replicate_n_range':[min(nrs),max(nrs)],
        'yield_ci':{s:{str(q):interval(vv[si,qi]) for qi,q in enumerate(Q)} for si,s in enumerate(['max','entropy','margin'])},
        'jaccard_ci':{s:{str(q):interval(jj[si,qi]) for qi,q in enumerate(Q)} for si,s in enumerate(['max','entropy','margin'])}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--replicates',type=int,default=10000);a=p.parse_args()
    if a.bootstrap and a.seed is None:p.error('--seed must be explicitly supplied for bootstrap')
    if a.replicates<1 or (a.seed is not None and a.seed<0):p.error('replicates must be positive; seed must be nonnegative')
    points=json.loads((ROOT/'point_results.json').read_text(encoding='utf-8'))
    for ci,key in enumerate(CELLS):
        rows,source,g,e,risks=load(key);n=len(source);w=np.ones(n,dtype=int)
        for si,signal in enumerate(['max','entropy','margin']):
            high=[np.lexsort((source,-risks[si,b])) for b in range(2)]
            for q in Q:
                k=math.ceil(q*n);aa,bb=[queue(o,w,k) for o in high];inter=int(np.minimum(aa,bb).sum());m=k-inter
                yin=int(np.maximum(bb-aa,0)@e[1]);yout=int(np.maximum(aa-bb,0)@e[1]);net=yin-yout;J=inter/(2*k-inter)
                expected=points[key]['signals'][signal][f'{q:.2f}'];assert net==expected['net'] and abs(J-expected['jaccard'])<1e-12
                assert abs(m/k-(1-J)/(1+J))<1e-12 and (not m or abs(net/k-(m/k)*((yin-yout)/m))<1e-12)
        au=[auc(np.unique(-risks[0,b],return_inverse=True)[1],e[1],w) for b in range(2)]
        ar=[aurc(np.lexsort((source,risks[0,b])),e[1],w) for b in range(2)]
        assert abs((au[1]-au[0])-points[key]['delta_auroc'])<1e-12
        assert abs((ar[1]-ar[0])-points[key]['delta_aurc'])<1e-12
        print(key,'PASS','net@20=',points[key]['signals']['max']['0.20']['net'],'delta AUROC=',au[1]-au[0],'delta AURC=',ar[1]-ar[0])
        if a.bootstrap:
            for unit in ['row','image']:print(json.dumps(bootstrap(ci,key,a.seed,a.replicates,unit)))
    # Same equal-capacity queues and same J; only exchanged error labels vary.
    k,m=4,2;J=(k-m)/(k+m);nets=[enter-1 for enter in [2,1,0]]
    assert J==1/3 and nets==[1,0,-1]
    print('Constructed accounting example PASS: J=1/3, C=1/2, net errors=+1,0,-1. This is not empirical evidence.')
if __name__=='__main__':main()
