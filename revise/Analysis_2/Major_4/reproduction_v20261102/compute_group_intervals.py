"""CPU-only paired image-group bootstrap, question-weighted endpoints."""
from pathlib import Path
import os
for thread_var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[thread_var]='1'
import argparse, csv, json, time, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'derived_audit'
CELLS=['internvl3_2b__mmstar','internvl3_2b__ai2d','qwen2_5_vl_7b__mmstar','qwen2_5_vl_7b__ai2d']
def load(cell):
    with (DATA/(cell+'_derived.csv')).open(encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
    return (np.array([int(x['source_row']) for x in r]),np.array([int(x['image_group']) for x in r]),
      np.array([[int(x[b+'_error']) for x in r] for b in ['bf16','int4']]),
      np.array([[float(x[b+'_risk_max']) for x in r] for b in ['bf16','int4']]),
      np.array([x['bf16_prediction']!=x['int4_prediction'] for x in r]))
def metrics(src,e,r,flip,w):
    n=int(w.sum()); pos=np.flatnonzero(w); out={}
    out['n']=n
    for b,name in enumerate(['bf16','int4']):out[name+'_accuracy']=1-float(w@e[b]/n)
    out['delta_accuracy']=out['int4_accuracy']-out['bf16_accuracy'];out['flip']=float(w@flip/n)
    ix=np.repeat(pos,w[pos]);a,b=r[:,ix]
    out['spearman']=float(spearmanr(a,b).statistic);out['kendall_tau_b']=float(kendalltau(a,b).statistic)
    for s,name in enumerate(['bf16','int4']):
        low=np.lexsort((src,r[s]));expanded=np.repeat(low,w[low])
        _,grp=np.unique(-r[s],return_inverse=True)
        for o,target in enumerate(['bf16','int4']):
            p=np.bincount(grp,weights=w*e[o]);z=np.bincount(grp,weights=w*(1-e[o]),minlength=len(p))
            denom=p.sum()*z.sum()
            out[name+'_risk_'+target+'_errors_auroc']=float(np.sum(p*(z.sum()-np.cumsum(z)+.5*z))/denom) if denom else float('nan')
            out[name+'_risk_'+target+'_errors_aurc']=float(np.mean(np.cumsum(e[o,expanded])/np.arange(1,n+1)))
    for q in [.05,.1,.2,.3]:
        k=int(np.ceil(q*n));queues=[]
        for risk in r:
            order=np.lexsort((src,-risk));cum=np.cumsum(w[order]);v=np.zeros(len(src),int);v[order]=np.clip(k-cum+w[order],0,w[order]);queues.append(v)
        u,v=queues; inter=np.minimum(u,v).sum();tag=f'{q:.2f}'
        out['jaccard_'+tag]=float(inter/(2*k-inter));out['net_'+tag]=float((v-u)@e[1]);out['yield_'+tag]=float(100*out['net_'+tag]/k)
    for target in ['bf16','int4']:
        for metric in ['auroc','aurc']:
            out['delta_'+target+'_errors_'+metric]=out['int4_risk_'+target+'_errors_'+metric]-out['bf16_risk_'+target+'_errors_'+metric]
    return out
def job(args):
    ci,B=args;cell=CELLS[ci];src,g,e,r,flip=load(cell);G=int(g.max()+1);n=len(src)
    assert len(np.unique(src))==n and set(g)==set(range(G)) and np.isfinite(r).all()
    point=metrics(src,e,r,flip,np.ones(n,int));rng=np.random.default_rng(np.random.SeedSequence([20261002,ci,1]));values={k:[] for k in point}
    start=time.perf_counter()
    for _ in range(B):
        w=np.bincount(rng.integers(0,G,size=G),minlength=G)[g]
        for k,v in metrics(src,e,r,flip,w).items():values[k].append(v)
    results={k:{'point':point[k],'ci95':np.nanquantile(v,[.025,.975]).tolist(),'invalid_replicates':int(np.sum(~np.isfinite(v)))} for k,v in values.items()}
    old={x['cell']:x for x in json.loads((DATA/'paired_bootstrap_results.json').read_text()) if x['unit']=='image'}[cell]
    checks={}
    if B==10000:
        for q in ['0.05','0.10','0.20','0.30']:
            for metric,oldkey in [('jaccard','jaccard_ci'),('net','net_ci'),('yield','yield_ci')]:
                checks[metric+'_'+q]=bool(np.allclose(results[metric+'_'+q]['ci95'],old['signals']['max'][q][oldkey],atol=1e-12,rtol=0))
        for m in ['auroc','aurc']:checks['delta_'+m]=bool(np.allclose(results['delta_int4_errors_'+m]['ci95'],old['delta_'+m+'_ci'],atol=1e-12,rtol=0))
        assert all(checks.values()),checks
    return {'cell':cell,'n':n,'image_groups':G,'replicates':B,'base_seed':20261002,'seed_sequence':[20261002,ci,1],'unit':'image','interval':'pointwise percentile 95%, unadjusted','endpoint':'question-weighted; recompute k=ceil(q*n*)','tie_rule':'risk then source_row; replicated occurrences retained','metrics':results,'archived_interval_checks':checks,'elapsed_seconds':time.perf_counter()-start}
def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=10000);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',default='group_intervals.json');a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:res=list(pool.map(job,[(i,a.replicates) for i in range(4)]))
    import scipy
    out={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'results':res}
    Path(__file__).with_name(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS',a.replicates,[round(x['elapsed_seconds'],2) for x in res],flush=True)
if __name__=='__main__':main()
