"""Paired image-group audit of backend errors, score sources and review budgets.

Uses cached derived predictions only. No model loading or GPU inference.
"""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse,csv,json,sys,time,platform
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CELLS=['internvl3_2b__mmstar','internvl3_2b__ai2d','qwen2_5_vl_7b__mmstar','qwen2_5_vl_7b__ai2d']
SOURCES=['BF16','TorchAO','HQQ'];SIGNALS=['max','entropy','margin'];BUDGETS=[.05,.10,.20,.30]
POLICIES=['BF16 risk','TorchAO risk','HQQ risk','native risk per backend']
PAIRS=[(1,0),(2,0),(2,1)]
def load(cell):
    def csv_rows(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    old=csv_rows(ROOT/f'manuscript/submission/numeric_audit_20261002/{cell}_derived.csv')
    hqq=csv_rows(ROOT/f'revise/Analysis_2/Major_2/derived_audit/{cell}_derived.csv')
    assert len(old)==len(hqq)==(1500 if cell.endswith('mmstar') else 3088)
    for a,b in zip(old,hqq):
        for k in ['source_row','image_group','bf16_prediction','bf16_error']:assert a[k]==b[k],(cell,k)
        for k in ['bf16_risk_'+s for s in SIGNALS]:assert float(a[k])==float(b[k]),(cell,k)
    src=np.array([int(x['source_row']) for x in old]);groups=np.array([int(x['image_group']) for x in old])
    assert len(np.unique(src))==len(src) and set(groups)==set(range(groups.max()+1))
    errors=np.array([[int(x['bf16_error']) for x in old],[int(x['int4_error']) for x in old],[int(x['int4_error']) for x in hqq]])
    risk=np.array([[[float(x['bf16_risk_'+s]) for x in old],[float(x['int4_risk_'+s]) for x in old],[float(x['int4_risk_'+s]) for x in hqq]] for s in SIGNALS])
    assert np.isfinite(risk).all() and set(np.unique(errors))<={0,1}
    orders=np.array([[np.lexsort((src,-r)) for r in sig] for sig in risk])
    return src,groups,errors,risk,orders

def fast(errors,orders,w):
    n=int(w.sum());k=np.ceil(np.array(BUDGETS)*n).astype(int)
    captures=np.zeros((3,4,3,3),dtype=np.int64)
    for si in range(3):
        for source in range(3):
            order=orders[si,source];cumw=np.cumsum(w[order]);idx=np.searchsorted(cumw,k,side='left')
            excess=cumw[idx]-k
            cumerr=np.cumsum(errors[:,order]*w[order],axis=1)
            captures[si,:,source,:]=(cumerr[:,idx]-errors[:,order[idx]]*excess).T
    totals=errors@w
    # new / corrected / shared errors / shared correct, each quantized backend relative to BF16
    composition=np.array([[int(((e==1)&(errors[0]==0))@w),int(((e==0)&(errors[0]==1))@w),int(((e==1)&(errors[0]==1))@w),int(((e==0)&(errors[0]==0))@w)] for e in errors[1:]])
    assert np.all(composition.sum(axis=1)==n)
    assert np.all(totals[1:]==totals[0]+composition[:,0]-composition[:,1])
    assert np.all((captures>=0)&(captures<=k[None,:,None,None]))
    assert np.all(captures<=totals[None,None,None,:])
    return n,k,totals,captures,composition

def slow(errors,orders,w):
    n=int(w.sum());k=np.ceil(np.array(BUDGETS)*n).astype(int);c=np.zeros((3,4,3,3),int)
    for si in range(3):
        for source in range(3):
            expanded=np.repeat(orders[si,source],w[orders[si,source]])
            for qi,kk in enumerate(k):c[si,qi,source]=errors[:,expanded[:kk]].sum(axis=1)
    return c

def stat(point,values):
    v=np.asarray(values,dtype=float);finite=np.isfinite(v)
    return {'point':float(point),'ci95':np.quantile(v[finite],[.025,.975]).tolist() if finite.any() else None,'invalid_replicates':int((~finite).sum())}

def one(job):
    ci,B,tag=job;cell=CELLS[ci];src,g,e,risk,orders=load(cell);n=len(src);G=int(g.max()+1);start=time.perf_counter()
    p=fast(e,orders,np.ones(n,dtype=int))
    # Independent expanded-occurrence check, including all three sources, outcomes and risk signals.
    test_rng=np.random.default_rng(71+ci)
    for _ in range(12):
        w=np.bincount(test_rng.integers(0,G,size=G),minlength=G)[g]
        assert np.array_equal(fast(e,orders,w)[3],slow(e,orders,w))
    cap=np.empty((B,3,4,3,3),dtype=np.int64);tot=np.empty((B,3),dtype=np.int64);comp=np.empty((B,2,4),dtype=np.int64);ns=np.empty(B,dtype=np.int64)
    rng=np.random.default_rng(np.random.SeedSequence([20261002,ci,1]))
    for bi in range(B):
        w=np.bincount(rng.integers(0,G,size=G),minlength=G)[g]
        ns[bi],_,tot[bi],cap[bi],comp[bi]=fast(e,orders,w)
        if (bi+1)%2000==0:print(cell,bi+1,'/',B,flush=True)
    pn,pk,pt,pc,px=p;ks=np.ceil(ns[:,None]*np.array(BUDGETS)[None,:]).astype(int)
    matrices=[];backend=[];source=[];composition=[]
    for ti in [1,2]:
        v={'backend':SOURCES[ti],'total_errors':stat(pt[ti],tot[:,ti]),'total_errors_per1000':stat(1000*pt[ti]/pn,1000*tot[:,ti]/ns)}
        for j,name in enumerate(['new_errors_vs_BF16','corrected_errors_vs_BF16','shared_errors','shared_correct']):
            v[name]=stat(px[ti-1,j],comp[:,ti-1,j]);v[name+'_per1000']=stat(1000*px[ti-1,j]/pn,1000*comp[:,ti-1,j]/ns)
        composition.append(v)
    for si,signal in enumerate(SIGNALS):
        for qi,q in enumerate(BUDGETS):
            for ss,sname in enumerate(SOURCES):
                for tt,tname in enumerate(SOURCES):
                    cp=pc[si,qi,ss,tt];cv=cap[:,si,qi,ss,tt];up=pt[tt]-cp;uv=tot[:,tt]-cv
                    recall=np.divide(cv,tot[:,tt],out=np.full(B,np.nan),where=tot[:,tt]!=0)
                    matrices.append({'cell':cell,'signal':signal,'q':q,'k':int(pk[qi]),'risk_source':sname,'error_target':tname,
                      'total_errors':int(pt[tt]),'captured_errors':stat(cp,cv),'error_recall':stat(cp/pt[tt],recall),
                      'unreviewed_errors':stat(up,uv),'unreviewed_per1000':stat(1000*up/pn,1000*uv/ns),
                      'random_recall':float(pk[qi]/pn),'oracle_recall':float(min(pk[qi],pt[tt])/pt[tt])})
            for pi,policy in enumerate(POLICIES):
                st,sh=(pi,pi) if pi<3 else (1,2)
                dt=pt[2]-pt[1];dtv=tot[:,2]-tot[:,1]
                dc=pc[si,qi,sh,2]-pc[si,qi,st,1];dcv=cap[:,si,qi,sh,2]-cap[:,si,qi,st,1]
                du=dt-dc;duv=dtv-dcv
                s=stat(du,duv);assert np.array_equal(duv+dcv,dtv)
                direction='HQQ lower exposure' if s['ci95'][1]<0 else ('TorchAO lower exposure' if s['ci95'][0]>0 else 'direction uncertain')
                backend.append({'cell':cell,'signal':signal,'q':q,'policy':policy,'delta_direction':'HQQ minus TorchAO','delta_total_errors':stat(dt,dtv),
                  'delta_captured_errors':stat(dc,dcv),'delta_unreviewed_errors':s,'delta_unreviewed_per1000':stat(1000*du/pn,1000*duv/ns),
                  'pointwise_interval_direction':direction})
            for tt,target in enumerate(SOURCES):
                for aa,bb in PAIRS:
                    dp=pc[si,qi,aa,tt]-pc[si,qi,bb,tt];dv=cap[:,si,qi,aa,tt]-cap[:,si,qi,bb,tt]
                    source.append({'cell':cell,'signal':signal,'q':q,'error_target':target,'source_a':SOURCES[aa],'source_b':SOURCES[bb],
                      'delta_direction':'source_a minus source_b','delta_captured_errors':stat(dp,dv),
                      'delta_yield_per100':stat(100*dp/pk[qi],100*dv/ks[:,qi]),'delta_unreviewed_per1000':stat(-1000*dp/pn,-1000*dv/ns)})
    checks=[]
    # Preserve the exact streams and recreate previously reported paired intervals.
    if B==10000:
        old=next(x for x in json.loads((ROOT/'manuscript/submission/numeric_audit_20261002/paired_bootstrap_results.json').read_text()) if x['cell']==cell and x['unit']=='image')
        hqq=next(x for x in json.loads((ROOT/'revise/Analysis_2/Major_2/hqq_group_intervals.json').read_text())['results'] if x['cell']==cell)
        for si,sig in enumerate(SIGNALS):
            for qi,q in enumerate(BUDGETS):
                dv=cap[:,si,qi,1,1]-cap[:,si,qi,0,1]
                for name,arr in [('net_ci',dv),('yield_ci',100*dv/ks[:,qi])]:
                    assert np.allclose(np.quantile(arr,[.025,.975]),old['signals'][sig][f'{q:.2f}'][name],atol=1e-12,rtol=0),(cell,sig,q,name)
                    checks.append('TorchAO '+sig+' '+str(q)+' '+name)
        for qi,q in enumerate(BUDGETS):
            dv=cap[:,0,qi,2,2]-cap[:,0,qi,0,2]
            for name,arr in [('net_'+f'{q:.2f}',dv),('yield_'+f'{q:.2f}',100*dv/ks[:,qi])]:
                assert np.allclose(np.quantile(arr,[.025,.975]),hqq['metrics'][name]['ci95'],atol=1e-12,rtol=0)
                checks.append('HQQ '+name)
        oldpoints=json.loads((ROOT/'revise/Analysis_2/Major_1/three_source_cross_target_points.json').read_text())
        expected=next(x for x in oldpoints if x['cell']==cell)['captured_error_matrix'];assert np.array_equal(pc[0,2],expected)
        checks.append('independent 3x3 point matrix')
    output={'cell':cell,'n':n,'image_groups':G,'replicates':B,'seed_sequence':[20261002,ci,1],
      'n_replicate_range':[int(ns.min()),int(ns.max())],'error_composition':composition,
      'cross_matrices':matrices,'backend_contrasts':backend,'risk_source_contrasts':source,
      'verification':{'expanded_occurrence_tests':12,'archived_checks_passed':checks,'identity_checks':'all replicates: U=E-Y; delta U=delta E-delta Y; E_quant=E_BF+new-corrected'},'elapsed_seconds':time.perf_counter()-start}
    path=HERE/tag;path.mkdir(exist_ok=True);(path/(cell+'.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {'cell':cell,'replicates':B,'archive_checks':len(checks),'elapsed_seconds':output['elapsed_seconds']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=10000);p.add_argument('--workers',type=int,default=2);p.add_argument('--tag',default='formal');a=p.parse_args();assert a.replicates>0
    plan={'date':'2026-11-03','design':'post-hoc paired image-group audit','replicates':a.replicates,'seed':20261002,
      'primary':'max-probability risk; q=0.20; HQQ minus TorchAO residual errors; native and fixed-BF16 policies',
      'sensitivity':'3 signals x 4 budgets x 4 policies; all 3 sources x 3 targets; all 3 source pairs',
      'interval':'pointwise 95% percentile, unadjusted for multiplicity','direction_labels':'interval direction on these evaluated data',
      'weighting':'question weighted; draw G image groups with replacement, retain every question occurrence, recompute n* and k*=ceil(q*n*)',
      'ties':'risk descending then source_row ascending; retain repeated occurrences',
      'error_exposure':'benchmark errors outside the selected queue','gpu_inference':False,'python':sys.version,'numpy':np.__version__,'workers':a.workers}
    out=HERE/a.tag;out.mkdir(exist_ok=True);(out/'analysis_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    jobs=[(i,a.replicates,a.tag) for i in range(4)]
    if a.workers==1:results=[one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,jobs))
    (out/'completion.json').write_text(json.dumps({'status':'PASS','results':results},indent=2),encoding='utf-8');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
