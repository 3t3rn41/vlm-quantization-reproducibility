"""Post-hoc paired interactions from immutable cached predictions; CPU only."""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,csv,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'CPU_backend_audit_20261103'))
from run_cpu_audit import load,CELLS,fast,slow,stat

def contrasts(c,t):
    # c[source,target], order BF16,TorchAO,HQQ. All differences HQQ minus TorchAO.
    native=(t[2]-c[2,2])-(t[1]-c[1,1])
    fixed=(t[2]-c[0,2])-(t[1]-c[0,1])
    policy=native-fixed
    target=(c[2,2]-c[1,2])-(c[2,1]-c[1,1])
    return np.array([policy,target],dtype=float)

def one(ci):
    cell=CELLS[ci]; src,g,e,risk,orders=load(cell); n=len(g); G=int(g.max()+1)
    def calc(w):
        # Weighted top-k matches occurrence-expanded reranking, including cutoff multiplicity.
        k=int(np.ceil(.2*w.sum())); c=[]
        for order in orders[0]:
            cw=np.cumsum(w[order]);j=int(np.searchsorted(cw,k));excess=cw[j]-k
            c.append((e[:,order]*w[order]).cumsum(axis=1)[:,j]-e[:,order[j]]*excess)
        t=e@w; c=np.array(c);return contrasts(c,t),c,t
    p,pc,pt=calc(np.ones(n,dtype=int))
    test=np.random.default_rng(9103+ci)
    for i in range(16):
        w=np.bincount(test.integers(G,size=G),minlength=G)[g]
        a,c,t=calc(w); ref=slow(e,orders,w)[0,2]
        assert np.array_equal(c,ref)
        assert np.array_equal(a,contrasts(ref,e@w))
    B=10000;values=np.empty((B,2));counts=np.empty(B,dtype=int);capture=np.empty((B,3,3),dtype=int);totals=np.empty((B,3),dtype=int)
    rng=np.random.default_rng(np.random.SeedSequence([20261002,ci,1]))
    for i in range(B):
        w=np.bincount(rng.integers(G,size=G),minlength=G)[g]
        values[i],capture[i],totals[i]=calc(w);counts[i]=w.sum()
    checks=0
    archived=json.loads((HERE.parent/'CPU_backend_audit_20261103/formal'/f'{cell}.json').read_text(encoding='utf8'))
    for row in archived['backend_contrasts']:
        if row['signal']=='max' and row['q']==.2 and row['policy'] in ['native risk per backend','BF16 risk']:
            st,sh=(1,2) if row['policy'].startswith('native') else (0,0)
            uv=(totals[:,2]-capture[:,sh,2])-(totals[:,1]-capture[:,st,1])
            assert np.allclose(np.quantile(uv,[.025,.975]),row['delta_unreviewed_errors']['ci95'],atol=1e-12,rtol=0); checks+=1
    for row in archived['cross_matrices']:
        if row['signal']=='max' and row['q']==.2:
            ss=['BF16','TorchAO','HQQ'].index(row['risk_source']);tt=['BF16','TorchAO','HQQ'].index(row['error_target'])
            assert np.allclose(np.quantile(capture[:,ss,tt],[.025,.975]),row['captured_errors']['ci95'],atol=1e-12,rtol=0);checks+=1
    np.savez_compressed(HERE/f'{cell}_paired_replicates.npz',contrast_counts=values,n=counts,capture=capture,totals=totals)
    out=[]
    for j,name in enumerate(['policy_interaction','target_interaction']):
        for unit,pv,v in [('errors',p[j],values[:,j]),('per1000',1000*p[j]/n,1000*values[:,j]/counts)]:
            row={'cell':cell,'contrast':name,'unit':unit,**stat(pv,v)}
            # Eight contrasts = 2 families x 4 cells. Each marginal interval 99.375%.
            row['bonferroni_ci']=np.quantile(v,[.05/(2*8),1-.05/(2*8)]).tolist()
            out.append(row)
    return {'cell':cell,'n':n,'groups':G,'results':out,'archive_checks':checks,'expanded_occurrence_checks':16}

def main():
    plan={'design':'post-hoc paired interaction analysis; specified after observing earlier aggregate results',
      'gpu_inference':False,'new_prediction_records':0,'replicates':10000,'signal':'max','q':.2,
      'seed_sequence':'[20261002, cell_index, 1]','cell_order':CELLS,
      'policy_interaction':'(U_HQQrisk,HQQ-U_TorchAOrisk,TorchAO)-(U_BF16risk,HQQ-U_BF16risk,TorchAO)',
      'target_interaction':'(Y_HQQrisk,HQQ-Y_TorchAOrisk,HQQ)-(Y_HQQrisk,TorchAO-Y_TorchAOrisk,TorchAO)',
      'uncertainty':'paired image-group percentile; 95% pointwise and 99.375% marginal Bonferroni percentile intervals for eight contrasts (nominal 95% family coverage, bootstrap approximation)',
      'normalization':'1000*contrast/n* within each replicate; same eight contrasts in count and normalized units',
      'weighting':'question-weighted; sampled image-group multiplicities retained; k*=ceil(.2*n*)',
      'selection':'all four model-dataset cells and both named interactions retained',
      'independence':'same historical questions, no new independent validation'}
    (HERE/'analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    with ProcessPoolExecutor(max_workers=4) as pool: results=list(pool.map(one,range(4)))
    (HERE/'interaction_results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    rows=[r for x in results for r in x['results']]
    with (HERE/'interaction_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cell','contrast','unit','point','ci95_low','ci95_high','bonferroni_low','bonferroni_high']);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in ['cell','contrast','unit','point']}|{'ci95_low':r['ci95'][0],'ci95_high':r['ci95'][1],'bonferroni_low':r['bonferroni_ci'][0],'bonferroni_high':r['bonferroni_ci'][1]})
    print(json.dumps({'status':'PASS','archive_checks':sum(x['archive_checks'] for x in results),'expanded_occurrence_checks':sum(x['expanded_occurrence_checks'] for x in results),'results':[r for r in rows if r['unit']=='errors']},indent=2))
if __name__=='__main__':main()
