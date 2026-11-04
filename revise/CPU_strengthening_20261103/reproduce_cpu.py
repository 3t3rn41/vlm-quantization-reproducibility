"""Portable cached-data reconstruction. No model imports or network access."""
from pathlib import Path
import json,sys,subprocess,importlib.util,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run(path,*args):subprocess.run([sys.executable,str(ROOT/path),*args],cwd=ROOT,check=True)
def verify():
    count=0
    for backend,path,expected in [('TorchAO','revise/Analysis_2/Major_3/compute_group_intervals.py','revise/Analysis_2/Major_3/group_intervals.json'),('HQQ','revise/Analysis_2/Major_2/compute_hqq_intervals.py','revise/Analysis_2/Major_2/hqq_group_intervals.json')]:
        mod=module('verify_'+backend,ROOT/path)
        records=json.loads((ROOT/expected).read_text(encoding='utf8'))['results']
        for c in records:
            src,g,e,r,f=mod.load(c['cell']);point=mod.metrics(src,e,r,f,np.ones(len(src),int))
            for k,v in c['metrics'].items():
                if k=='n':continue
                assert np.isclose(point[k],v['point'],rtol=0,atol=1e-12),(backend,c['cell'],k);count+=1
    # Recompute every saved interaction interval from the released paired replicates.
    rows=json.loads((ROOT/'revise/CPU_strengthening_20261103/interaction_results.json').read_text(encoding='utf8'))
    for d in rows:
        data=np.load(ROOT/f'revise/CPU_strengthening_20261103/{d["cell"]}_paired_replicates.npz')
        for v in d['results']:
            j=['policy_interaction','target_interaction'].index(v['contrast']);arr=data['contrast_counts'][:,j]
            if v['unit']=='per1000':arr=1000*arr/data['n']
            assert np.allclose(np.quantile(arr,[.025,.975]),v['ci95'],rtol=0,atol=1e-12)
            assert np.allclose(np.quantile(arr,[.003125,.996875]),v['bonferroni_ci'],rtol=0,atol=1e-12)
    print(json.dumps({'status':'PASS','endpoint_point_checks':count,'interaction_interval_reconstructions':32,'gpu_inference':False}))
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['verify','interactions','full','data'],default='verify');a=p.parse_args()
    if a.mode=='interactions':run('revise/CPU_strengthening_20261103/run_interactions.py')
    if a.mode=='full':
        run('revise/Analysis_2/Major_3/compute_group_intervals.py','--replicates','10000','--output','group_intervals_regenerated.json')
        run('revise/Analysis_2/Major_2/compute_hqq_intervals.py','--replicates','10000','--output','hqq_group_intervals_regenerated.json')
        for folder,new,old in [('Analysis_2/Major_3','group_intervals_regenerated.json','group_intervals.json'),('Analysis_2/Major_2','hqq_group_intervals_regenerated.json','hqq_group_intervals.json')]:
            aa=json.loads((ROOT/'revise'/folder/new).read_text())['results'];bb=json.loads((ROOT/'revise'/folder/old).read_text())['results']
            for x,y in zip(aa,bb):
                for k in x['metrics']:assert np.allclose(x['metrics'][k]['ci95'],y['metrics'][k]['ci95'],atol=1e-12,rtol=0)
        run('revise/CPU_backend_audit_20261103/run_cpu_audit.py','--tag','regenerated','--workers','4')
        run('revise/CPU_strengthening_20261103/run_interactions.py')
    if a.mode in ('data','full'):run('revise/CPU_strengthening_20261103/assemble_evidence.py')
    verify()
if __name__=='__main__':main()
