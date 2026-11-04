from pathlib import Path
import json,csv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REV=ROOT/'revise'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvout(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def flatten(x,prefix=''):
    o={}
    for k,v in x.items():
        key=prefix+k
        if isinstance(v,dict):o.update(flatten(v,key+'_'))
        elif isinstance(v,list):
            for j,item in enumerate(v):o[key+('_low' if j==0 else '_high' if j==1 else '_'+str(j))]=item
        else:o[key]=v
    return o

def main():
    rows=[]
    for backend,path in [('TorchAO',REV/'Analysis_2/Major_3/group_intervals.json'),('HQQ',REV/'Analysis_2/Major_2/hqq_group_intervals.json')]:
        for cell in load(path)['results']:
            for metric,v in cell['metrics'].items():
                if metric=='n':continue
                assert v['invalid_replicates']==0
                rows.append({'backend':backend,'cell':cell['cell'],'n':cell['n'],'image_groups':cell['image_groups'],'metric':metric,'point':v['point'],'ci95_low':v['ci95'][0],'ci95_high':v['ci95'][1],'replicates':10000,'resampling_unit':'image group','source_file':str(path.relative_to(ROOT))})
    assert len(rows)==240
    csvout(HERE/'complete_image_group_endpoints.csv',rows)
    save(HERE/'complete_image_group_endpoints.json',rows)
    for f in ['cross_matrices','backend_contrasts','risk_source_contrasts','error_composition']:
        csvout(HERE/(f+'_complete.csv'),[flatten(x) for x in load(REV/'CPU_backend_audit_20261103'/f'{f}.json')])
    print('PASS: 240 complete endpoint intervals and five numeric audit outputs.')

if __name__=='__main__':main()
