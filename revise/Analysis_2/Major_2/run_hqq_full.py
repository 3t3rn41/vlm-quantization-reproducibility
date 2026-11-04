"""HQQ full-manifest extension using frozen VLM input and scoring functions."""
from pathlib import Path
import importlib.util,json,time,sys,os,argparse
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
BASE=None
OUT=None
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 global BASE,OUT
 import torch,numpy as np
 from hqq.core.quantize import BaseQuantizeConfig,HQQLinear,HQQBackend
 import importlib.metadata as md
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--phase',choices=['smoke','formal'],required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--model-path',type=Path);a=p.parse_args()
 BASE=a.base.resolve();OUT=a.output_root.resolve()
 if not (BASE/'scripts/084_latency_memory.py').is_file():raise FileNotFoundError('missing evaluation workspace scripts')
 torch.manual_seed(20261102);np.random.seed(20261102)
 cost=module('cost',BASE/'scripts/084_latency_memory.py');worker=cost.load_worker_module();scorer=module('score',BASE/'scripts/073_run_full_inference.py')
 path=a.model_path or cost.find_model_path(a.model);d=OUT/a.phase/a.model;d.mkdir(parents=True,exist_ok=True)
 model,tok,proc,meta=cost.load_cell(a.model,'bf16',path,worker)
 manifest=[]
 for name,mod in model.named_modules():
  if isinstance(mod,torch.nn.Linear):manifest.append({'name':name,'weight_numel':mod.weight.numel(),'shape':list(mod.weight.shape),'quantized':False,'eligible':('lm_head' not in name and mod.weight.shape[-1]%128==0)})
 dump(d/'module_manifest_bf16.json',manifest)
 config=json.loads((path/'config.json').read_text());dump(d/'source_config.json',config)
 meta.update({'model':a.model,'model_path':str(path),'phase':a.phase,'hqq_version':md.version('hqq'),'torch':torch.__version__,'transformers':md.version('transformers'),'backend':'HQQ_PYTORCH_FORWARD_INT4_g128_axis1','seed':20261102,'calibration_data':False,'target_rule':'Linear input dimension divisible by 128; exclude lm_head','model_num_image_token':getattr(model,'num_image_token',None),'gpu':torch.cuda.get_device_name(0)})
 def infer(row):
  ins=cost.make_inputs(a.model,model,tok,proc,row,BASE,worker)
  ids=[tok.convert_tokens_to_ids(c) for c in 'ABCD'];assert ids==[32,33,34,35],ids
  with torch.inference_mode():o=model(**ins)
  score=scorer.quantize_candidates(o.logits[0,-1,:],ids)
  result={**score,'sample_id':row['sample_id'],'source_row':row['source_row'],'dataset':row['dataset'],'dataset_revision':row['dataset_revision'],'backend':'hqq_int4_g128','prompt_token_count':ins['input_ids'].shape[-1],'visual_token_count':int((ins['image_grid_thw'].prod(-1)//4).sum()) if 'image_grid_thw' in ins else int(ins['pixel_values'].shape[0]*model.num_image_token),'gold_accessed':False}
  del o,ins;return result
 rows={ds:[json.loads(s) for s in (BASE/f'artifacts/eswa_revision_v1/01_manifests/{ds}_full_clean.jsonl').read_text().splitlines()] for ds in ['mmstar','ai2d']}
 assert [len(rows[x]) for x in rows]==[1500,3088]
 if a.phase=='smoke':
  bf=[]
  for ds in rows:
   for row in rows[ds][:3]:bf.append(infer(row))
  dump(d/'bf16_parity_rows.json',bf)
 HQQLinear.set_backend(HQQBackend.PYTORCH_FORWARD)
 qc=BaseQuantizeConfig(nbits=4,group_size=128,quant_zero=False,quant_scale=False,axis=1)
 for r in manifest:
  if not r['eligible']:continue
  name=r['name'];parent,_,child=name.rpartition('.');mod=model.get_submodule(name)
  replacement=HQQLinear(mod,qc,del_orig=True,compute_dtype=torch.bfloat16,device='cuda')
  setattr(model.get_submodule(parent) if parent else model,child,replacement);r['quantized']=True
 count=sum(r['quantized'] for r in manifest);assert count==({'internvl3_2b':294,'qwen2_5_vl_7b':326}[a.model]),count
 meta['quantized_modules']=count;dump(d/'metadata.json',meta);dump(d/'module_manifest_hqq.json',manifest)
 start=time.perf_counter()
 for ds,rs in rows.items():
  file=d/(ds+'_predictions.jsonl');assert not file.exists(),file
  with file.open('w') as f:
   for i,row in enumerate(rs[:3] if a.phase=='smoke' else rs):
    result=infer(row);f.write(json.dumps(result)+'\n');f.flush()
    if (i+1)%50==0:dump(d/'progress.json',{'dataset':ds,'completed':i+1,'elapsed_seconds':time.perf_counter()-start})
 dump(d/'completion.json',{'status':'PASS','elapsed_seconds':time.perf_counter()-start,'counts':{k:3 if a.phase=='smoke' else len(v) for k,v in rows.items()},'max_allocated_bytes':torch.cuda.max_memory_allocated()})
 print(a.model,a.phase,'PASS',flush=True)
if __name__=='__main__':main()
