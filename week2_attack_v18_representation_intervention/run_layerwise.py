"""Layer-wise auxiliary hidden-state repair probe; diagnostic only."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
HERE = ROOT/'week2_attack_v18_representation_intervention'
V5B = ROOT/'week2_attack_v5b_expanded_pristine40'
sys.path[:0] = [str(ROOT), str(ROOT/'week1_mechanism'), str(ROOT/'gate_experiment'), str(ROOT/'week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation')]
from adapters.runtime import clear_hooks, install
from metrics_task import anls_score
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import production_generate

IDS=('1025','2008','7006','39196'); RATIO=.25; LAYERS=(1,2,3)

def configure():
 import os
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); os.environ.setdefault('VATTACK_ATTN_IMPLEMENTATION','sdpa'); os.environ.setdefault('VATTACK_DETERMINISTIC_RESIZE','1'); os.environ.setdefault('VATTACK_RESIZE_MODE','bilinear')
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)

def run(model, processor, inputs, spans, layer_idx, replacement=None, capture=False):
 clear_hooks(model); state=install(model,'fastv',RATIO,int(model.config.image_token_index)); state.spans=spans; holder={}; layer=model.model.language_model.layers[layer_idx]
 def hook(module,args,kwargs):
  hidden=args[0]
  if hidden.shape[1]==1 or not spans: return None
  if capture: holder['hidden']=hidden.detach().clone()
  if replacement:
   updated=hidden.clone()
   for page, clean_hidden in replacement.items():
    s,e=spans[page]; updated[:,s:e,:]=clean_hidden[:,s:e,:].to(updated.dtype)
   return (updated,),kwargs
  return None
 h=layer.register_forward_pre_hook(hook,with_kwargs=True,prepend=True)
 try: out=production_generate(model,inputs); return out,list(state.kept_visual),holder.get('hidden')
 finally: h.remove(); clear_hooks(model)

def prep(processor,item,image):
 images=list(item['images']); images[item['aux_idx']]=image; inputs=v7d.prepare(processor,images,item['meta']['question']); spans=v7d.spans_of(inputs,int(item['model_image_token_index']),len(images)); return inputs,spans
def dec(processor,t): return v7d.decode(processor,torch.tensor(t)).strip()
def sha_obj(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main():
 configure(); model,processor=base.load_model_frozen(); model.eval(); rows=[]
 for sid in IDS:
  item=v7d.load_dev(sid); item['model_image_token_index']=int(model.config.image_token_index); clean=item['images'][item['aux_idx']]; attack=Image.open(V5B/'attacks'/'artifacts'/sid/'final.png').convert('RGB')
  ci,cs=prep(processor,item,clean); ai,ass=prep(processor,item,attack)
  if cs!=ass: raise RuntimeError(f'{sid}: spans differ')
  fc=v7d.generate_tokens(model,ci,ci['pixel_values'],cs,None)[0]; fa=v7d.generate_tokens(model,ai,ai['pixel_values'],ass,None)[0]
  row={'sample_id':sid,'full_clean':dec(processor,fc),'full_attack':dec(processor,fa),'layers':{}}
  aux=int(item['aux_idx'])
  for li in LAYERS:
   co,ck,ch=run(model,processor,ci,cs,li,capture=True); ao,ak,ah=run(model,processor,ai,ass,li,capture=True)
   ro,rk,_=run(model,processor,ai,ass,li,replacement={aux:ch})
   txt=dec(processor,ro['tokens']); row['layers'][str(li)]={'clean':dec(processor,co['tokens']),'attack':dec(processor,ao['tokens']),'repair':txt,'k_clean':len(ck),'k_attack':len(ak),'k_repair':len(rk),'repair_anls':float(anls_score(txt,list(item['meta']['answers']))),'repair_tokens':ro['tokens'],'keep_hash_repair':sha_obj(rk)}
  rows.append(row); print(json.dumps({'id':sid,'layers':{k:v['repair'] for k,v in row['layers'].items()}},ensure_ascii=False),flush=True)
 (HERE/'layerwise_intervention.json').write_text(json.dumps({'scope':'diagnostic_only','rows':rows},indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
