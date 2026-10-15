"""Clean Full/FastV admission for v12 candidates; no perturbation or training."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import torch
ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project'); OUT=ROOT/'week2_attack_v12_true_unseen_admission'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'gate_experiment')); sys.path.insert(0,str(ROOT/'week1_mechanism'))
from adapters.runtime import clear_hooks
from data_mpdoc import iter_meta,list_val_files
from metrics_task import anls_score
from run_causal_v2 import load_model,run_fastv,run_plain
from run_step0_v2 import load_images,select_pages
from week2_attack_v4_answer_aligned.common import RATIO,keep_hash,sha_image,spans_for
from week2_attack_v4_answer_aligned.forward import hook_kept
from week2_attack_v4_answer_aligned.loss import official_keep_count,output_valid
FIELDS=['sample_id','M','evidence_idx','aux_idx','expected_k','k','pred_full','pred_fastv','anls_full','anls_fastv','valid_full','valid_fastv','eligible','keep_hash','page_sha_ok','k_ok','hook_ok','seconds']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--shard',type=int,required=True); p.add_argument('--nshards',type=int,default=4); a=p.parse_args()
 man=json.loads((OUT/'candidate_manifest.json').read_text()); specs=[x for i,x in enumerate(man['samples']) if i%a.nshards==a.shard]
 metas={x['sample_id']:x for x in iter_meta(list_val_files())}; path=OUT/f'clean_screen_s{a.shard}.csv'; done=set()
 if path.exists(): done={x['sample_id'] for x in csv.DictReader(path)}
 pending=[x for x in specs if x['sample_id'] not in done]; model,processor=load_model()
 for q in model.parameters(): q.requires_grad_(False)
 token_id=int(model.config.image_token_index); loaded=load_images([metas[x['sample_id']] for x in pending])
 for spec in pending:
  sid=spec['sample_id']; t=time.time(); images,evidence,order,dropped=select_pages(loaded[sid],metas[sid]['answer_page_idx'],sid,768)
  page_ok=[sha_image(x) for x in images]==spec['page_sha256'] and order==spec['order'] and evidence==spec['evidence_idx']
  clear_hooks(model); full,*_=run_plain(model,processor,images,spec['question'],token_id,False,evidence); fast,_,kept,_=run_fastv(model,processor,images,spec['question'],token_id,RATIO,evidence,spec['aux_idx']); clear_hooks(model)
  inputs=__import__('run_causal_v2',fromlist=['prepare']).prepare(processor,images,spec['question']); spans=spans_for(model,inputs,len(images)); hooked=hook_kept(model,inputs,spans,RATIO); clear_hooks(model)
  k=len(kept); expected=official_keep_count(len(images)*3699,RATIO); af=anls_score(full,spec['answers']); at=anls_score(fast,spec['answers']); vf=output_valid(full); vt=output_valid(fast); ko=(k==expected==spec['expected_k']); ho=(list(hooked)==list(kept)); eligible=bool(spec['M']>1 and vf and vt and af>=.5 and at>=.5 and page_ok and ko and ho and evidence!=spec['aux_idx'])
  row={'sample_id':sid,'M':spec['M'],'evidence_idx':evidence,'aux_idx':spec['aux_idx'],'expected_k':expected,'k':k,'pred_full':full,'pred_fastv':fast,'anls_full':f'{af:.4f}','anls_fastv':f'{at:.4f}','valid_full':int(vf),'valid_fastv':int(vt),'eligible':int(eligible),'keep_hash':keep_hash(list(kept)),'page_sha_ok':int(page_ok),'k_ok':int(ko),'hook_ok':int(ho),'seconds':f'{time.time()-t:.1f}'}
  new=not path.exists();
  with path.open('a',newline='') as h:
   w=csv.DictWriter(h,fieldnames=FIELDS); (w.writeheader() if new else None); w.writerow(row)
  print(json.dumps({'id':sid,'eligible':int(eligible),'full':row['anls_full'],'fastv':row['anls_fastv']}),flush=True); torch.cuda.empty_cache()
if __name__=='__main__': main()
