"""Finite-step forensic diagnostic; not a method result.

Each state is restarted from a frozen PNG.  For every step size we compare
dynamic (natural) FastV selection with the keep-set frozen at the start.
Updates are sign-gradient descent on a fixed reference sequence plus the
existing Native-FDP term.  Every candidate is checked through real Full
exact/replay before being accepted.
"""
import copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
OUT=ROOT/'week2_attack_v19_finite_step_diagnostic'
V13=ROOT/'week2_attack_v13_branch_reachability_probe'
sys.path[:0]=[str(ROOT),str(ROOT/'week1_mechanism'),str(ROOT/'gate_experiment'),str(V13)]
import run_branch_probe as probe
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project
from week2_attack_v4_answer_aligned.common import keep_hash
from week2_attack_v4_answer_aligned.loss import output_valid
from metrics_task import anls_score

EPS=8/255
STEPS=4
ALPHAS=(0.25/255,0.5/255,1.0/255,2.0/255)
STATES={
 '60700_error': {'sample':'60700','png':V13/'runs/state_retention_v1/60700/AWARE_FREEZE/step_011.png'},
 '60700_recovery': {'sample':'60700','png':ROOT/'week2_attack_v16_corrected_direction_audit/60700_attack_only/step_001.png'},
 '7006_near': {'sample':'7006','png':ROOT/'week2_attack_v15_recovery_path_probe/construction/7006/B/checkpoints/step_010.png'},
 # Clean-start additions are deliberately short development diagnostics.  They
 # test whether the content objective can reproduce a content transition from
 # the canonical input, rather than reusing a previously selected endpoint.
 '2008_clean': {'sample':'2008','png':'CLEAN_AUX'},
 '47450_clean': {'sample':'47450','png':'CLEAN_AUX'},
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(x): return bool(x.get('full_text_exact') and x.get('full_token_exact') and x.get('full_replay'))
def hash_tensor(x): return hashlib.sha256(x.detach().float().cpu().numpy().tobytes()).hexdigest()

def answer_token_positions(processor, reference, answers):
    """Locate the registered answer span in the clean FastV token sequence."""
    tok = processor.tokenizer
    for answer in answers:
        ids = tok(str(answer), add_special_tokens=False).get('input_ids', [])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(x) for x in ids]
        for start in range(max(0, len(reference) - len(ids) + 1)):
            if reference[start:start + len(ids)] == ids:
                return list(range(start, start + len(ids))), ids
    # Some manifest answers concatenate a name while the tokenizer emits the
    # same clean answer as multiple pieces (e.g. RJ + Reynolds).  The clean
    # FastV reference is then the only unambiguous span; exclude EOS and use
    # all remaining reference tokens, recording the fallback in metadata.
    fallback = list(range(max(0, len(reference) - 1)))
    return fallback, [int(reference[i]) for i in fallback]

def evaluate(model,processor,item,state,page,base_page,ref):
    img=page_uint8(base_page,page-base_page)
    attacked=list(item['images']); attacked[item['aux_idx']]=img
    inp=v7d.prepare(processor,attacked,item['meta']['question'])
    spans=v7d.spans_of(inp,int(model.config.image_token_index),len(attacked))
    # Always reconstruct the auxiliary page through the same page->pixel path
    # used by the gradient code.  Using inp['pixel_values'] here silently
    # evaluates a different image than the one represented by ``page``.
    pixels=v7d.pixels_from_page(inp,item['aux_idx'],page)
    toks,kept=v7d.generate_tokens(model,inp,pixels,spans,0.25)
    txt=v7d.decode(processor,torch.tensor(toks)).strip()
    full=base.full_exact_check(model,processor,item,state,img)
    linf=int(np.abs(np.asarray(img,dtype=np.int16)-np.asarray(item['images'][item['aux_idx']],dtype=np.int16)).max())
    return {'text':txt,'tokens':[int(x) for x in toks],'anls':float(anls_score(txt,list(item['meta']['answers']))),'valid':bool(output_valid(txt)),'k':len(kept),'keep_hash':keep_hash(kept),'full_exact':exact(full),'full':full,'linf_uint8':linf,'png_sha256':hashlib.sha256(img.tobytes()).hexdigest(),'image':img,'kept':kept}

def forward_snapshot(model, processor, item, state, page, reference, mode, frozen_kept, core_positions, content_only=False):
    """Record the exact surrogate input used by an update, without gradients."""
    pixels=v7d.pixels_from_page(state['inputs'],item['aux_idx'],page)
    with torch.no_grad():
        natural_tokens,natural_kept=v7d.generate_tokens(model,state['inputs'],pixels,state['spans'],0.25)
        kept=frozen_kept if mode=='frozen' else natural_kept
        keep_idx=v7d.keep_index_from_kept(kept,state['spans'],int(state['inputs']['input_ids'].shape[1]),page.device)
        rows=v7d.fastv_rows(model,state['inputs'],pixels,reference,keep_idx)
        selected_rows=[rows[i] for i in core_positions if i < len(rows)] if content_only else rows
        selected_tokens=[reference[i] for i in core_positions if i < len(reference)] if content_only else reference
        if not selected_rows:
            raise RuntimeError('content-only objective has no core token positions')
        attack=v7d.mean_soft(selected_rows,selected_tokens)
        fdp,_=base.fdp_value(model,state,item,page)
        all_margins=[float(v7d.hard_margin_row(row,t).detach()) for row,t in zip(rows,reference)]
        core_margins=[all_margins[i] for i in core_positions if i < len(all_margins)]
    natural_text=v7d.decode(processor,torch.tensor(natural_tokens)).strip()
    return {'attack_loss':float(attack.detach()),'fdp':float(fdp.detach()),
            'natural_tokens':[int(x) for x in natural_tokens],
            'natural_text':natural_text,
            'natural_keep_hash':keep_hash(natural_kept),'natural_k':len(natural_kept),
            'used_keep_hash':keep_hash(kept),'used_k':len(kept),
            'used_keep':list(kept),
            'all_token_margin_mean':float(sum(all_margins)/len(all_margins)) if all_margins else None,
            'core_positions':list(core_positions),
            'core_token_margins':core_margins,
            'core_token_margin_mean':float(sum(core_margins)/len(core_margins)) if core_margins else None}

def main():
    probe.configure_determinism()
    model,processor=base.load_model_frozen(); model.eval()
    records=[]
    selected={x.strip() for x in os.environ.get('V19_STATES','').split(',') if x.strip()}
    state_items=((k,v) for k,v in STATES.items() if not selected or k in selected)
    for state_name, spec in state_items:
      sample=spec['sample']; item=v7d.load_dev(sample)
      clean_page=v7d.page_to_float(item['images'][item['aux_idx']]).to('cuda')
      clean_inputs=v7d.prepare(processor,item['images'],item['meta']['question'])
      clean_spans=v7d.spans_of(clean_inputs,int(model.config.image_token_index),len(item['images']))
      # Keep the Full sequence for the FDP guard, but use a separate fixed
      # clean FastV sequence for the attack objective.
      clean_full_tokens,_=v7d.generate_tokens(model,clean_inputs,clean_inputs['pixel_values'],clean_spans,None)
      clean_full_tokens=[int(x) for x in clean_full_tokens]
      clean_pixels=v7d.pixels_from_page(clean_inputs,item['aux_idx'],clean_page)
      clean_fast_tokens,clean_fast_kept=v7d.generate_tokens(model,clean_inputs,clean_pixels,clean_spans,0.25)
      clean_fast_tokens=[int(x) for x in clean_fast_tokens]
      with torch.no_grad():
        clean_rows=v7d.full_token_rows(model,clean_inputs,clean_page,item['aux_idx'],clean_full_tokens,False)
        clean_hard=torch.stack([v7d.hard_margin_row(r,t) for r,t in zip(clean_rows,clean_full_tokens)]).detach()
      if spec['png'] == 'CLEAN_AUX':
        start_img=item['images'][item['aux_idx']].convert('RGB')
      else:
        start_img=Image.open(spec['png']).convert('RGB')
      start_page=v7d.page_to_float(start_img).to('cuda')
      attacked=list(item['images']); attacked[item['aux_idx']]=start_img
      inp=v7d.prepare(processor,attacked,item['meta']['question']); spans=v7d.spans_of(inp,int(model.config.image_token_index),len(attacked))
      state={'inputs':inp,'spans':spans,'page':start_page,'base_page':clean_page,
             'clean_tokens':clean_full_tokens,
             'clean_hard':clean_hard,
             'clean_text':v7d.decode(processor,torch.tensor(clean_full_tokens)).strip()}
      with torch.no_grad():
        start_pixels=v7d.pixels_from_page(inp,item['aux_idx'],start_page)
        start_tokens, frozen_kept=v7d.generate_tokens(model,inp,start_pixels,spans,0.25)
      ref=[int(x) for x in clean_fast_tokens]
      core_positions, core_token_ids=answer_token_positions(processor,ref,list(item['meta']['answers']))
      objective=os.environ.get('V19_OBJECTIVE','sequence').strip().lower()
      content_only=objective=='content'
      base_eval=evaluate(model,processor,item,state,start_page,clean_page,ref)
      start_meta={k:v for k,v in base_eval.items() if k not in ('image','kept','full')}; start_meta['reference_tokens']=ref
      start_meta['reference_policy']='clean_fastv_fixed_for_all_states'
      start_meta['start_fastv_tokens']=[int(x) for x in start_tokens]
      start_meta['start_fastv_keep_hash']=keep_hash(frozen_kept)
      start_meta['start_fastv_k']=len(frozen_kept)
      start_meta['core_positions']=core_positions
      start_meta['core_token_ids']=core_token_ids
      modes=tuple(x.strip() for x in os.environ.get('V19_MODES','natural,frozen').split(',') if x.strip())
      alpha_names=os.environ.get('V19_ALPHAS','0.25,0.5,1.0,2.0').split(',')
      alphas=tuple(float(x)/255 for x in alpha_names if x.strip())
      step_count=int(os.environ.get('V19_STEPS',str(STEPS)))
      for mode in modes:
       for alpha in alphas:
        delta=(start_page-clean_page).detach().clone()
        hold_active=False
        for step in range(step_count+1):
          page=(clean_page+delta).clamp(0,1)
          ev=evaluate(model,processor,item,state,page,clean_page,ref)
          row={**start_meta,'state':state_name,'sample':sample,'mode':mode,'alpha':alpha,'step':step,'text':ev['text'],'tokens':ev['tokens'],'anls':ev['anls'],'k':ev['k'],'keep_hash':ev['keep_hash'],'full_exact':ev['full_exact'],'linf_uint8':ev['linf_uint8'],'png_sha256':ev['png_sha256']}
          row['strict_endpoint']=bool(ev['full_exact'] and ev['anls']<0.5 and ev['linf_uint8']>0 and ev['linf_uint8']<=8)
          # Causal state-retention ablation.  This deliberately uses a
          # pre-registered semantic rule only for the 60700 PPBS/PPSB probe;
          # it is not used by the ordinary attack protocol.
          content_failure = (sample == '60700' and ev['text'].strip().lower() in {'ppsb','ppjs','ppjs 14, 1967'})
          if os.environ.get('V19_HOLD_CONTENT','0') == '1' and not hold_active and content_failure and ev['full_exact']:
            hold_active=True
            row['hold_trigger']=True
          else:
            row['hold_trigger']=False
          row['held']=bool(hold_active)
          if step==step_count: records.append(row); break
          if hold_active:
            records.append(row)
            continue
          leaf=(clean_page+delta).clamp(0,1).detach().clone().requires_grad_(True)
          if mode=='natural':
            rows,gen,kept=probe.fast_rows_for_reference(model,state,item,leaf,ref)
          else:
            pixels=v7d.pixels_from_page(state['inputs'],item['aux_idx'],leaf)
            keep_idx=v7d.keep_index_from_kept(frozen_kept,state['spans'],int(state['inputs']['input_ids'].shape[1]),leaf.device)
            rows=v7d.fastv_rows(model,state['inputs'],pixels,ref,keep_idx); gen=ref; kept=frozen_kept
          selected_rows=[rows[i] for i in core_positions if i < len(rows)] if content_only else rows
          selected_tokens=[ref[i] for i in core_positions if i < len(ref)] if content_only else ref
          if not selected_rows:
            raise RuntimeError('content-only objective has no core token positions')
          attack=v7d.mean_soft(selected_rows,selected_tokens)
          fdp,_=base.fdp_value(model,state,item,leaf)
          total=attack+probe.LAMBDA_FULL*fdp
          total.backward(); grad=leaf.grad.detach().clone()
          old=delta.detach().clone(); candidate=(delta - float(alpha)*grad.sign()).detach()
          candidate=(clean_page+candidate).clamp(0,1)-clean_page
          candidate=(candidate.clamp(-EPS,EPS)).detach()
          cand_page=(clean_page+candidate).clamp(0,1)
          cand_img=page_uint8(clean_page,candidate)
          guard=base.full_exact_check(model,processor,item,state,cand_img)
          accepted=exact(guard) and int(np.abs(np.asarray(cand_img,dtype=np.int16)-np.asarray(item['images'][item['aux_idx']],dtype=np.int16)).max())<=8
          row.update({'attack_loss':float(attack.detach()),'fdp':float(fdp.detach()),'grad_hash':hash_tensor(grad),'grad_l2':float(grad.float().norm()),'grad_linf':float(grad.float().abs().max()),'pre_update_delta_hash':hash_tensor(delta),'post_projection_delta_hash':hash_tensor(candidate),'guard_exact':exact(guard),'accepted':bool(accepted)})
          row.update({'gradient_tokens':[int(x) for x in gen],
                      'gradient_keep_hash':keep_hash(kept),'gradient_k':len(kept),
                      'reference_tokens':ref})
          pre_snap=forward_snapshot(model,processor,item,state,leaf.detach(),ref,mode,frozen_kept,core_positions,content_only)
          post_float_snap=forward_snapshot(model,processor,item,state,cand_page.detach(),ref,mode,frozen_kept,core_positions,content_only)
          png_page=v7d.page_to_float(cand_img).to(leaf.device)
          post_png_snap=forward_snapshot(model,processor,item,state,png_page,ref,mode,frozen_kept,core_positions,content_only)
          row.update({'pre_update_snapshot':{k:v for k,v in pre_snap.items() if k!='used_keep'},
                      'post_float_snapshot':{k:v for k,v in post_float_snap.items() if k!='used_keep'},
                      'post_png_snapshot':{k:v for k,v in post_png_snap.items() if k!='used_keep'},
                      'post_float_page_hash':hash_tensor(cand_page),
                      'post_png_page_hash':hash_tensor(png_page)})
          if accepted: delta=candidate
          records.append(row)
          del leaf,rows,attack,fdp,total,grad
          torch.cuda.empty_cache()
    OUT.mkdir(parents=True,exist_ok=True)
    suffix=os.environ.get('V19_TAG','all')
    out=OUT/f'results_{suffix}.json'
    state_meta={}
    for k,v in STATES.items():
      if v['png'] == 'CLEAN_AUX':
        state_meta[k]={'sample':v['sample'],'png':'CLEAN_AUX','sha256':None}
      else:
        state_meta[k]={'sample':v['sample'],'png':str(v['png'].relative_to(ROOT)),'sha256':sha(v['png'])}
    out.write_text(json.dumps({'status':'DONE','objective':os.environ.get('V19_OBJECTIVE','sequence'),'steps':STEPS,'alphas':ALPHAS,'epsilon_uint8':8,'selected_states':sorted(selected) if selected else sorted(STATES),'states':state_meta,'records':records},indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'DONE','records':len(records),'out':str(out)}),flush=True)

if __name__=='__main__': main()
