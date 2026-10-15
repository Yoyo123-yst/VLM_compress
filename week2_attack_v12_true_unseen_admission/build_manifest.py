"""Build a clean-admission manifest from candidates never used by prior attacks."""
from __future__ import annotations
import csv, hashlib, json, random, sys
from pathlib import Path
ROOT = Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
OUT = ROOT / 'week2_attack_v12_true_unseen_admission'
V7 = ROOT / 'week2_attack_v7_pristine60'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'gate_experiment')); sys.path.insert(0, str(ROOT/'week1_mechanism'))
from data_mpdoc import iter_meta, list_val_files, parquet_dir
from run_step0_v2 import load_images, select_pages
from week2_attack_v4_answer_aligned.common import MODEL_PATH, RATIO, sha_image
from week2_attack_v4_answer_aligned.loss import official_keep_count

SEED = 2026101201
N = 400
TOKENS = 3699

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist = {r['sample_id'] for r in csv.DictReader((V7/'exclusions/historical_union.csv').open())}
    v7 = {r['sample_id'] for r in csv.DictReader((V7/'frozen_manifest.csv').open())}
    hold = set()
    for line in (ROOT/'week2_attack_v11c_deterministic_native_fdp/holdout_run_manifest.json').read_text().splitlines():
        s=line.strip().strip(',').strip('"')
        if s.isdigit(): hold.add(s)
    prior = hist | v7 | hold | {'1025','2008','47450','60700','7006','39204'}
    catalog = {r['sample_id']: r for r in iter_meta(list_val_files())}
    candidates = [sid for sid in sorted(catalog, key=int) if sid not in prior and len(catalog[sid]['page_ids']) > 1]
    if len(candidates) < N:
        raise SystemExit(f'BLOCKED_INSUFFICIENT_CANDIDATES {len(candidates)}')
    rng = random.Random(SEED); chosen = rng.sample(candidates, N)
    metas = [catalog[sid] for sid in chosen]; pages = load_images(metas); samples=[]
    for meta in metas:
        sid=meta['sample_id']; images,evidence,order,dropped=select_pages(pages[sid], meta['answer_page_idx'], sid, 768)
        if len(images)<=1 or evidence<0: raise SystemExit(f'bad layout {sid}')
        aux=next(i for i in range(len(images)) if i!=evidence)
        samples.append({'sample_id':sid,'shard':meta['shard'],'row':meta['row'],'question':meta['question'],'answers':meta['answers'],'page_ids':meta['page_ids'],'answer_page_idx':meta['answer_page_idx'],'order':order,'dropped':dropped,'M':len(images),'evidence_idx':evidence,'aux_idx':aux,'page_sha256':[sha_image(x) for x in images],'expected_k':official_keep_count(len(images)*TOKENS,RATIO)})
    samples.sort(key=lambda x:int(x['sample_id']))
    payload={'status':'CLEAN_ADMISSION_PENDING','seed':SEED,'n':N,'sampling':'random sample from MP-DocVQA multipage ids excluding historical union, v7 pristine60, v11c dev and v11c prospective holdout','excluded_counts':{'historical_union':len(hist),'v7_pristine60':len(v7),'v11c_holdout':len(hold)},'model_path':MODEL_PATH,'data_source':str(parquet_dir()),'fastv_r':RATIO,'samples':samples}
    raw=json.dumps(payload,sort_keys=True).encode(); payload['manifest_sha256']=hashlib.sha256(raw).hexdigest()
    (OUT/'candidate_manifest.json').write_text(json.dumps(payload,indent=2)+'\n'); (OUT/'candidate_manifest.sha256').write_text(payload['manifest_sha256']+'\n')
    print(json.dumps({'status':payload['status'],'candidates_available':len(candidates),'chosen':N,'excluded':len(prior),'sha256':payload['manifest_sha256']}))
if __name__=='__main__': main()
