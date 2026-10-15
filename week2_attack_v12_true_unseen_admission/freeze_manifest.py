"""Freeze a clean-only true-unseen evaluation cohort."""
from __future__ import annotations
import csv, hashlib, json, random
from pathlib import Path
ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
OUT=ROOT/'week2_attack_v12_true_unseen_admission'
SEED=2026101202; N=20
def main():
    man=json.loads((OUT/'candidate_manifest.json').read_text())
    by={x['sample_id']:x for x in man['samples']}; rows=[]
    for path in sorted(OUT.glob('clean_screen_s*.csv')):
        with path.open(newline='') as h: rows.extend(csv.DictReader(h))
    eligible=[r for r in rows if r['eligible']=='1']
    if len(eligible)<N: raise SystemExit(f'BLOCKED_INSUFFICIENT_ELIGIBLE {len(eligible)}')
    rng=random.Random(SEED); chosen=rng.sample(eligible,N); chosen.sort(key=lambda r:int(r['sample_id']))
    samples=[]
    for row in chosen:
        spec=dict(by[row['sample_id']]); spec.update({'clean_full_pred':row['pred_full'],'clean_fastv_pred':row['pred_fastv'],'clean_full_anls':float(row['anls_full']),'clean_fastv_anls':float(row['anls_fastv']),'clean_keep_hash':row['keep_hash'],'clean_k':int(row['k'])}); samples.append(spec)
    payload={'status':'TRUE_UNSEEN_CLEAN_GATE_FROZEN','seed':SEED,'n':N,'source_candidate_manifest_sha256':man['manifest_sha256'],'sampling':'random.Random(2026101202).sample of clean-eligible candidates; no attack outcome used','fastv_r':0.25,'steps':200,'epsilon':8/255,'samples':samples}
    raw=json.dumps(payload,sort_keys=True).encode(); payload['manifest_sha256']=hashlib.sha256(raw).hexdigest()
    (OUT/'true_unseen_manifest.json').write_text(json.dumps(payload,indent=2)+'\n'); (OUT/'true_unseen_manifest.sha256').write_text(payload['manifest_sha256']+'\n')
    with (OUT/'true_unseen_manifest.csv').open('w',newline='') as h:
        fields=['sample_id','M','evidence_idx','aux_idx','expected_k','clean_full_pred','clean_fastv_pred','clean_keep_hash']; w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows({k:s[k] for k in fields} for s in samples)
    print(json.dumps({'status':payload['status'],'eligible':len(eligible),'frozen':N,'ids':[s['sample_id'] for s in samples],'sha256':payload['manifest_sha256']}))
if __name__=='__main__': main()
