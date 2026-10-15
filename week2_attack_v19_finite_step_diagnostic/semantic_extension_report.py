"""Rule-based semantic audit for the small content-objective extension."""
import json, re
from pathlib import Path

HERE = Path(__file__).parent
FILES = [
    HERE/'results_content_60700_nat.json', HERE/'results_content_60700_frozen.json',
    HERE/'results_content_60700_repeat.json', HERE/'results_content_60700_recovery_ext.json',
    HERE/'results_content_7006_nat.json', HERE/'results_content_7006_frozen.json',
    HERE/'results_content_2008_clean.json', HERE/'results_content_47450_clean.json',
]

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

def semantic(sample, text):
    t = norm(text)
    if sample == '60700':
        if t == 'ppbs': return 'ANSWER_PRESERVED'
        if t == 'ppsb': return 'CONTENT_FAILURE'
        if 'ppbs' in t: return 'ANSWER_PRESERVED_AUGMENTED'
        return 'CONTENT_FAILURE'
    if sample == '7006':
        if 'rj reynolds tobacco company' in t: return 'ANSWER_PRESERVED'
        if 'tobacco company' in t: return 'ANSWER_PRESERVED_AUGMENTED'
        return 'CONTENT_FAILURE'
    if sample == '2008':
        if 'march 1968' in t: return 'ANSWER_PRESERVED_AUGMENTED' if t != 'march 1968' else 'ANSWER_PRESERVED'
        if 'february 1968' in t: return 'CONTENT_FAILURE'
        return 'CONTENT_FAILURE'
    if sample == '47450':
        if 'william j darby' in t: return 'ANSWER_PRESERVED_AUGMENTED'
        return 'CONTENT_FAILURE'
    return 'UNSCOPED'

rows=[]
for f in FILES:
    if not f.exists(): continue
    d=json.loads(f.read_text())
    for r in d.get('records',[]):
        if r.get('step') != 4: continue
        rows.append({'file':f.name,'sample':r['sample'],'state':r['state'],'mode':r['mode'],
                     'alpha':r['alpha'],'text':r['text'],'anls':r['anls'],
                     'full_exact':r['full_exact'],'strict_endpoint':r['strict_endpoint'],
                     'semantic_status':semantic(r['sample'],r['text']),
                     'keep_hash':r['keep_hash'],'linf_uint8':r['linf_uint8']})
out=HERE/'semantic_extension_report.json'
out.write_text(json.dumps({'status':'DONE','records':rows},indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'DONE','records':len(rows),'out':str(out)}))
for r in rows: print(r['sample'],r['state'],r['mode'],r['alpha'],repr(r['text']),r['semantic_status'],r['full_exact'])
