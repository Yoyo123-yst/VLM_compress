"""Reclassify v19 endpoints without loading a model.

The registered ANLS threshold is retained as a diagnostic field, but the
paper-facing failure field requires the normalized answer core to disappear.
This prevents suffixes, punctuation and formatting changes from being called
semantic attacks.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'week1_mechanism'), str(ROOT / 'gate_experiment')]
from metrics_task import normalize
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d


def core_tokens(answer: str) -> list[str]:
    """Return content-bearing normalized spans for a conservative audit."""
    toks = re.findall(r"[a-z0-9]+", normalize(answer))
    return [t for t in toks if len(t) >= 2]


def classify(record: dict, answers: list[str]) -> dict:
    text = str(record.get('text', ''))
    norm_text = normalize(text)
    aliases = [normalize(a) for a in answers if normalize(a)]
    contained_alias = any(a in norm_text for a in aliases)
    target_cores = [core_tokens(a) for a in answers]
    preserved_core = any(all(tok in norm_text for tok in toks) for toks in target_cores if toks)
    text_exact = any(norm_text == a for a in aliases)
    valid = bool(record.get('valid', False))
    anls_failure = bool(record.get('anls', 1.0) < 0.5)
    full_ok = bool(record.get('full_exact', False))
    if not valid:
        endpoint = 'INVALID_OUTPUT'
    elif full_ok and not preserved_core:
        endpoint = 'CONTENT_FAILURE'
    elif not text_exact:
        endpoint = 'ANSWER_PRESERVED_AUGMENTED'
    else:
        endpoint = 'ANSWER_PRESERVED_EXACT'
    return {
        'text_exact_failure': not text_exact,
        'anls_failure': anls_failure,
        'content_failure': endpoint == 'CONTENT_FAILURE',
        'endpoint_class': endpoint,
        'normalized_text': norm_text,
        'normalized_aliases': aliases,
        'core_tokens': target_cores,
        'core_preserved': preserved_core,
        'alias_contained': contained_alias,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=ROOT / 'week2_attack_v19_finite_step_diagnostic' / 'results_repaired_followup.json')
    ap.add_argument('--output', type=Path, default=ROOT / 'week2_attack_v19_finite_step_diagnostic' / 'results_repaired_followup_reclassified.json')
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    samples = {}
    for rec in data['records']:
        sid = str(rec['sample'])
        if sid not in samples:
            samples[sid] = list(run_v7d.load_dev(sid)['meta']['answers'])
        rec.update(classify(rec, samples[sid]))
    # A state such as 7006_near is already semantically wrong at step 0.  It
    # must not be counted as an attack-induced content failure.  Record the
    # step-0 class for each trajectory and compare later endpoints against it.
    groups = {}
    for rec in data['records']:
        key = (rec['state'], rec['mode'], float(rec['alpha']))
        if int(rec['step']) == 0:
            groups[key] = rec['endpoint_class']
    for rec in data['records']:
        key = (rec['state'], rec['mode'], float(rec['alpha']))
        baseline = groups.get(key)
        rec['baseline_endpoint_class'] = baseline
        rec['induced_content_failure'] = bool(
            rec['content_failure'] and baseline != 'CONTENT_FAILURE' and int(rec['step']) > 0
        )
    counts = {}
    for rec in data['records']:
        key = rec['endpoint_class']
        counts[key] = counts.get(key, 0) + 1
    data['endpoint_policy'] = {
        'version': 'content_core_v1',
        'description': 'ANLS remains diagnostic; CONTENT_FAILURE requires Full exact and disappearance of every registered answer core.',
        'sample_answers': samples,
        'counts': counts,
        'induced_content_failure_count': sum(int(r['induced_content_failure']) for r in data['records']),
    }
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'DONE', 'records': len(data['records']), 'counts': counts, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
