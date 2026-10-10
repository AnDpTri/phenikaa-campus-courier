"""Conservative deterministic triage, without changing source rows or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from courier.nlp import synth
from courier.nlp.text import fold


def normalize(text):
    return ' '.join(re.findall(r'[a-z0-9]+', fold(text)))


def phrase_pattern(values):
    phrases = sorted({normalize(s) for s in values}, key=len, reverse=True)
    return re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(s) for s in phrases if s) + r')(?!\w)')


NEG = {
    'urgent': phrase_pattern(synth.URGENT_FALSE),
    'fragile': phrase_pattern(synth.FRAGILE_FALSE),
}
POS = {
    'urgent': phrase_pattern((*synth.URGENT_TRUE, 'can gap', 'khan cap', 'cap toc', 'hoa toc', 'gap nhe', 'dang doi gap')),
    'fragile': phrase_pattern((*synth.FRAGILE_TRUE, 'de vo', 'de be', 'de nut vo', 'mong manh', 'thuy tinh')),
}
WAITING_URGENT = re.compile(r'\bdang (?:doi|cho)(?: [a-z0-9]+){1,10} gap\b')
BAD_COMPOSITION = re.compile(r'\b(?:tai|toi|den|vao|sang|o) (?:hay tim|xac dinh|khoanh vung|truy tim|phan loai|danh dau|chon lua|tim kiem)\b')
TYPES = set(synth.TYPES)
REFS = {'north','south','east','west','near','far','anchor_near','north_most','south_most','east_most','west_most'}


def inspect_row(row):
    hard, review, evidence = [], [], {}
    if not isinstance(row, dict):
        return ['invalid_object'], [], {}
    text = row.get('text')
    if not isinstance(text,str) or not text.strip():
        return ['empty_or_invalid_text'], [], {}
    if re.search(r'[{}]',text):
        hard.append('unresolved_placeholder')
    for role in ('goal','via'):
        spec = row.get(role)
        if spec is None:
            if role=='goal': hard.append('missing_goal')
            continue
        if not isinstance(spec,dict) or not {'type','ref','anchor'} <= spec.keys():
            hard.append(role+'_invalid_schema')
            continue
        kind, ref, anchor = (spec[k] for k in ('type','ref','anchor'))
        if kind not in TYPES|{None} or ref not in REFS|{None} or anchor not in TYPES|{None}:
            hard.append(role+'_invalid_label')
            continue
        map_only = ref == 'anchor_near' or (ref or '').endswith('_most')
        if (map_only and kind is not None) or (not map_only and kind is None):
            hard.append(role+'_type_ref_inconsistent')
        if (ref in {'near','far','anchor_near'}) != (anchor is not None):
            hard.append(role+'_anchor_inconsistent')
    norm = normalize(text)
    for flag in ('urgent','fragile'):
        if type(row.get(flag)) is not bool:
            hard.append(flag+'_invalid_boolean')
            continue
        neg = list(NEG[flag].finditer(norm))
        masked = list(norm)
        for hit in neg:
            masked[hit.start():hit.end()] = ' '*(hit.end()-hit.start())
        remaining = ''.join(masked)
        pos = list(POS[flag].finditer(remaining))
        if flag=='urgent':
            pos += list(WAITING_URGENT.finditer(remaining))
        if neg and pos:
            review.append(flag+'_contradictory_cues')
        elif pos and row[flag] is False:
            review.append(flag+'_positive_cue_false_label')
        elif neg and row[flag] is True:
            review.append(flag+'_negative_cue_true_label')
        if neg or pos:
            evidence[flag] = {'negative':[m.group() for m in neg], 'positive':[m.group() for m in pos]}
    match = BAD_COMPOSITION.search(norm)
    if match:
        review.append('awkward_nested_instruction')
        evidence['composition'] = match.group()
    return hard, review, evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    counts, reasons = Counter(), Counter()
    samples = defaultdict(list)
    seen_text, seen_ids = {}, set()
    hashes = {s:hashlib.sha256() for s in ('retained','review','rejected')}
    source_hash = hashlib.sha256()
    handles = {s:(args.out/(s+'.jsonl')).open('xb') for s in hashes}
    log = (args.out/'filter_reasons.jsonl').open('w',encoding='utf-8')
    try:
        with args.source.open('rb') as source:
            for number, raw in enumerate(source,1):
                source_hash.update(raw)
                try:
                    row = json.loads(raw)
                    hard, suspect, evidence = inspect_row(row)
                except (json.JSONDecodeError,UnicodeDecodeError):
                    row, hard, suspect, evidence = {}, ['invalid_json'], [], {}
                if isinstance(row,dict):
                    ident = row.get('id')
                    if not isinstance(ident,str) or not ident:
                        hard.append('missing_or_invalid_id')
                    elif ident in seen_ids:
                        hard.append('duplicate_id')
                    else: seen_ids.add(ident)
                    if isinstance(row.get('text'),str):
                        # Preserve punctuation: do not conflate different clause boundaries.
                        signature = hashlib.sha256(' '.join(fold(row['text']).split()).encode()).digest()
                        label = json.dumps({k:row.get(k) for k in ('goal','via','urgent','fragile')},sort_keys=True)
                        if signature in seen_text:
                            hard.append('duplicate_text' if seen_text[signature]==label else 'duplicate_text_conflicting_labels')
                        else: seen_text[signature] = label
                status = 'rejected' if hard else 'review' if suspect else 'retained'
                counts[status] += 1
                handles[status].write(raw)
                hashes[status].update(raw)
                if hard or suspect:
                    detail = {'line':number,'id':row.get('id') if isinstance(row,dict) else None,'status':status,'rules':hard+suspect,'evidence':evidence}
                    log.write(json.dumps(detail,ensure_ascii=False)+'\n')
                    for reason in hard+suspect:
                        reasons[reason] += 1
                        if len(samples[reason])<5: samples[reason].append({'row':row,**detail})
                if number%50000==0: print(f'Filtered {number}: {dict(counts)}',flush=True)
    finally:
        log.close()
        for handle in handles.values(): handle.close()
    report = {'source':str(args.source.resolve()),'source_sha256':source_hash.hexdigest(),'total_rows':sum(counts.values()),'counts':dict(counts),'rule_counts':dict(reasons),'output_sha256':{s:h.hexdigest() for s,h in hashes.items()},'limits':'Retained means no implemented rule triggered, not proven correct. Review rows are suspects, not confirmed errors. No label edits, no fuzzy spelling correction, no AGY verdicts used, no training.'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.out/'examples_by_rule.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
