"""One sequential Opus worker re-examines a frozen snapshot of flagged rows."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from audit_v4_agy import RULES, pack, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out/'input_snapshot.json'
    if snapshot.exists():
        rows = json.loads(snapshot.read_text(encoding='utf-8'))
    else:
        sampled = {r['id']: r for r in json.loads((args.audit/'sampled_rows.json').read_text(encoding='utf-8'))}
        flags = json.loads((args.audit/'findings.json').read_text(encoding='utf-8'))
        rows = [{**sampled[f['id']], 'first_review': f} for f in sorted(flags, key=lambda f:f['id'])]
        save(snapshot, rows)
    model = 'claude-opus-4-6-thinking'
    executable = r'C:\Users\andan\AppData\Local\agy\bin\agy.exe'
    results = []
    failures = []
    total = (len(rows)+24)//25
    for index, start in enumerate(range(0,len(rows),25)):
        batch = rows[start:start+25]
        target = args.out/f'batch_{index+1:03d}.json'
        if target.exists():
            result = json.loads(target.read_text(encoding='utf-8'))
        else:
            prompt = RULES + '''\nYou are the senior independent reviewer. These rows were flagged by a cheaper reviewer, which often makes mistakes. Re-evaluate text yourself before consulting its claim. P means the original labels are supported and first review was a false positive; F means a genuine wrong label; U means text is ambiguous or contradictory. For every F/U reason also identify likely root cause: typo destroys cue, contradictory urgency/fragility, compositional template grammar, alias ambiguity, spatial ambiguity, or other. Explain in Vietnamese without accents. Never change data or run training. Only return requested TSV verdict lines. The extra last column is the cheaper review, not ground truth.\n'''
            prompt += '\n'.join(pack(r)+'\tFIRST_REVIEW='+r['first_review']['status']+': '+r['first_review']['reason'] for r in batch)
            assert len(prompt.encode('utf-16-le'))//2 < 29000
            ids = {r['id'] for r in batch}
            result = None
            for attempt in range(2):
                try:
                    proc = subprocess.run([executable,'--mode','plan','--model',model,'--effort','high','--print-timeout','300s','--print',prompt],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=330)
                    (args.out/f'batch_{index+1:03d}_attempt_{attempt}.txt').write_text(proc.stdout+'\nSTDERR:\n'+proc.stderr,encoding='utf-8')
                    verdicts = []
                    for line in proc.stdout.splitlines():
                        p = line.strip().split('\t',2)
                        if len(p)==3 and p[0] in ids and p[1] in ('P','F','U'):
                            verdicts.append({'id':p[0],'status':p[1],'reason':p[2]})
                    if proc.returncode==0 and len(verdicts)==len(ids) and {v['id'] for v in verdicts}==ids:
                        result = {'batch':index+1,'verdicts':verdicts}
                        save(target,result)
                        break
                except subprocess.TimeoutExpired:
                    (args.out/f'batch_{index+1:03d}_timeout_{attempt}.txt').write_text('Timed out',encoding='utf-8')
                time.sleep(2)
            if result is None:
                failures.append(index+1)
        if result:
            results.extend(result['verdicts'])
        save(args.out/'verdicts.json',results)
        save(args.out/'progress.json',{'model':model,'snapshot_rows':len(rows),'reviewed_rows':len(results),'finished_batches':index+1,'total_batches':total,'failures':failures,'verdict_counts':{s:sum(v['status']==s for v in results) for s in ('P','F','U')},'complete':len(results)==len(rows)})
        print(f"Opus reviewed {len(results)}/{len(rows)}; batch {index+1}/{total}; failed={len(failures)}",flush=True)
    # Summarize only the second review; retain every per-row verdict for audit.
    groups = {s:[v for v in results if v['status']==s] for s in ('P','F','U')}
    evidence = {s:items[:35] for s,items in groups.items()}
    summary_prompt = 'Write a Vietnamese report of this independent semantic audit. No tools, no file reads, no training. Counts are for flagged samples only, NOT the full dataset; do not estimate overall accuracy from this selected subset. Explain common true errors, cheaper-review false positives, uncertain cases, root causes and concrete prioritized generator fixes. Cite row IDs. Do not claim fixes were applied. Give recommendations only. Counts: '+json.dumps({s:len(items) for s,items in groups.items()})+'; failed batches: '+str(failures)+'; Illustrative evidence (at most first 35 per category, not all verdicts): '+json.dumps(evidence,ensure_ascii=False)
    summary = subprocess.run([executable,'--mode','plan','--model',model,'--effort','high','--print-timeout','300s','--print',summary_prompt],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=330)
    (args.out/'analysis.md').write_text(summary.stdout,encoding='utf-8')
    (args.out/'summary.stderr.log').write_text(summary.stderr,encoding='utf-8')
    if summary.returncode:
        raise SystemExit(summary.returncode)


if __name__=='__main__':
    main()
