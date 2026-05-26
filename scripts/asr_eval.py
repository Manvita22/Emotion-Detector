import io, pandas as pd
from pathlib import Path
import sys, os
# ensure project root is importable when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import transcribe_audio_fileobj, normalize_transcript

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    import numpy as np
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0]=i
    for j in range(len(h)+1): d[0,j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            if r[i-1]==h[j-1]: cost=0
            else: cost=1
            d[i,j]=min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r), len(h)], len(r)

meta = pd.read_csv('dataset/metadata.csv')
rows = []
count=0
for _, row in meta.iterrows():
    p=Path(row['audio_path'])
    if not p.exists():
        continue
    if count>=10: break
    try:
        b = io.BytesIO(p.read_bytes())
        b.filename = p.name
        hyp = transcribe_audio_fileobj(b)
    except Exception as e:
        hyp = ''
    ref = normalize_transcript(str(row.get('transcript','')).strip().lower())
    hyp2 = normalize_transcript(str(hyp).strip().lower())
    err, ref_len = wer(ref, hyp2)
    rows.append({'file':p.name, 'ref':ref, 'hyp':hyp2, 'err':int(err), 'ref_len':int(ref_len)})
    count += 1

# compute WER
tot_err = sum(r['err'] for r in rows)
tot_ref = sum(r['ref_len'] for r in rows)
print('checked', len(rows), 'files')
print('WER =', (tot_err / tot_ref) if tot_ref>0 else float('nan'))

# print samples where mismatch
mismatch = [r for r in rows if r['ref'] != r['hyp']]
for r in mismatch[:10]:
    print('FILE:', r['file'])
    print('REF:', r['ref'])
    print('HYP:', r['hyp'])
    print('---')

# save report
import json
Path('debug_failures/asr_eval.json').parent.mkdir(parents=True, exist_ok=True)
Path('debug_failures/asr_eval.json').write_text(json.dumps({'checked':len(rows),'wer':(tot_err/tot_ref if tot_ref>0 else None),'rows':rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved debug_failures/asr_eval.json')
