"""
Scans the `dataset/` folder for audio files and creates `dataset/metadata.csv`.
Columns: audio_path,label,transcript
"""
import os
import csv
from pathlib import Path


AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a'}


def discover(root='dataset'):
    rows = []
    rootp = Path(root)
    for p in rootp.rglob('*'):
        if p.suffix.lower() in AUDIO_EXTS:
            rel_parts = p.relative_to(rootp).parts
            if rel_parts and rel_parts[0] == rootp.name:
                continue
            # label: use parent directory name
            label = p.parent.name
            rel = os.path.relpath(str(p), start=os.getcwd())
            # derive transcript from filename where possible
            # e.g. OAF_back_angry.wav -> back
            stem = p.stem
            parts = stem.split('_')
            transcript = ''
            # common pattern: [speaker]_[word]_[emotion]
            if len(parts) >= 3:
                # take middle parts as transcript
                middle = parts[1:-1]
                transcript = ' '.join(middle)
            elif len(parts) == 2:
                # assume pattern speaker_word
                transcript = parts[1]
            else:
                transcript = stem
            # if transcript token equals label (emotion), clear it
            if transcript.lower() == label.lower():
                transcript = ''
            rows.append({'audio_path': rel, 'label': label, 'transcript': transcript})
    unique_rows = []
    seen = set()
    for row in rows:
        if row['audio_path'] in seen:
            continue
        seen.add(row['audio_path'])
        unique_rows.append(row)
    return unique_rows


def main():
    rows = discover()
    out = Path('dataset/metadata.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_path', 'label', 'transcript'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Wrote {len(rows)} entries to {out}')


if __name__ == '__main__':
    main()
