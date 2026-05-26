"""Train a fusion baseline that matches `app.py` expectations.

It combines:
- MFCC statistics from the audio file
- TF-IDF + SVD reduced transcript features

Saved artifact shape:
{
  'model': LogisticRegression,
  'scaler': StandardScaler,
  'label_encoder': LabelEncoder,
  'tfidf': TfidfVectorizer,
  'svd': TruncatedSVD,
}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


EMOTION_SEED_TEXTS = {
    'OAF_angry': [
        'I am furious and fed up.',
        'This is unfair and frustrating.',
        'I am annoyed and irritated right now.',
        'Stop blaming me, I am angry.',
    ],
    'OAF_disgust': [
        'That is disgusting and gross.',
        'I feel repulsed and sick.',
        'The whole thing is nasty and unpleasant.',
        'It turned my stomach.',
    ],
    'OAF_Fear': [
        'I am scared and worried.',
        'This makes me anxious and nervous.',
        'I feel unsafe and frightened.',
        'What if something bad happens?',
    ],
    'OAF_happy': [
        'I am happy and excited.',
        'That is wonderful news.',
        'I feel joyful and proud.',
        'This made me smile and feel great.',
    ],
    'OAF_neutral': [
        'It is a normal day.',
        'Nothing special happened.',
        'I am calm and okay.',
        'This is just an ordinary update.',
    ],
    'OAF_Pleasant_surprise': [
        'Wow, I did not expect that.',
        'That is a delightful surprise.',
        'I am amazed by the result.',
        'This is unexpectedly wonderful news.',
    ],
    'OAF_Sad': [
        'I feel sad and disappointed.',
        'This is heartbreaking and heavy.',
        'I am lonely and upset.',
        'I miss how things used to be.',
    ],
}


def extract_mfcc_stats(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    if y.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def load_samples(sample_path: Path) -> list[str]:
    if not sample_path.exists():
        return []
    lines = []
    for line in sample_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if len(line) >= 8:
            lines.append(line)
    return lines


def infer_label_from_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ('furious', 'angry', 'frustrat', 'annoyed', 'fed up', 'irritated')):
        return 'OAF_angry'
    if any(k in t for k in ('gross', 'disgust', 'repuls', 'sick', 'nasty', 'unpleasant')):
        return 'OAF_disgust'
    if any(k in t for k in ('scared', 'afraid', 'worried', 'anxious', 'frightened', 'terrified')):
        return 'OAF_Fear'
    if any(k in t for k in ('happy', 'joy', 'excited', 'smile', 'proud', 'grateful', 'hopeful')):
        return 'OAF_happy'
    if any(k in t for k in ('surprise', 'surprised', 'wow', 'amazed', 'unexpected')):
        return 'OAF_Pleasant_surprise'
    if any(k in t for k in ('sad', 'lonely', 'heartbroken', 'upset', 'disappointed', 'heavy', 'miss')):
        return 'OAF_Sad'
    return 'OAF_neutral'


def build_rows(metadata_path: Path, samples_path: Path):
    df = pd.read_csv(metadata_path)
    rows = []

    # Curated sentence augmentation for transcript branch
    for label, texts in EMOTION_SEED_TEXTS.items():
        for text in texts:
            rows.append({'label': label, 'transcript': text, 'audio_path': None, 'source': 'seed'})

    sample_lines = load_samples(samples_path)
    for text in sample_lines:
        rows.append({'label': infer_label_from_text(text), 'transcript': text, 'audio_path': None, 'source': 'sample'})

    # Real paired audio+transcript rows from metadata
    for _, r in df.iterrows():
        audio_path = Path(str(r['audio_path']))
        if not audio_path.exists():
            continue
        transcript = str(r.get('transcript', '')).strip()
        if not transcript:
            transcript = audio_path.stem
        rows.append({
            'label': str(r['label']),
            'transcript': transcript,
            'audio_path': str(audio_path),
            'source': 'paired',
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='dataset/metadata.csv')
    parser.add_argument('--samples', default='dataset/text_samples/text_only_samples.txt')
    parser.add_argument('--outdir', default='models/fusion_pipeline')
    parser.add_argument('--max_features', type=int, default=12000)
    parser.add_argument('--svd_components', type=int, default=64)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    samples_path = Path(args.samples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(metadata_path, samples_path)

    # Split only on paired samples so validation reflects real fusion.
    paired = [r for r in rows if r['source'] == 'paired']
    if not paired:
        raise RuntimeError('No paired audio rows found in metadata.csv')

    X_train_rows, X_val_rows = train_test_split(
        paired,
        test_size=0.2,
        random_state=args.random_state,
        stratify=[r['label'] for r in paired],
    )

    # Add all seed/sample text rows to training only.
    train_rows = X_train_rows + [r for r in rows if r['source'] != 'paired']

    tfidf = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), sublinear_tf=True)
    tfidf.fit([r['transcript'] for r in train_rows])
    svd = TruncatedSVD(n_components=min(args.svd_components, max(2, tfidf.transform([r['transcript'] for r in train_rows]).shape[1] - 1)), random_state=args.random_state)
    svd.fit(tfidf.transform([r['transcript'] for r in train_rows]))

    def build_matrix(items):
        audio_feats = []
        text_feats = []
        labels = []
        for r in items:
            ap = r['audio_path']
            if ap is None or not Path(ap).exists():
                continue
            feat = extract_mfcc_stats(ap)
            if feat is None:
                continue
            audio_feats.append(feat)
            text_feats.append(r['transcript'])
            labels.append(r['label'])
        if not audio_feats:
            return None, None, None
        X_audio = np.vstack(audio_feats)
        X_text = svd.transform(tfidf.transform(text_feats))
        X = np.hstack([X_audio, X_text])
        return X, np.array(labels), text_feats

    X_train, y_train, _ = build_matrix(X_train_rows)
    X_val, y_val, _ = build_matrix(X_val_rows)
    if X_train is None or X_val is None:
        raise RuntimeError('Failed to build train/val matrices for fusion training')

    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    yv = le.transform(y_val)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=4000, class_weight='balanced')
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xv)
    report = classification_report(yv, pred, target_names=le.classes_)
    print(report)

    joblib.dump(
        {
            'model': clf,
            'scaler': scaler,
            'label_encoder': le,
            'tfidf': tfidf,
            'svd': svd,
        },
        outdir / 'fusion_baseline.joblib',
    )
    (outdir / 'fusion_report.txt').write_text(report, encoding='utf-8')
    print('Saved', outdir / 'fusion_baseline.joblib')


if __name__ == '__main__':
    main()