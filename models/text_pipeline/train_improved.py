"""Train a stronger text baseline for emotion labels.

This trainer uses a mix of:
- curated emotion sentences from dataset/text_samples/text_only_samples.txt
- dataset metadata transcripts
- lightweight label-to-text augmentation so the text classifier can generalize
  better to natural speech transcripts.

It saves the same artifact shape expected by app.py:
{
  'model': LogisticRegression,
  'vectorizer': TfidfVectorizer,
  'label_encoder': LabelEncoder
}
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


def _normalize_label(label: str) -> str:
    label = str(label)
    if label.lower() == 'oaf_pleasant_surprise':
        return 'OAF_Pleasant_surprise'
    if label.lower() == 'yaf_pleasant_surprised':
        return 'YAF_pleasant_surprised'
    if label.lower() == 'oaf_fear':
        return 'OAF_Fear'
    if label.lower() == 'yaf_fear':
        return 'YAF_fear'
    return label


def load_curated_samples(sample_file: Path) -> list[str]:
    if not sample_file.exists():
        return []
    texts = []
    for line in sample_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if len(line) >= 8:
            texts.append(line)
    return texts


def build_training_rows(metadata_path: Path, sample_file: Path, seed: int = 42):
    random.seed(seed)
    df = pd.read_csv(metadata_path)
    rows = []

    # Use curated samples as weak supervision for the text classifier.
    curated = load_curated_samples(sample_file)
    if curated:
        # Assign curated samples by keyword heuristics and label folders.
        for text in curated:
            t = text.lower()
            if any(k in t for k in ('furious', 'angry', 'frustrat', 'annoyed', 'fed up', 'irritated')):
                lbl = 'OAF_angry'
            elif any(k in t for k in ('gross', 'disgust', 'repuls', 'sick', 'nasty', 'unpleasant')):
                lbl = 'OAF_disgust'
            elif any(k in t for k in ('scared', 'afraid', 'worried', 'anxious', 'frightened', 'terrified')):
                lbl = 'OAF_Fear'
            elif any(k in t for k in ('happy', 'joy', 'excited', 'smile', 'proud', 'grateful', 'hopeful')):
                lbl = 'OAF_happy'
            elif any(k in t for k in ('surprise', 'surprised', 'wow', 'amazed', 'unexpected')):
                lbl = 'OAF_Pleasant_surprise'
            elif any(k in t for k in ('sad', 'lonely', 'heartbroken', 'upset', 'disappointed', 'heavy', 'miss')):
                lbl = 'OAF_Sad'
            else:
                lbl = 'OAF_neutral'
            rows.append((text, lbl))

    # Dataset transcripts are short lexical tokens. Add them as-is and augment
    # each label with a few label-consistent seed sentences.
    for _, r in df.iterrows():
        label = _normalize_label(r['label'])
        transcript = str(r.get('transcript', '')).strip()
        if transcript:
            rows.append((transcript, label))

    for label, seed_texts in EMOTION_SEED_TEXTS.items():
        for text in seed_texts:
            rows.append((text, label))

    # Augment a bit by shuffling phrase order and adding punctuation variants.
    augmented = []
    for text, label in rows:
        augmented.append((text, label))
        if len(text.split()) >= 3:
            augmented.append((text + '!', label))
            augmented.append((text + '.', label))
        if len(text.split()) >= 4:
            words = text.split()
            random.shuffle(words)
            augmented.append((' '.join(words), label))

    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='dataset/metadata.csv')
    parser.add_argument('--samples', default='dataset/text_samples/text_only_samples.txt')
    parser.add_argument('--outdir', default='models/text_pipeline')
    parser.add_argument('--max_features', type=int, default=15000)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=0.98)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    sample_file = Path(args.samples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_training_rows(metadata_path, sample_file, seed=args.random_state)
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xv = vectorizer.transform(X_val)

    label_encoder = LabelEncoder()
    ytr = label_encoder.fit_transform(y_train)
    yv = label_encoder.transform(y_val)

    model = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        n_jobs=None,
        multi_class='auto',
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xv)
    report = classification_report(yv, pred, target_names=label_encoder.classes_)
    print(report)

    joblib.dump(
        {'model': model, 'vectorizer': vectorizer, 'label_encoder': label_encoder},
        outdir / 'text_baseline.joblib',
    )
    (outdir / 'text_report.txt').write_text(report, encoding='utf-8')
    print('Saved', outdir / 'text_baseline.joblib')


if __name__ == '__main__':
    main()
