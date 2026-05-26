"""Train a simple speech baseline using MFCC features and LogisticRegression.
Usage: python train.py --metadata dataset/metadata.csv --outdir models/speech_pipeline
"""
import argparse
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def extract_mfcc_stats(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    if y.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # statistics over time
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--outdir', default='models/speech_pipeline')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    X = []
    y = []
    for _, r in df.iterrows():
        ap = Path(r['audio_path'])
        if not ap.exists():
            continue
        feat = extract_mfcc_stats(str(ap))
        if feat is None:
            continue
        X.append(feat)
        y.append(r['label'])

    X = np.vstack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xs, y_enc)

    joblib.dump({'model': clf, 'scaler': scaler, 'label_encoder': le}, outdir / 'speech_baseline.joblib')
    print('Saved model to', outdir / 'speech_baseline.joblib')


if __name__ == '__main__':
    main()
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import librosa


def extract_mfcc(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    if len(y) == 0:
        return np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def main(args):
    meta = pd.read_csv(args.metadata)
    X = []
    y = []
    for _, row in meta.iterrows():
        audio = row['audio_path']
        label = row['label']
        if not os.path.isabs(audio):
            audio = os.path.join(args.base_path, audio)
        if not os.path.exists(audio):
            continue
        feats = extract_mfcc(audio)
        X.append(feats)
        y.append(label)

    X = np.vstack(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    report = classification_report(y_val, preds)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out_dir, 'speech_model.joblib'))
    with open(os.path.join(args.out_dir, 'speech_report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='dataset/metadata.csv')
    parser.add_argument('--base-path', default='.')
    parser.add_argument('--out-dir', default='Results/speech')
    args = parser.parse_args()
    main(args)
