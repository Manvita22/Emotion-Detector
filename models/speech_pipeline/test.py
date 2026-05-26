"""Test the speech baseline model and print classification metrics."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import librosa
from sklearn.metrics import classification_report, confusion_matrix


def extract_mfcc_stats(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--model', default='models/speech_pipeline/speech_baseline.joblib')
    args = p.parse_args()

    meta = pd.read_csv(args.metadata)
    m = joblib.load(args.model)
    clf = m['model']
    scaler = m['scaler']
    le = m['label_encoder']

    X = []
    y = []
    for _, r in meta.iterrows():
        ap = Path(r['audio_path'])
        if not ap.exists():
            continue
        feat = extract_mfcc_stats(str(ap))
        X.append(feat)
        y.append(r['label'])

    X = np.vstack(X)
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)
    y_true = le.transform(y)
    print(classification_report(y_true, preds, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, preds))


if __name__ == '__main__':
    main()
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import librosa


def extract_mfcc(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    if len(y) == 0:
        return np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def main():
    meta = pd.read_csv('dataset/metadata.csv')
    X = []
    y = []
    for _, row in meta.iterrows():
        audio = row['audio_path']
        label = row['label']
        if not os.path.exists(audio):
            audio = os.path.join('.', audio)
        if not os.path.exists(audio):
            continue
        feats = extract_mfcc(audio)
        X.append(feats)
        y.append(label)

    X = np.vstack(X)
    y = np.array(y)

    clf = joblib.load('Results/speech/speech_model.joblib')
    preds = clf.predict(X)
    report = classification_report(y, preds)
    print(report)
    with open('Results/speech/speech_test_report.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
