"""Test fusion baseline and print metrics."""
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
    p.add_argument('--model', default='models/fusion_pipeline/fusion_baseline.joblib')
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    m = joblib.load(args.model)
    clf = m['model']
    scaler = m['scaler']
    le = m['label_encoder']
    tf = m['tfidf']
    svd = m['svd']

    X_audio = []
    texts = []
    labels = []
    for _, r in df.iterrows():
        ap = Path(r['audio_path'])
        if not ap.exists():
            continue
        feat = extract_mfcc_stats(str(ap))
        X_audio.append(feat)
        texts.append(str(r['transcript']))
        labels.append(r['label'])

    X_audio = np.vstack(X_audio)
    X_text = tf.transform(texts)
    X_text_red = svd.transform(X_text)
    X = np.hstack([X_audio, X_text_red])
    Xs = scaler.transform(X)

    y_true = le.transform(labels)
    preds = clf.predict(Xs)
    print(classification_report(y_true, preds, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, preds))


if __name__ == '__main__':
    main()
import pandas as pd
import joblib
import numpy as np
import os
import librosa


def extract_mfcc_mean(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    if len(y) == 0:
        return np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def main():
    meta = pd.read_csv('dataset/metadata.csv')
    texts = []
    aud_feats = []
    labels = []
    for _, row in meta.iterrows():
        audio = row['audio_path']
        if not os.path.exists(audio):
            audio = os.path.join('.', audio)
        if not os.path.exists(audio):
            continue
        aud_feats.append(extract_mfcc_mean(audio))
        txt = row.get('transcript', '')
        if not isinstance(txt, str) or txt.strip() == '':
            txt = os.path.basename(audio)
        texts.append(txt)
        labels.append(row['label'])

    aud_feats = np.vstack(aud_feats)
    model = joblib.load('Results/fusion/fusion_model.joblib')
    vec = model['vec']
    clf = model['clf']

    X_text = vec.transform(texts).toarray()
    X = np.hstack([aud_feats, X_text])

    preds = clf.predict(X)
    from sklearn.metrics import classification_report
    report = classification_report(labels, preds)
    print(report)
    with open('Results/fusion/fusion_test_report.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
