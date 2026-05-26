"""Train a simple text baseline using TF-IDF and LogisticRegression."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--outdir', default='models/text_pipeline')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    texts = df['transcript'].fillna('').astype(str).tolist()
    labels = df['label'].tolist()

    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X, y)

    joblib.dump({'model': clf, 'vectorizer': vect, 'label_encoder': le}, outdir / 'text_baseline.joblib')
    print('Saved text baseline to', outdir / 'text_baseline.joblib')


if __name__ == '__main__':
    main()
import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main(args):
    meta = pd.read_csv(args.metadata)
    texts = []
    labels = []
    for _, row in meta.iterrows():
        txt = row.get('transcript', '')
        if not isinstance(txt, str) or txt.strip() == '':
            # fallback: use filename (may be limited)
            txt = os.path.basename(row['audio_path'])
        texts.append(txt)
        labels.append(row['label'])

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vec = TfidfVectorizer(max_features=2000)
    Xtr = vec.fit_transform(X_train)
    Xv = vec.transform(X_val)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xv)
    report = classification_report(y_val, preds)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump({'vec': vec, 'clf': clf}, os.path.join(args.out_dir, 'text_model.joblib'))
    with open(os.path.join(args.out_dir, 'text_report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='dataset/metadata.csv')
    parser.add_argument('--out-dir', default='Results/text')
    args = parser.parse_args()
    main(args)
