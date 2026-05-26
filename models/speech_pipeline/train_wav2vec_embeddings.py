import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys, os
# ensure project root is importable when running from this subdirectory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app import get_audio_embedding_from_fileobj, normalize_transcript
import io


def main(metadata_path, outpath):
    meta = pd.read_csv(metadata_path)
    X = []
    y = []
    for _, row in meta.iterrows():
        p = Path(row['audio_path'])
        if not p.exists():
            continue
        with p.open('rb') as f:
            b = io.BytesIO(f.read())
            b.filename = p.name
            emb = get_audio_embedding_from_fileobj(b)
        X.append(emb)
        y.append(row.get('label', row.get('emotion', None) or row.get('class', None)))
    X = np.vstack(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.1, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=3000, class_weight='balanced')
    clf.fit(Xs_train, y_train)
    val_pred = clf.predict(Xs_val)
    print('Validation accuracy:', round(float(accuracy_score(y_val, val_pred)), 4))
    out = {
        'scaler': scaler,
        'model': clf,
        'label_encoder': le,
    }
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, outpath)
    print('Saved', outpath)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--out', default='models/speech_pipeline/speech_baseline.joblib')
    args = p.parse_args()
    main(args.metadata, args.out)
