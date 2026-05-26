import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys, os
# ensure project root is importable when running from this subdirectory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app import normalize_transcript, get_text_embedding


def main(metadata_path, outpath):
    meta = pd.read_csv(metadata_path)
    X = []
    y = []
    for _, row in meta.iterrows():
        txt = str(row.get('transcript','')).strip()
        txt = normalize_transcript(txt)
        if not txt:
            continue
        emb = get_text_embedding(txt)
        X.append(emb)
        y.append(row.get('label', row.get('emotion', None) or row.get('class', None)))
    X = np.vstack(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y_enc)
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
    p.add_argument('--out', default='models/text_pipeline/text_baseline.joblib')
    args = p.parse_args()
    main(args.metadata, args.out)
