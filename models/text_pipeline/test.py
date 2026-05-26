"""Test the text baseline model and print classification metrics."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--model', default='models/text_pipeline/text_baseline.joblib')
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    m = joblib.load(args.model)
    clf = m['model']
    vect = m['vectorizer']
    le = m['label_encoder']

    texts = df['transcript'].fillna('').astype(str).tolist()
    X = vect.transform(texts)
    y_true = le.transform(df['label'].tolist())
    preds = clf.predict(X)
    print(classification_report(y_true, preds, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, preds))


if __name__ == '__main__':
    main()
import pandas as pd
import joblib


def main():
    meta = pd.read_csv('dataset/metadata.csv')
    texts = []
    labels = []
    for _, row in meta.iterrows():
        txt = row.get('transcript', '')
        if not isinstance(txt, str) or txt.strip() == '':
            txt = row['audio_path']
        texts.append(txt)
        labels.append(row['label'])

    obj = joblib.load('Results/text/text_model.joblib')
    vec = obj['vec']
    clf = obj['clf']

    X = vec.transform(texts)
    preds = clf.predict(X)

    from sklearn.metrics import classification_report
    report = classification_report(labels, preds)
    print(report)
    with open('Results/text/text_test_report.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
