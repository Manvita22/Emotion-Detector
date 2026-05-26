import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import librosa


def extract_mfcc_mean(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def main():
    meta = pd.read_csv('dataset/metadata.csv')
    X = []
    labels = []
    for _, row in meta.iterrows():
        path = row['audio_path']
        try:
            feat = extract_mfcc_mean(path)
        except Exception:
            continue
        X.append(feat)
        labels.append(row['label'])

    X = np.vstack(X)
    emb = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8,6))
    uniq = list(sorted(set(labels)))
    for u in uniq:
        idx = [i for i,l in enumerate(labels) if l==u]
        plt.scatter(emb[idx,0], emb[idx,1], label=u, s=8)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Results/embeddings_mfcc_tsne.png')
    print('Saved Results/embeddings_mfcc_tsne.png')


if __name__ == '__main__':
    main()
