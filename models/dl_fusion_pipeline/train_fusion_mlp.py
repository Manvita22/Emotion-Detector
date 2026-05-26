import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import DistilBertTokenizerFast, DistilBertModel
import librosa
import io

class FusionMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
    def forward(self,x):
        return self.net(x)


def get_text_emb(model, tokenizer, text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = model(**{k:v for k,v in inputs.items()})
    return out.last_hidden_state[:,0,:].squeeze().numpy()


def get_audio_emb(path, sr=16000, n_mfcc=40):
    data, _ = librosa.load(path, sr=sr)
    if data.ndim>1: data = data.mean(axis=0)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat


def main(metadata, outpath, epochs=10, batch_size=32):
    df = pd.read_csv(metadata)
    df = df.dropna(subset=['audio_path'])
    df['text'] = df['transcript'].fillna('').astype(str)
    df['label'] = df.get('label', df.get('emotion', df.get('class')))
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['label'].astype(str))
    # load text encoder
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # precompute embeddings
    text_embs = []
    audio_embs = []
    for _, row in df.iterrows():
        t = row['text']
        te = get_text_emb(text_model, tokenizer, t)
        ae = get_audio_emb(row['audio_path'])
        text_embs.append(te)
        audio_embs.append(ae)
    X_text = np.vstack(text_embs)
    X_audio = np.vstack(audio_embs)
    X = np.hstack([X_audio, X_text])
    y = df['y'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    n_classes = len(le.classes_)
    model = FusionMLP(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X_train,y_train)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        total=0; acc=0
        for xb,yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            acc += (preds==yb).sum().item()
        print(f'Epoch {ep+1}/{epochs} train_acc={acc/total:.4f}')
    with torch.no_grad():
        val_logits = model(X_val)
        val_pred = val_logits.argmax(dim=1)
        print('Validation accuracy:', round(float(accuracy_score(y_val.numpy(), val_pred.numpy())), 4))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'label_classes': le.classes_}, outpath)
    print('Saved', outpath)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--out', default='models/dl_fusion_pipeline/fusion_mlp.pt')
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    main(args.metadata, args.out, epochs=args.epochs)
