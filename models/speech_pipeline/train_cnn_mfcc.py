import argparse
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split

class MFCCDataset(Dataset):
    def __init__(self, rows, sr=16000, n_mfcc=40, max_len=160):
        self.rows = rows
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        p = Path(row['audio_path'])
        data, sr = librosa.load(p, sr=self.sr)
        if data.ndim>1:
            data = data.mean(axis=0)
        mfcc = librosa.feature.mfcc(y=data, sr=self.sr, n_mfcc=self.n_mfcc)
        # pad or trim
        if mfcc.shape[1] < self.max_len:
            pad = np.zeros((self.n_mfcc, self.max_len - mfcc.shape[1]))
            mfcc = np.hstack([mfcc, pad])
        else:
            mfcc = mfcc[:, :self.max_len]
        x = torch.from_numpy(mfcc).float().unsqueeze(0)  # 1 x n_mfcc x time
        return x, int(row['y'])

class SimpleCNN(nn.Module):
    def __init__(self, n_mfcc=40, n_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self,x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


def main(metadata, outpath, epochs=5, max_samples=1200):
    df = pd.read_csv(metadata)
    df = df.dropna(subset=['audio_path'])
    if max_samples and len(df) > max_samples:
        df, _ = train_test_split(df, train_size=max_samples, random_state=42, stratify=df['label'] if 'label' in df.columns else None)
    lb = LabelEncoder()
    df['label'] = df.get('label', df.get('emotion', df.get('class')))
    df['y'] = lb.fit_transform(df['label'].astype(str))
    ds = MFCCDataset(df)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    n_classes = len(lb.classes_)
    model = SimpleCNN(n_mfcc=40, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total=0; acc=0
        for x,y in dl:
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            total += y.size(0)
            acc += (preds==y).sum().item()
        print(f'Epoch {ep+1}/{epochs} acc={acc/total:.4f}')
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'label_classes': lb.classes_}, outpath)
    print('Saved', outpath)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--out', default='models/dl_speech_pipeline/speech_cnn.pt')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=1200)
    args = p.parse_args()
    main(args.metadata, args.out, epochs=args.epochs, max_samples=args.max_samples)
