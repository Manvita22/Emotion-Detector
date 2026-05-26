"""Train a small MLP classifier on pre-extracted audio embeddings (from scripts/extract_embeddings.py).
Usage: python models/dl_speech_pipeline/train.py --features features/audio_emb.npy --labels features/labels.npy
"""
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='features/audio_emb.npy')
    p.add_argument('--labels', default='features/labels.npy')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    X = np.load(args.features)
    labels = np.load(args.labels)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    te_ds = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long())
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch)

    device = torch.device(args.device)
    model = MLP(X.shape[1], len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={total_loss/len(tr_ds):.4f}')

    # save model and label encoder
    Path('models/dl_speech_pipeline').mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'label_classes': le.classes_.tolist()}, 'models/dl_speech_pipeline/speech_mlp.pt')
    print('Saved model to models/dl_speech_pipeline/speech_mlp.pt')


if __name__ == '__main__':
    main()
