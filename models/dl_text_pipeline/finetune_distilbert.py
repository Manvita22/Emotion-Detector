import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertModel


def normalize_text(text):
    if not text:
        return ''
    return ' '.join(str(text).lower().strip().split())

def load_metadata(path):
    df = pd.read_csv(path)
    # expect 'transcript' and 'label' columns
    df['text'] = df['transcript'].fillna('').astype(str).map(normalize_text)
    df['label'] = df.get('label', df.get('emotion', df.get('class')))
    df = df[df['text'].str.strip() != '']
    return df


def main(metadata, outdir, epochs=3):
    df = load_metadata(metadata)
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['label'].astype(str))
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

    embeddings = []
    labels = []
    batch_size = 16
    texts = df['text'].tolist()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = encoder(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        labels.extend(df['y'].iloc[start:start + batch_size].tolist())

    X = np.vstack(embeddings)
    y = np.array(labels)
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    class TextHead(nn.Module):
        def __init__(self, in_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_len = int(len(dataset) * 0.9)
    val_len = max(len(dataset) - train_len, 1)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = TextHead(X.shape[1], len(le.classes_))
    counts = np.bincount(y, minlength=len(le.classes_)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    class_weight = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_state = None
    best_val = -1.0
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            train_total += yb.size(0)
            train_correct += (preds == yb).sum().item()
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_total += yb.size(0)
                val_correct += (preds == yb).sum().item()
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f'Epoch {epoch + 1}/{epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'text_mlp.pt')
    torch.save({'model_state': model.state_dict(), 'label_classes': le.classes_}, outpath)
    import joblib
    joblib.dump({'label_encoder': le}, os.path.join(outdir, 'label_encoder.joblib'))
    print('Saved model to', outpath)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--outdir', default='models/dl_text_pipeline')
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    main(args.metadata, args.outdir, epochs=args.epochs)
