"""Extract audio and text embeddings for the dataset using pretrained models.
Usage examples:
  python scripts/extract_embeddings.py --metadata dataset/metadata.csv --outdir features --max-samples 200
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForSpeechSeq2Seq
import torchaudio


def extract_audio_embeddings(paths, model_name='facebook/wav2vec2-base-960h', device='cpu'):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    embs = []
    for p in paths:
        speech, sr = torchaudio.load(p)
        # convert to mono
        speech = speech.mean(dim=0)
        # resample to target sampling rate expected by model
        target_sr = processor.feature_extractor.sampling_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            speech = resampler(speech)
        speech = speech.numpy()
        inputs = processor(speech, sampling_rate=target_sr, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inputs.items()})
        # mean pooling over time
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embs.append(emb)
    return np.vstack(embs)


def extract_text_embeddings(texts, model_name='distilbert-base-uncased', device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    embs = []
    for t in texts:
        inputs = tokenizer(t, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inputs.items()})
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embs.append(emb)
    return np.vstack(embs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='dataset/metadata.csv')
    p.add_argument('--outdir', default='features')
    p.add_argument('--max-samples', type=int, default=200)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.iloc[: args.max_samples]

    audio_paths = [str(Path(p)) for p in df['audio_path'].tolist()]
    texts = df['transcript'].fillna('').astype(str).tolist()
    labels = df['label'].tolist()

    print('Extracting audio embeddings for', len(audio_paths), 'samples')
    audio_emb = extract_audio_embeddings(audio_paths, device=args.device)
    print('Audio embeddings shape', audio_emb.shape)
    np.save(outdir / 'audio_emb.npy', audio_emb)

    print('Extracting text embeddings for', len(texts), 'samples')
    text_emb = extract_text_embeddings(texts, device=args.device)
    print('Text embeddings shape', text_emb.shape)
    np.save(outdir / 'text_emb.npy', text_emb)

    np.save(outdir / 'labels.npy', np.array(labels))
    print('Saved features to', outdir)


if __name__ == '__main__':
    main()
