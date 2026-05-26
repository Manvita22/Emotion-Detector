
Emotion Detector is a speech-and-text emotion recognition project built around the TESS Toronto Emotional Speech Set. The application takes an audio sample, transcribes it, and predicts the emotion from speech, text, or a fused combination of both signals.

## Project

The project is organized as a Flask-based web app backed by multiple emotion-classification pipelines. It supports speech-only inference, text-only inference, and multimodal fusion inference from the same `/predict` endpoint.

The dataset directory contains the TESS audio corpus, and `dataset/metadata.csv` is generated from that corpus to provide labels and transcript metadata for training and evaluation.

## Setup

1. Create and activate the virtual environment.

```bash
.venv\Scripts\Activate.ps1
```

2. Install the project dependencies.

```bash
pip install -r requirements.txt
```

3. Start the application.

```bash
python app.py
```

4. Open the local app in the browser.

```text
http://127.0.0.1:8002
```

## Architecture

The system follows a three-branch architecture:

- Speech branch: extracts acoustic features from audio and predicts emotion from the voice signal.
- Text branch: uses the transcript produced by ASR and predicts emotion from the spoken content.
- Fusion branch: combines speech and text representations to produce a final emotion prediction.

The current implementation uses a layered inference strategy:

- Whisper-first ASR with wav2vec2 fallback for transcription.
- Baseline speech models built from MFCC features.
- Baseline text models built from transcript embeddings or text heuristics depending on the saved artifact.
- Fusion logic that blends audio and transcript evidence and falls back safely when the transcript is too short or low-confidence.

## Features

- Speech emotion prediction from uploaded audio.
- Text emotion prediction from free-form input or transcribed speech.
- Fusion prediction that uses both audio and transcript context.
- Automatic transcript normalization for prompt-like phrases such as “say the word ...”.
- Rewindable upload handling so the same audio can be reused across ASR, speech features, and fusion.
- Support for local sample collection through labeled uploads.
- CPU-friendly baseline pipelines alongside higher-level deep-learning checkpoints.
- Browser UI for trying speech, text, and fusion predictions from the same interface.
