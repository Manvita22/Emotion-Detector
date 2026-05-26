from flask import Flask, request, jsonify, render_template, Response
from flask import send_file, abort
import os
# Force CPU-only execution: hide CUDA devices before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import joblib
from pathlib import Path
import librosa
import numpy as np
import io
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel, AutoModelForCTC, AutoTokenizer
import tempfile
import subprocess
import shutil

app = Flask(__name__)

SAMPLE_OUTPUT_DIR = Path('dataset/text_samples')
SAMPLE_OUTPUT_DIR_LEGACY = Path('samples')
SAMPLE_FILE_NAMES = ('text_only_samples.txt', 'fusion_samples.txt')

# Lazy-loaded models
models = {}
dl_models = {}
nlp_cache = {}

TEXT_EMOTION_LABELS = {
    'angry': ['YAF_angry', 'OAF_angry'],
    'disgust': ['YAF_disgust', 'OAF_disgust'],
    'fear': ['YAF_fear', 'OAF_Fear'],
    'happy': ['YAF_happy', 'OAF_happy'],
    'neutral': ['YAF_neutral', 'OAF_neutral'],
    'pleasant_surprise': ['YAF_pleasant_surprised', 'OAF_Pleasant_surprise'],
    'sad': ['YAF_sad', 'OAF_Sad'],
}

TEXT_EMOTION_KEYWORDS = {
    'angry': ['angry', 'furious', 'mad', 'frustrated', 'annoyed', 'fed up', 'unfair', 'ignored', 'blame', 'irritated', 'snapped', 'ridiculous'],
    'disgust': ['disgust', 'gross', 'nasty', 'repulsed', 'sick', 'filthy', 'ashamed', 'creepy', 'turned my stomach', 'unpleasant'],
    'fear': ['scared', 'afraid', 'worried', 'anxious', 'frightened', 'terrified', 'unsafe', 'panic', 'nervous', 'uncertain', 'what if', 'dread'],
    'happy': ['happy', 'joy', 'delighted', 'excited', 'smile', 'cheerful', 'relieved', 'proud', 'grateful', 'hopeful', 'light', 'good news'],
    'neutral': ['calm', 'okay', 'ordinary', 'neutral', 'normal', 'usual', 'regular', 'nothing special', 'steady'],
    'pleasant_surprise': ['surprise', 'surprised', 'wow', 'amazed', 'unexpected', 'delightful', 'did not expect', 'suddenly', 'out of nowhere', 'wonderful news'],
    'sad': ['sad', 'lonely', 'miserable', 'down', 'upset', 'heartbroken', 'tired', 'empty', 'miss', 'disappointed', 'heavy', 'hurt', 'quiet', 'bad news'],
}

EMOTION_ALIASES = {
    'pleasant': 'pleasant_surprise',
    'pleasant_surprised': 'pleasant_surprise',
    'pleasant_surprise': 'pleasant_surprise',
    'surprise': 'pleasant_surprise',
    'ps': 'pleasant_surprise',
    'fear': 'fear',
    'sad': 'sad',
    'happy': 'happy',
    'angry': 'angry',
    'disgust': 'disgust',
    'neutral': 'neutral',
}

BUILTIN_TEXT_SAMPLES = [
    'I kept saying I was fine, but the whole evening felt heavy and quiet, like I was carrying bad news I could not explain.',
    'Nothing dramatic happened today, yet every small delay made me feel ignored, tense, and ready to snap at someone.',
    'The message was short and vague, but now I keep replaying it and worrying that something has gone wrong.',
    'I expected another ordinary update, and then the result came in better than I hoped, which genuinely made me smile.',
    'The room smelled stale and the mess on the table made me want to step back immediately.',
    'I am not excited or upset about it; it feels like a regular task that simply needs to be finished.',
    'I thought the meeting would be routine, but the sudden appreciation caught me off guard in the best way.',
    'They laughed after I explained the problem, and I felt hurt, embarrassed, and smaller than I expected.',
    'The plan changed again without telling me, and that unfairness is starting to make me really frustrated.',
    'I heard a noise outside and even though it may be nothing, my chest tightened and I could not relax.',
    'After weeks of trying, seeing it finally work made the whole day feel lighter and hopeful.',
    'The food looked fine at first, but the taste was so unpleasant that I felt sick almost immediately.',
    'I walked home quietly because I missed how things used to be and did not know who to tell.',
    'There is no strong feeling here; I received the update, understood it, and moved on with my day.',
]




def load_speech_model():
    path = Path('models/speech_pipeline/speech_baseline.joblib')
    if not path.exists():
        return None
    return joblib.load(path)


def load_text_model():
    candidates = [
        Path('models/text_pipeline/text_baseline.joblib'),
        Path('models/text_pipeline/text_model.joblib'),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    data = joblib.load(path)
    if isinstance(data, dict):
        if 'vectorizer' not in data and 'vec' in data:
            data['vectorizer'] = data['vec']
        if 'model' not in data and 'clf' in data:
            data['model'] = data['clf']
        if 'label_encoder' not in data and 'le' in data:
            data['label_encoder'] = data['le']
        if 'vectorizer' in data:
            data['feature_kind'] = 'tfidf'
        elif 'scaler' in data:
            data['feature_kind'] = 'embedding'
    return data


def transform_text_features(text, mdl):
    if not text:
        raise ValueError('No text provided')
    if isinstance(mdl, dict) and 'vectorizer' in mdl:
        return mdl['vectorizer'].transform([text])
    if isinstance(mdl, dict) and 'scaler' in mdl:
        emb = get_text_embedding(text)
        return mdl['scaler'].transform([emb])
    raise KeyError('vectorizer')


def load_fusion_model():
    path = Path('models/fusion_pipeline/fusion_baseline.joblib')
    if not path.exists():
        return None
    return joblib.load(path)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def is_flat_text_model(mdl):
    model = mdl.get('model') if isinstance(mdl, dict) else None
    coef = getattr(model, 'coef_', None)
    intercept = getattr(model, 'intercept_', None)
    if coef is None:
        return False
    coef_flat = np.allclose(coef, 0)
    intercept_flat = intercept is None or np.allclose(intercept, 0)
    return bool(coef_flat and intercept_flat)


def keyword_matches(normalized_text, keyword):
    if ' ' in keyword:
        return keyword in normalized_text
    return re.search(rf'\b{re.escape(keyword)}\b', normalized_text) is not None


def heuristic_text_prediction(text, classes, fallback_probs=None):
    normalized = f' {text.lower()} '
    scores = {label: 0.0 for label in classes}
    matched = False

    for emotion, labels in TEXT_EMOTION_LABELS.items():
        hits = 0
        for keyword in TEXT_EMOTION_KEYWORDS[emotion]:
            if keyword_matches(normalized, keyword):
                hits += 1
        if hits:
            matched = True
            primary, secondary = labels
            scores[primary] += 3.0 * hits
            if secondary in scores:
                scores[secondary] += 1.0 * hits

    if not matched:
        if fallback_probs is None:
            fallback_probs = np.full(len(classes), 1.0 / max(len(classes), 1))
        top_idx = np.argsort(fallback_probs)[::-1][:3]
        return [{'label': str(classes[i]), 'prob': float(fallback_probs[i])} for i in top_idx]

    total = sum(scores.values())
    if total <= 0:
        top_idx = np.argsort(fallback_probs)[::-1][:3] if fallback_probs is not None else range(min(3, len(classes)))
        probs = fallback_probs if fallback_probs is not None else np.full(len(classes), 1.0 / max(len(classes), 1))
        return [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top = [{'label': label, 'prob': float(score / total)} for label, score in ranked[:3]]
    return top


def emotion_from_label(label):
    normalized = str(label).lower().replace('-', '_')
    for key, emotion in EMOTION_ALIASES.items():
        if key in normalized:
            return emotion
    return normalized.rsplit('_', 1)[-1]


def text_emotion_scores(text):
    normalized = f' {text.lower()} '
    scores = {emotion: 0.0 for emotion in TEXT_EMOTION_KEYWORDS}
    for emotion, keywords in TEXT_EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword_matches(normalized, keyword):
                scores[emotion] += 1.0
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {emotion: score / total for emotion, score in scores.items() if score > 0}


def ranked_probs(classes, probs, mode='fusion', note=None, model_variant=None):
    top_idx = np.argsort(probs)[::-1][:3]
    top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
    result = {'mode': mode, 'top': top, 'pred': top[0]}
    if note:
        result['note'] = note
    if model_variant:
        result['model_variant'] = model_variant
    return result


def blend_fusion_probs(classes, base_probs, text):
    text_scores = text_emotion_scores(text)
    if not text_scores:
        return base_probs

    text_label_probs = np.zeros(len(classes), dtype=float)
    for emotion, score in text_scores.items():
        matching = [idx for idx, label in enumerate(classes) if emotion_from_label(label) == emotion]
        if not matching:
            continue
        share = score / len(matching)
        for idx in matching:
            text_label_probs[idx] += share

    if text_label_probs.sum() <= 0:
        return base_probs

    text_label_probs = text_label_probs / text_label_probs.sum()
    # Adjust weights based on transcript length: favor text more when transcript is longer
    n_words = len(str(text).split()) if text else 0
    if n_words >= 5:
        text_w = 0.45
    elif n_words >= 3:
        text_w = 0.40
    else:
        # For very short transcripts (likely single-word prompts), favor audio more
        text_w = 0.20
    audio_w = 1.0 - text_w
    blended = (audio_w * base_probs) + (text_w * text_label_probs)
    blended = blended / blended.sum()
    return blended


def map_and_blend_with_baseline(target_classes, dl_probs, baseline_mdl, feat_X=None, text_X=None):
    """Map baseline model probabilities to target_classes ordering and blend with dl_probs.
    baseline_mdl: dict with keys 'model' and 'label_encoder' (scikit pipeline)
    feat_X / text_X: features already computed for baseline if needed (either MFCC or TF-IDF array)
    """
    try:
        base_classes = baseline_mdl['label_encoder'].classes_
        # If feat_X provided, predict
        if feat_X is not None:
            probs_base = baseline_mdl['model'].predict_proba(feat_X)[0]
        else:
            # Not enough info
            return dl_probs
        mapped = np.zeros(len(target_classes), dtype=float)
        for i, bc in enumerate(base_classes):
            # find index in target_classes
            matches = [j for j, t in enumerate(target_classes) if str(t) == str(bc)]
            if not matches:
                continue
            share = probs_base[i] / len(matches)
            for j in matches:
                mapped[j] += share
        if mapped.sum() <= 0:
            return dl_probs
        mapped = mapped / mapped.sum()
        blended = 0.6 * dl_probs + 0.4 * mapped
        blended = blended / blended.sum()
        return blended
    except Exception:
        return dl_probs


def build_fusion_text(audio, typed_text=''):
    transcript = ''
    asr_error = None
    try:
        transcript = transcribe_audio_fileobj(audio)
    except Exception as exc:
        asr_error = str(exc)

    # If ASR failed, save the raw uploaded audio for debugging
    if asr_error or not transcript:
        try:
            save_debug_audio(audio, prefix='fusion_asr_fail')
        except Exception:
            pass

    typed_text = typed_text.strip()
    # ensure transcript is normalized
    try:
        transcript = normalize_transcript(transcript)
    except Exception:
        pass
    fusion_text = ' '.join(part for part in (transcript, typed_text) if part)
    return transcript, fusion_text, asr_error


def save_debug_audio(fileobj, out_dir='debug_failures', prefix='audio'):
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rewind_fileobj(fileobj)
        content = fileobj.read()
        if not content:
            return None
        ts = int(__import__('time').time())
        name = f"{prefix}_{ts}.bin"
        p = Path(out_dir) / name
        with p.open('wb') as f:
            f.write(content)
        rewind_fileobj(fileobj)
        return str(p)
    except Exception:
        return None


def attach_fusion_metadata(result, transcript='', asr_error=None):
    result['transcript'] = transcript
    if asr_error:
        result['asr_error'] = asr_error
        suffix = ' ASR transcription failed, so Fusion used the available typed text branch only.'
        result['note'] = f"{result.get('note', '').rstrip()} {suffix}".strip()
    return result


def predict_fusion_ensemble(audio, text, transcript='', asr_error=None):
    mdl = models.get('speech') or load_speech_model()
    if mdl is None:
        return None
    models['speech'] = mdl
    rewind_fileobj(audio)
    feat = extract_mfcc_stats_fileobj(audio)
    Xs = mdl['scaler'].transform([feat])
    audio_probs = mdl['model'].predict_proba(Xs)[0]
    classes = mdl['label_encoder'].classes_
    probs = blend_fusion_probs(classes, audio_probs, text)
    result = ranked_probs(
        classes,
        probs,
        note='Fusion transcribed the uploaded speech, processed speech and transcript branches, then fused both signals for classification.',
        model_variant='ensemble',
    )
    return attach_fusion_metadata(result, transcript, asr_error)


def load_dl_speech():
    candidates = [
        Path('models/dl_speech_pipeline/speech_cnn.pt'),
        Path('models/dl_speech_pipeline/speech_mlp.pt'),
    ]
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:
        return None
    data = torch.load(str(p), map_location='cpu')
    classes = data['label_classes']
    state_keys = set(data.get('model_state', {}).keys())
    if any(key.startswith('conv.') for key in state_keys):
        class SimpleCNN(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Linear(64, n_classes)

            def forward(self, x):
                h = self.conv(x)
                h = h.view(h.size(0), -1)
                return self.fc(h)

        m = SimpleCNN(len(classes))
        m.load_state_dict(data['model_state'])
        m.to('cpu')
        m.eval()
        return {'model': m, 'classes': classes, 'kind': 'cnn'}

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

    m = MLP(768, len(classes))
    m.load_state_dict(data['model_state'])
    m.to('cpu')
    m.eval()
    return {'model': m, 'classes': classes, 'kind': 'mlp'}


def load_dl_text():
    candidates = [
        Path('models/dl_text_pipeline/text_mlp.pt'),
        Path('models/dl_text_pipeline/text_classifier/model.pt'),
    ]
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:
        return None
    data = torch.load(str(p), map_location='cpu', weights_only=False)
    classes = data['label_classes']
    in_dim = 768
    n_classes = len(classes)
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
    m = MLP(in_dim, n_classes)
    m.load_state_dict(data['model_state'])
    # ensure model on CPU
    m.to('cpu')
    m.eval()
    return {'model': m, 'classes': classes}


def load_dl_fusion():
    candidates = [
        Path('models/dl_fusion_pipeline/fusion_mlp.pt'),
        Path('models/dl_fusion_pipeline/fusion_classifier.pt'),
    ]
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:
        return None
    data = torch.load(str(p), map_location='cpu')
    classes = data['label_classes']
    # fusion input = audio 768 + text 768
    in_dim = 768 + 768
    n_classes = len(classes)
    class MLP(nn.Module):
        def __init__(self, in_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )
        def forward(self, x):
            return self.net(x)
    m = MLP(in_dim, n_classes)
    m.load_state_dict(data['model_state'])
    # ensure model on CPU
    m.to('cpu')
    m.eval()
    return {'model': m, 'classes': classes}


def get_audio_embedding_from_fileobj(fileobj):
    rewind_fileobj(fileobj)
    # lazy load processor and model
    if 'audio_processor' not in nlp_cache:
        proc = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')
        mdl = AutoModel.from_pretrained('facebook/wav2vec2-base-960h')
        # ensure model on CPU
        try:
            mdl.to('cpu')
        except Exception:
            pass
        nlp_cache['audio_processor'] = proc
        nlp_cache['audio_model'] = mdl
    proc = nlp_cache['audio_processor']
    mdl = nlp_cache['audio_model']
    try:
        wav_buf = convert_uploaded_audio_to_wav_bytes(fileobj)
        wav_buf.seek(0)
        data, sr = librosa.load(wav_buf, sr=None)
    except RuntimeError:
        rewind_fileobj(fileobj)
        data, sr = librosa.load(io.BytesIO(fileobj.read()), sr=None)
    # mono
    if data.ndim > 1:
        data = data.mean(axis=0)
    target_sr = proc.feature_extractor.sampling_rate
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    inputs = proc(data, sampling_rate=target_sr, return_tensors='pt', padding=True)
    with torch.no_grad():
        out = mdl(**{k: v for k, v in inputs.items()})
    emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb


def rewind_fileobj(fileobj):
    try:
        fileobj.stream.seek(0)
    except Exception:
        try:
            fileobj.seek(0)
        except Exception:
            pass


def _guess_suffix_from_filename(fname: str):
    if not fname:
        return ''
    name = fname.lower()
    for ext in ('.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac'):
        if name.endswith(ext):
            return ext
    return ''


def convert_uploaded_audio_to_wav_bytes(fileobj, target_sr=16000):
    """Convert uploaded file-like (Flask FileStorage or bytes) to 16k mono WAV bytes using ffmpeg.
    Returns a BytesIO containing WAV data sampled at target_sr.
    Raises RuntimeError if conversion is not possible (ffmpeg missing).
    """
    # read raw bytes
    try:
        # Flask FileStorage
        raw = fileobj.read()
    except Exception:
        try:
            rewind_fileobj(fileobj)
            raw = fileobj.stream.read()
        except Exception:
            raise RuntimeError('Could not read uploaded audio bytes')

    # If the uploaded bytes already look like a WAV header, skip conversion
    if raw[:4] == b'RIFF' and b'WAVE' in raw[:12]:
        return io.BytesIO(raw)

    # Ensure ffmpeg is available
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError('ffmpeg not found on PATH. Please install ffmpeg to enable browser audio conversion (https://ffmpeg.org/download.html).')

    # create temp files
    suffix = _guess_suffix_from_filename(getattr(fileobj, 'filename', '') or '') or '.in'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_f:
        in_f.write(raw)
        in_path = in_f.name
    out_fd, out_path = tempfile.mkstemp(suffix='.wav')
    os.close(out_fd)

    try:
        cmd = [ffmpeg_path, '-y', '-i', in_path, '-ac', '1', '-ar', str(int(target_sr)), '-vn', out_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f'ffmpeg conversion failed: {stderr}')
        with open(out_path, 'rb') as f:
            wav_bytes = f.read()
        return io.BytesIO(wav_bytes)
    finally:
        try:
            os.unlink(in_path)
        except Exception:
            pass
        try:
            os.unlink(out_path)
        except Exception:
            pass


def load_audio_array_from_fileobj(fileobj, sr=None):
    rewind_fileobj(fileobj)
    try:
        wav_buf = convert_uploaded_audio_to_wav_bytes(fileobj, target_sr=sr or 16000)
        wav_buf.seek(0)
        data, file_sr = librosa.load(wav_buf, sr=sr)
    except RuntimeError:
        # fallback: try raw bytes
        rewind_fileobj(fileobj)
        data, file_sr = librosa.load(io.BytesIO(fileobj.read()), sr=sr)
    if data.ndim > 1:
        data = data.mean(axis=0)
    rewind_fileobj(fileobj)
    return data, file_sr


def transcribe_audio_fileobj(fileobj):
    # Try Whisper ASR first for better robustness; fall back to wav2vec2 CTC if unavailable
    try:
        if 'whisper_processor' not in nlp_cache:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            wproc = WhisperProcessor.from_pretrained('openai/whisper-small')
            wmdl = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
            try:
                wmdl.to('cpu')
            except Exception:
                pass
            wmdl.config.forced_decoder_ids = None
            wmdl.eval()
            nlp_cache['whisper_processor'] = wproc
            nlp_cache['whisper_model'] = wmdl

        wproc = nlp_cache['whisper_processor']
        wmdl = nlp_cache['whisper_model']
        target_sr = 16000
        try:
            wav_buf = convert_uploaded_audio_to_wav_bytes(fileobj, target_sr=target_sr)
            wav_buf.seek(0)
            data, _ = librosa.load(wav_buf, sr=target_sr)
        except RuntimeError:
            data, _ = load_audio_array_from_fileobj(fileobj, sr=target_sr)

        inputs = wproc(data, sampling_rate=target_sr, return_tensors='pt')
        input_features = inputs.input_features
        with torch.no_grad():
            generated_ids = wmdl.generate(input_features)
        transcript = wproc.batch_decode(generated_ids, skip_special_tokens=True)[0]
        norm = normalize_transcript(transcript)
        try:
            print(f"ASR (whisper) raw: [{transcript}] normalized: [{norm}]")
        except Exception:
            pass
        return norm
    except Exception:
        # fallback: use wav2vec2 CTC
        if 'asr_processor' not in nlp_cache:
            proc = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')
            mdl = AutoModelForCTC.from_pretrained('facebook/wav2vec2-base-960h')
            try:
                mdl.to('cpu')
            except Exception:
                pass
            mdl.eval()
            nlp_cache['asr_processor'] = proc
            nlp_cache['asr_model'] = mdl

        proc = nlp_cache['asr_processor']
        mdl = nlp_cache['asr_model']
        target_sr = proc.feature_extractor.sampling_rate
        data, _ = load_audio_array_from_fileobj(fileobj, sr=target_sr)
        inputs = proc(data, sampling_rate=target_sr, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = mdl(**{k: v for k, v in inputs.items()}).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = proc.batch_decode(predicted_ids)[0]
        norm = normalize_transcript(transcript)
        try:
            print(f"ASR (wav2vec2) raw: [{transcript}] normalized: [{norm}]")
        except Exception:
            pass
        return norm


def normalize_transcript(text):
    """Normalize ASR transcripts and metadata for consistent comparison.
    Removes common prompt phrases (e.g. "say the word"), punctuation, and
    lowercases the text.
    """
    if not text:
        return ''
    try:
        import re
        t = text.lower().strip()
        # remove common spoken prompts like 'say the word', 'please say the word', etc.
        # Be permissive to punctuation and optional 'please', 'say', 'the', 'word' tokens.
        t = re.sub(r"^(?:please\W*)?(?:say|speak)?\W*(?:the)?\W*(?:word)?\W*", "", t)
        # also remove occurrences of 'say the word' appearing mid-sentence
        t = re.sub(r"\bsay\W+the\W+word\b", "", t)
        # strip punctuation
        t = re.sub(r"[^\w\s']", "", t)
        t = t.strip()
        return t
    except Exception:
        return text.lower().strip()


def get_text_embedding(text):
    if 'text_tokenizer' not in nlp_cache:
        tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        mdl = AutoModel.from_pretrained('distilbert-base-uncased')
        # ensure model on CPU
        try:
            mdl.to('cpu')
        except Exception:
            pass
        nlp_cache['text_tokenizer'] = tok
        nlp_cache['text_model'] = mdl
    tok = nlp_cache['text_tokenizer']
    mdl = nlp_cache['text_model']
    inputs = tok(text, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = mdl(**{k: v for k, v in inputs.items()})
    emb = out.last_hidden_state[:, 0, :].squeeze().numpy()
    return emb


def extract_mfcc_stats_fileobj(fileobj, sr=16000, n_mfcc=13):
    rewind_fileobj(fileobj)
    try:
        wav_buf = convert_uploaded_audio_to_wav_bytes(fileobj, target_sr=sr)
        wav_buf.seek(0)
        data, file_sr = librosa.load(wav_buf, sr=sr)
    except RuntimeError:
        rewind_fileobj(fileobj)
        data, file_sr = librosa.load(io.BytesIO(fileobj.read()), sr=None)
        if file_sr != sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    rewind_fileobj(fileobj)
    return feat


def extract_mfcc_tensor_fileobj(fileobj, sr=16000, n_mfcc=40, max_len=160):
    rewind_fileobj(fileobj)
    try:
        wav_buf = convert_uploaded_audio_to_wav_bytes(fileobj, target_sr=sr)
        wav_buf.seek(0)
        data, _ = librosa.load(wav_buf, sr=sr)
    except RuntimeError:
        rewind_fileobj(fileobj)
        data, _ = librosa.load(io.BytesIO(fileobj.read()), sr=sr)
    if data.ndim > 1:
        data = data.mean(axis=0)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad = np.zeros((n_mfcc, max_len - mfcc.shape[1]), dtype=mfcc.dtype)
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :max_len]
    rewind_fileobj(fileobj)
    return torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)


def predict_dl_fusion(audio, text, transcript='', asr_error=None):
    dl = dl_models.get('fusion') or load_dl_fusion()
    if dl is None:
        return None
    dl_models['fusion'] = dl
    rewind_fileobj(audio)
    emb_a = get_audio_embedding_from_fileobj(audio)
    emb_t = get_text_embedding(text)
    x = torch.from_numpy(np.hstack([emb_a, emb_t])).float().unsqueeze(0)
    with torch.no_grad():
        logits = dl['model'](x).squeeze().numpy()
    probs = blend_fusion_probs(dl['classes'], softmax(logits), text)
    # If DL fusion is not confident, try blending with the fusion ensemble baseline
    classes = dl['classes']
    if float(np.max(probs)) < 0.55:
        ensemble = predict_fusion_ensemble(audio, text, transcript, asr_error)
        if ensemble is not None:
            try:
                # ensemble top probabilities -> align classes
                base_probs = np.zeros(len(classes), dtype=float)
                for item in ensemble.get('top', []):
                    # find class index
                    matches = [i for i, c in enumerate(classes) if str(c) == str(item['label'])]
                    if matches:
                        base_probs[matches[0]] = item['prob']
                if base_probs.sum() > 0:
                    probs = 0.6 * probs + 0.4 * base_probs
                    probs = probs / probs.sum()
            except Exception:
                pass

    top_idx = probs.argsort()[::-1][:3]
    top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
    result = {'mode': 'fusion', 'model_variant': 'dl', 'top': top, 'pred': top[0], 'note': 'Fusion transcribed the uploaded speech and used audio plus transcript embeddings.'}
    return attach_fusion_metadata(result, transcript, asr_error)


def read_sample_lines():
    lines = []
    for base_dir in (SAMPLE_OUTPUT_DIR, SAMPLE_OUTPUT_DIR_LEGACY):
        sample_file = base_dir / 'text_only_samples.txt'
        if sample_file.exists():
            with sample_file.open('r', encoding='utf-8') as f:
                lines.extend(line.strip() for line in f if len(line.strip()) >= 60)
    merged = []
    for line in BUILTIN_TEXT_SAMPLES + lines:
        if line not in merged:
            merged.append(line)
    return merged


def ensure_sample_files(dataset_name='go_emotions', n=30):
    from scripts.fetch_text_samples import write_samples
    return write_samples(dataset_name, n, out_dir=str(SAMPLE_OUTPUT_DIR))


@app.route('/')
def index():
    # Embed sample lines directly into the rendered page to avoid client fetches
    text_samples = read_sample_lines()
    return render_template('index.html', sample_lines=text_samples)


@app.route('/models', methods=['GET'])
def models_info():
    info = {
        'speech_baseline': Path('models/speech_pipeline/speech_baseline.joblib').exists(),
        'text_baseline': Path('models/text_pipeline/text_baseline.joblib').exists(),
        'fusion_baseline': Path('models/fusion_pipeline/fusion_baseline.joblib').exists(),
        'dl_speech': Path('models/dl_speech_pipeline/speech_mlp.pt').exists() or Path('models/dl_speech_pipeline/speech_cnn.pt').exists(),
        'dl_text': Path('models/dl_text_pipeline/text_mlp.pt').exists() or Path('models/dl_text_pipeline/text_classifier/model.pt').exists(),
        'dl_fusion': Path('models/dl_fusion_pipeline/fusion_mlp.pt').exists() or Path('models/dl_fusion_pipeline/fusion_classifier.pt').exists(),
    }
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    mode = request.form.get('mode', 'speech')
    model_variant = request.form.get('model_variant', 'baseline')
    text = request.form.get('text', '')
    audio = request.files.get('audio')

    # If audio is a Flask FileStorage, read into a rewindable BytesIO so
    # multiple conversions/reads are possible (ASR + MFCC + DL paths).
    if audio is not None and hasattr(audio, 'read'):
        try:
            raw = audio.read()
            # wrap into BytesIO, preserve filename
            buf = io.BytesIO(raw)
            buf.filename = getattr(audio, 'filename', '')
            buf.content_type = getattr(audio, 'content_type', '')
            audio = buf
        except Exception:
            # leave original
            pass

    # Speech
    if mode == 'speech':
        if model_variant == 'baseline':
            mdl = models.get('speech') or load_speech_model()
            if mdl is None:
                return jsonify({'error': 'Speech model not found. Run speech baseline training.'}), 400
            if audio is None:
                return jsonify({'error': 'No audio file provided'}), 400
            feat = extract_mfcc_stats_fileobj(audio)
            Xs = mdl['scaler'].transform([feat])
            probs = mdl['model'].predict_proba(Xs)[0]
            top_idx = probs.argsort()[::-1][:3]
            classes = mdl['label_encoder'].classes_
            top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
            return jsonify({'mode': 'speech', 'top': top, 'pred': top[0]})
        elif model_variant == 'dl':
            dl = dl_models.get('speech') or load_dl_speech()
            if dl is None:
                return jsonify({'error': 'DL speech model not found. Train DL speech model.'}), 400
            if audio is None:
                return jsonify({'error': 'No audio file provided'}), 400
            classes = dl['classes']
            if dl.get('kind') == 'cnn':
                x = extract_mfcc_tensor_fileobj(audio)
                with torch.no_grad():
                    logits = dl['model'](x).squeeze().numpy()
                probs = softmax(logits)
            else:
                emb = get_audio_embedding_from_fileobj(audio)
                x = torch.from_numpy(emb).float().unsqueeze(0)
                with torch.no_grad():
                    logits = dl['model'](x).squeeze().numpy()
                probs = softmax(logits)

            # If DL is not confident, try blending with baseline speech model
            if float(np.max(probs)) < 0.55:
                base_mdl = models.get('speech') or load_speech_model()
                if base_mdl is not None:
                    try:
                        feat = extract_mfcc_stats_fileobj(audio)
                        Xs = base_mdl['scaler'].transform([feat])
                        probs = map_and_blend_with_baseline(classes, probs, base_mdl, feat_X=Xs)
                    except Exception:
                        pass
            top_idx = probs.argsort()[::-1][:3]
            top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
            return jsonify({'mode': 'speech', 'top': top, 'pred': top[0]})

    # Text
    if mode == 'text':
        if model_variant == 'baseline':
            mdl = models.get('text') or load_text_model()
            if mdl is None:
                return jsonify({'error': 'Text model not found. Run text baseline training.'}), 400
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            X = transform_text_features(text, mdl)
            probs = mdl['model'].predict_proba(X)[0]
            top_idx = probs.argsort()[::-1][:3]
            classes = mdl['label_encoder'].classes_
            top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
            if is_flat_text_model(mdl) or (float(np.max(probs)) - float(np.min(probs)) < 1e-6):
                top = heuristic_text_prediction(text, classes, fallback_probs=probs)
            return jsonify({'mode': 'text', 'top': top, 'pred': top[0]})
        elif model_variant == 'dl':
            dl = dl_models.get('text') or load_dl_text()
            if dl is None:
                return jsonify({'error': 'DL text model not found. Train DL text model.'}), 400
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            emb = get_text_embedding(text)
            x = torch.from_numpy(emb).float().unsqueeze(0)
            with torch.no_grad():
                logits = dl['model'](x).squeeze().numpy()
            probs = softmax(logits)
            classes = dl['classes']
            # The DL text checkpoint is weak on free-form sentences. When its
            # distribution is close to uniform, fall back to the stronger text
            # baseline or keyword heuristic instead of returning a repeated label.
            spread = float(np.max(probs) - np.min(probs))
            if float(np.max(probs)) < 0.20 or spread < 1e-3:
                base_mdl = models.get('text') or load_text_model()
                if base_mdl is not None:
                    try:
                        X = transform_text_features(text, base_mdl)
                        base_probs = base_mdl['model'].predict_proba(X)[0]
                        base_classes = base_mdl['label_encoder'].classes_
                        base_top = [{'label': str(base_classes[i]), 'prob': float(base_probs[i])} for i in base_probs.argsort()[::-1][:3]]
                        if is_flat_text_model(base_mdl) or float(np.max(base_probs)) < 0.30:
                            base_top = heuristic_text_prediction(text, base_classes, fallback_probs=base_probs)
                        top = base_top
                        return jsonify({'mode': 'text', 'top': top, 'pred': top[0]})
                    except Exception:
                        pass
                top = heuristic_text_prediction(text, classes, fallback_probs=probs)
                return jsonify({'mode': 'text', 'top': top, 'pred': top[0]})

            # Otherwise blend the DL output with the baseline text model if available.
            if float(np.max(probs)) < 0.55:
                base_mdl = models.get('text') or load_text_model()
                if base_mdl is not None:
                    try:
                        X = transform_text_features(text, base_mdl)
                        probs = map_and_blend_with_baseline(classes, probs, base_mdl, feat_X=X)
                    except Exception:
                        pass
            top_idx = probs.argsort()[::-1][:3]
            top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
            if float(np.max(probs)) - float(np.min(probs)) < 1e-4:
                top = heuristic_text_prediction(text, classes, fallback_probs=probs)
            return jsonify({'mode': 'text', 'top': top, 'pred': top[0]})

    # Fusion
    if mode == 'fusion':
        if audio is None:
            return jsonify({'error': 'Fusion requires an audio file so it can transcribe speech and fuse speech + text.'}), 400
        transcript, fusion_text, asr_error = build_fusion_text(audio, text)
        # enforce normalization again to be safe
        try:
            transcript = normalize_transcript(transcript)
            fusion_text = ' '.join(part for part in (transcript, text.strip()) if part)
        except Exception:
            pass
        # ASR/transcription is required for fusion — fail if ASR did not produce text
        if not transcript or asr_error:
            return jsonify({'error': 'ASR transcription failed. Fusion requires a successful transcription of the uploaded audio.', 'asr_error': asr_error}), 400
        # If transcript is very short (single-word prompts), prefer speech-only prediction
        try:
            # treat obvious prompt phrases (e.g. 'say the word ...') as prompt-only and prefer audio
            low = transcript.lower()
            if ('say' in low and 'word' in low) or len(transcript.split()) <= 2:
                # use speech baseline if available
                base = models.get('speech') or load_speech_model()
                if base is not None:
                    rewind_fileobj(audio)
                    feat = extract_mfcc_stats_fileobj(audio)
                    Xs = base['scaler'].transform([feat])
                    probs = base['model'].predict_proba(Xs)[0]
                    classes = base['label_encoder'].classes_
                    top_idx = probs.argsort()[::-1][:3]
                    top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
                    result = {'mode': 'speech', 'top': top, 'pred': top[0], 'note': 'Short transcript detected — returned speech-only prediction.'}
                    return jsonify(attach_fusion_metadata(result, transcript, asr_error))
        except Exception:
            pass
        if model_variant == 'baseline':
            mdl = models.get('fusion') or load_fusion_model()
            if mdl is None:
                fallback = predict_fusion_ensemble(audio, fusion_text, transcript, asr_error)
                if fallback is not None:
                    return jsonify(fallback)
                rewind_fileobj(audio)
                fallback = predict_dl_fusion(audio, fusion_text, transcript, asr_error)
                if fallback is not None:
                    return jsonify(fallback)
                return jsonify({'error': 'Fusion model not found. Run fusion baseline training or select Deep (DL).'}), 400
            models['fusion'] = mdl
            rewind_fileobj(audio)
            feat_audio = extract_mfcc_stats_fileobj(audio)
            X_text = mdl['tfidf'].transform([fusion_text])
            X_text_red = mdl['svd'].transform(X_text)
            X = np.hstack([feat_audio, X_text_red[0]])
            Xs = mdl['scaler'].transform([X])
            probs = mdl['model'].predict_proba(Xs)[0]
            probs = blend_fusion_probs(mdl['label_encoder'].classes_, probs, fusion_text)
            top_idx = probs.argsort()[::-1][:3]
            classes = mdl['label_encoder'].classes_
            top = [{'label': str(classes[i]), 'prob': float(probs[i])} for i in top_idx]
            result = {'mode': 'fusion', 'top': top, 'pred': top[0], 'note': 'Fusion transcribed the uploaded speech, extracted audio/text features, fused them, and classified the speaker emotion.'}
            # ensure the transcript returned to the client is normalized
            result = attach_fusion_metadata(result, transcript, asr_error)
            result['transcript'] = transcript
            return jsonify(result)
        elif model_variant == 'dl':
            result = predict_dl_fusion(audio, fusion_text, transcript, asr_error)
            if result is None:
                return jsonify({'error': 'DL fusion model not found. Train DL fusion model.'}), 400
            return jsonify(result)

        return jsonify({'error': 'Unknown mode or model variant'}), 400


@app.route('/samples/<path:name>')
def serve_sample_file(name):
    # Serve predefined sample files from the samples directory.
    allowed = set(SAMPLE_FILE_NAMES)
    if name not in allowed:
        return abort(404)
    if name == 'text_only_samples.txt':
        return Response('\n'.join(read_sample_lines()) + '\n', mimetype='text/plain')
    for base_dir in (SAMPLE_OUTPUT_DIR, SAMPLE_OUTPUT_DIR_LEGACY):
        file_path = base_dir / name
        if file_path.exists():
            return send_file(file_path, mimetype='text/plain')
    return abort(404)


@app.route('/download-samples', methods=['POST'])
def download_samples():
    try:
        result = ensure_sample_files()
        sample_lines = read_sample_lines()
        return jsonify({
            'ok': True,
            'message': f"Saved samples to {result['text_path']} and {result['fusion_path']}",
            'count': result['count'],
            'sample_lines': sample_lines,
        })
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.after_request
def _normalize_transcript_in_response(response):
    # Ensure any 'transcript' field in JSON responses is normalized for consistency.
    try:
        ct = response.content_type or ''
        if 'application/json' in ct:
            import json
            body = response.get_data(as_text=True)
            data = json.loads(body)
            if isinstance(data, dict) and 'transcript' in data:
                data['transcript'] = normalize_transcript(data.get('transcript', ''))
                response.set_data(json.dumps(data))
    except Exception:
        pass
    return response


@app.errorhandler(Exception)
def _handle_all_exceptions(e):
    # Return JSON for any unexpected server errors so the client receives JSON, not HTML.
    try:
        import traceback
        tb = traceback.format_exc()
    except Exception:
        tb = None
    return jsonify({'error': 'Internal server error', 'detail': str(e), 'trace': tb}), 500


@app.route('/upload_sample', methods=['POST'])
def upload_sample():
    """Upload a labeled audio sample for manual calibration/training.
    Expects form fields: 'label' and file field 'audio'. Saves to dataset/user_samples/<label>/
    """
    label = request.form.get('label')
    audio = request.files.get('audio')
    if not label or audio is None:
        return jsonify({'ok': False, 'error': 'Provide label and audio file'}), 400
    dest = Path('dataset/user_samples') / label
    dest.mkdir(parents=True, exist_ok=True)
    try:
        content = audio.read()
        fname = audio.filename or f'sample_{int(__import__("time").time())}.wav'
        outp = dest / fname
        with outp.open('wb') as f:
            f.write(content)
        return jsonify({'ok': True, 'path': str(outp)}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print('Starting IIITHEMOJI local server on http://127.0.0.1:8002')
    try:
        # Run without the reloader to avoid multiple processes and reload loops.
        app.run(host='127.0.0.1', port=8002, debug=False, use_reloader=False)
    except Exception as e:
        print('Failed to start server:', e)
    

