import io, os, sys
sys.path.insert(0, os.path.abspath('.'))
from app import transcribe_audio_fileobj, normalize_transcript
p = 'dataset/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav'
with open(p, 'rb') as f:
    b = io.BytesIO(f.read())
    b.filename = 'OAF_back_angry.wav'
    raw = transcribe_audio_fileobj(b)
    print('raw:', repr(raw))
    print('norm:', repr(normalize_transcript(raw)))
