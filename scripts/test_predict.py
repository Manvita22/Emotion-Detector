import requests
import json
from pathlib import Path

url = 'http://127.0.0.1:8002/predict'
# adjust to a sample file present in dataset
fpath = Path('dataset/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav')
if not fpath.exists():
    print('Sample not found:', fpath)
else:
    files = {'audio': (fpath.name, open(fpath, 'rb'), 'audio/wav')}
    data = {'mode': 'fusion', 'model_variant': 'baseline', 'text': ''}
    r = requests.post(url, files=files, data=data)
    try:
        print('STATUS:', r.status_code)
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print('Response text:', r.text)
