"""
Create small local text samples for the emotion demo.

The project text baseline works best when sample lines contain clear emotion
keywords. This script now writes curated emotion-aligned examples into:
- dataset/text_samples/text_only_samples.txt
- dataset/text_samples/fusion_samples.txt

The files are also mirrored into the legacy samples/ folder for compatibility.
"""
import argparse
from pathlib import Path


EMOTION_SAMPLES = {
    'angry': [
        'I am furious and completely fed up.',
        'That made me angry and frustrated.',
        'I cannot believe how annoying this is.',
        'I am mad right now.',
    ],
    'disgust': [
        'That is disgusting and gross.',
        'I feel sick and repulsed.',
        'What a nasty and horrible mess.',
        'This really turned my stomach.',
    ],
    'fear': [
        'I am scared and worried about what happens next.',
        'This makes me anxious and afraid.',
        'I feel nervous and frightened.',
        'Something bad is about to happen and I am terrified.',
    ],
    'happy': [
        'I am happy and excited today.',
        'This is wonderful and makes me smile.',
        'I feel delighted and full of joy.',
        'What a great and cheerful moment.',
    ],
    'neutral': [
        'I am calm and okay with that.',
        'This seems fine and ordinary.',
        'Nothing special happened today.',
        'I feel neutral about the whole thing.',
    ],
    'pleasant_surprise': [
        'Wow, that is a pleasant surprise.',
        'I am amazed and pleasantly surprised.',
        'What a delightful unexpected moment.',
        'That caught me off guard in a good way.',
    ],
    'sad': [
        'I feel sad and lonely today.',
        'This is heartbreaking and upsetting.',
        'I am deeply sad and down.',
        'I feel miserable and disappointed.',
    ],
}


def build_samples(n):
    samples = []
    ordered_emotions = list(EMOTION_SAMPLES.keys())
    while len(samples) < n:
        for emotion in ordered_emotions:
            for sentence in EMOTION_SAMPLES[emotion]:
                samples.append(sentence)
                if len(samples) >= n:
                    return samples
    return samples


def write_samples(dataset_name=None, n=30, out_dir='dataset/text_samples'):
    del dataset_name
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    text_path = out_path / 'text_only_samples.txt'
    fusion_path = out_path / 'fusion_samples.txt'
    legacy_dir = Path('samples')
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_text_path = legacy_dir / 'text_only_samples.txt'
    legacy_fusion_path = legacy_dir / 'fusion_samples.txt'

    samples = build_samples(n)
    with open(text_path, 'w', encoding='utf-8') as f_txt, open(fusion_path, 'w', encoding='utf-8') as f_fus, open(legacy_text_path, 'w', encoding='utf-8') as f_txt_legacy, open(legacy_fusion_path, 'w', encoding='utf-8') as f_fus_legacy:
        for txt in samples:
            f_txt.write(txt + '\n')
            f_fus.write(txt + '\n')
            f_txt_legacy.write(txt + '\n')
            f_fus_legacy.write(txt + '\n')

    print(f'Wrote {len(samples)} lines to {text_path} and {fusion_path}')
    return {
        'count': len(samples),
        'text_path': str(text_path),
        'fusion_path': str(fusion_path),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20, help='Number of sample lines to save')
    p.add_argument('--out-dir', default='dataset/text_samples', help='Directory to write sample files into')
    args = p.parse_args()
    write_samples(None, args.n, args.out_dir)
