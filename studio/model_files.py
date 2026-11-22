"""Model files used by ClipForge (standard library only: tools/preflight.py imports it before Django is set up).

* YuNet face detector  - bundled in studio/models (232 KB, MIT, from huggingface.co/opencv/face_detection_yunet).
* YAMNet sound classes - 16 MB ONNX (AudioSet, 521 classes; waveform in, scores out), downloaded once into
  <project>/models/ from Hugging Face, pinned to a commit and checked by SHA-256 (like the Whisper model).
"""
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv('CLIPFORGE_MODELS_DIR', str(ROOT / 'models')))

YAMNET = {
    'file': 'yamnet.onnx',
    'url': 'https://huggingface.co/jafet21/yamnetonnx/resolve/337f563004630f00673d4a773a0d657d0233ce56/yamnet.onnx',
    'sha256': '04e27fca08e7a3aea2630d1a63a51e6b437c803e0ff6e26399f80870ac251dda',
    'size': 16124200,
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def model_path(spec=YAMNET):
    return MODELS_DIR / spec['file']


def is_ready(spec=YAMNET):
    """File present with the right size and hash (hash verified once, remembered in a .ok marker)."""
    p = model_path(spec)
    if not p.is_file() or p.stat().st_size != spec['size']:
        return False
    marker = p.with_suffix(p.suffix + '.ok')
    if marker.is_file() and marker.read_text().strip() == spec['sha256']:
        return True
    if sha256(p) == spec['sha256']:
        marker.write_text(spec['sha256'])
        return True
    return False


def download(spec=YAMNET, timeout=60, progress=None):
    """Download + verify. Returns the path or raises (callers fall back to the heuristic analysis)."""
    if is_ready(spec):
        return model_path(spec)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = model_path(spec).with_suffix('.part')
    req = urllib.request.Request(spec['url'], headers={'User-Agent': 'clipforge-model-fetch'})
    got = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(part, 'wb') as f:
        while True:
            block = r.read(1 << 16)
            if not block:
                break
            f.write(block)
            got += len(block)
            if progress:
                progress(min(1.0, got / spec['size']))
    if sha256(part) != spec['sha256']:
        part.unlink(missing_ok=True)
        raise RuntimeError('контрольная сумма модели не совпала — файл не сохранён')
    part.replace(model_path(spec))
    is_ready(spec)
    return model_path(spec)
