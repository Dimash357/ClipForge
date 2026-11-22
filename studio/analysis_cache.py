"""On-disk cache of per-source analysis, keyed by the source fingerprint.

MEDIA_ROOT/analysis/<fp>/
    audio16k.wav        mono 16 kHz PCM (Whisper input + audio analysis input)
    audio.npz           AudioProfile frames
    screen.npz          keyframe screen scan
    scenes/<a>_<b>_<mode>.json   per-clip scene analysis (faces, cam, motion) for fast re-edits

Everything here is a cache: a missing or corrupt file is simply recomputed, never an error.
"""
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from django.conf import settings

from .procutil import run_ffmpeg

log = logging.getLogger('studio.cache')
VERSION = 2


def cache_dir(fp, create=True):
    if not fp:
        return None
    d = Path(settings.MEDIA_ROOT) / 'analysis' / fp[:24]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def extract_audio(source, dest, duration=None, cancel=None, on_progress=None):
    """Step 2 of the pipeline: mono 16 kHz PCM WAV. Returns the path or None if the source has no audio."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    tmp = dest.with_suffix('.part.wav')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-nostdin', '-i', str(source), '-vn', '-sn', '-dn',
           '-map', '0:a:0?', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(tmp)]
    try:
        run_ffmpeg(cmd, duration=duration, on_progress=on_progress, cancel_check=cancel, timeout=3600)
    except RuntimeError as exc:
        tmp.unlink(missing_ok=True)
        if 'does not contain any stream' in str(exc):
            return None
        raise
    if not tmp.is_file() or tmp.stat().st_size <= 1000:      # no audio stream -> empty/absent file
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------ npz helpers
def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, version=np.array(VERSION), **arrays)
    tmp.replace(path)


def load_npz(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z['version']) != VERSION:
                return None
            return {k: z[k] for k in z.files}
    except Exception as exc:
        log.warning('cache file %s unreadable (%s); recomputing', path, exc)
        return None


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def scene_key(start, end, faces=True):
    return f'{start:.2f}_{end:.2f}_{"f" if faces else "n"}.json'


def clear(fp):
    d = cache_dir(fp, create=False)
    if d and d.exists():
        shutil.rmtree(d, ignore_errors=True)
