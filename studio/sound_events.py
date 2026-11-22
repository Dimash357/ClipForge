"""Sound event classifier (YAMNet, AudioSet) on top of the loudness heuristics of audio_analysis.

Runs once per source on the cached mono 16 kHz track (about 15 s of CPU per hour of audio), keeps only the
groups ClipForge needs and caches them. If the model cannot be loaded (offline first run, missing onnxruntime),
callers keep the heuristic events - nothing fails.
"""
import csv
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import model_files
from .procutil import JobCancelled, _flags, kill_tree

log = logging.getLogger('studio.sound')

SR = 16000
HOP = 0.48                 # YAMNet frame hop (s); each frame looks at 0.96 s
WIN_SAMPLES = 15600
HOP_SAMPLES = 7680
GROUPS = {
    'laughter': ['Laughter', 'Baby laughter', 'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle'],
    'scream': ['Screaming', 'Shout', 'Yell', 'Bellow', 'Whoop', 'Battle cry', 'Children shouting'],
    'explosion': ['Explosion', 'Boom', 'Eruption'],
    'gunshot': ['Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire'],
    'cheer': ['Cheering', 'Applause', 'Crowd'],
    'speech': ['Speech'],
    'music': ['Music'],
    # "look-alikes": an event only counts if the model does not hear its look-alike at least as strongly
    'x_explosion': ['Walk, footsteps', 'Run', 'Shuffle', 'Thump, thud', 'Knock', 'Tap', 'Slam', 'Door', 'Chop',
                    'Wood', 'Chopping (food)'],
    'x_laughter': ['Crying, sobbing', 'Baby cry, infant cry', 'Whimper', 'Animal', 'Frog', 'Chicken, rooster',
                   'Bird'],
}
LOOKALIKE = {'explosion': 'x_explosion', 'gunshot': 'x_explosion', 'laughter': 'x_laughter', 'scream': 'x_laughter'}
# probability needed for an event (YAMNet scores are per-class sigmoids)
THRESH = {'laughter': 0.25, 'scream': 0.30, 'explosion': 0.40, 'gunshot': 0.40, 'cheer': 0.40}
# a classifier-only event of these kinds must also be clearly louder than the stream's own level (dB over the local
# baseline): a real in-game explosion is loud, footsteps or a can opening that "sound like" one are not
LOUD_NEEDED = {'explosion': 8.0, 'gunshot': 6.0, 'cheer': 4.0, 'scream': 3.0, 'laughter': -99.0}
EVENT_KINDS = tuple(THRESH)
CLASS_MAP = Path(__file__).resolve().parent / 'models' / 'yamnet_class_map.csv'

_session = None


def _class_index():
    names = [row[2] for row in list(csv.reader(CLASS_MAP.open(encoding='utf-8')))[1:]]
    return {g: [names.index(n) for n in members if n in names] for g, members in GROUPS.items()}


def load(download=True, progress=None):
    """ONNX session or None (no model / no onnxruntime / offline)."""
    global _session
    if _session is not None:
        return _session
    try:
        if not model_files.is_ready():
            if not download:
                return None
            model_files.download(progress=progress)
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(str(model_files.model_path()), opts, providers=['CPUExecutionProvider'])
        return _session
    except Exception as exc:
        log.warning('sound classifier unavailable (%s); loudness heuristics only', str(exc)[:200])
        return None


class SoundTimeline:
    """Per-frame group probabilities (frame k covers [k*HOP, k*HOP + 0.96 s])."""

    def __init__(self, groups, hop=HOP):
        self.hop = hop
        self.groups = {k: np.asarray(v, np.float32) for k, v in groups.items()}
        self.n = len(next(iter(self.groups.values()))) if self.groups else 0
        self.events = self._events()

    def _events(self):
        out = []
        for kind, thr in THRESH.items():
            p = self.groups.get(kind)
            if p is None:
                continue
            hot = p >= thr
            alike = self.groups.get(LOOKALIKE.get(kind, ''))
            if alike is not None:
                # e.g. footsteps / a door slam that "sound like" a blast: compare with the look-alike in the same
                # frame AND in the surrounding +-3 s (a lone thump inside a walking sequence is not an explosion)
                w = int(round(3.0 / self.hop))
                near = np.array([alike[max(0, i - w):i + w + 1].max() for i in range(self.n)], np.float32)
                hot &= (p >= 0.9 * alike) & (p * 0.5 >= near * (1.0 if kind in ('explosion', 'gunshot') else 0.0))
            k = 0
            while k < self.n:
                if not hot[k]:
                    k += 1
                    continue
                j = k
                while j + 1 < self.n and (hot[j + 1] or (j + 2 < self.n and hot[j + 2])):   # bridge 1-frame gaps
                    j += 1
                seg = p[k:j + 1]
                peak = k + int(np.argmax(seg))
                out.append({'start': round(k * self.hop, 2), 'end': round(j * self.hop + 0.96, 2),
                            'peak': round(peak * self.hop + 0.48, 2), 'kind': kind, 'prob': round(float(seg.max()), 3)})
                k = j + 1
        out.sort(key=lambda e: e['start'])
        return out

    def max_in(self, start, end):
        """Max probability per group in [start, end); a group whose look-alike is stronger in a frame scores 0 there."""
        a = max(0, int(start / self.hop) - 1)
        b = min(self.n, int(end / self.hop) + 1)
        if b <= a:
            return {}
        out = {}
        for k, v in self.groups.items():
            seg = v[a:b]
            alike = self.groups.get(LOOKALIKE.get(k, ''))
            if alike is not None:
                seg = np.where(seg >= 0.9 * alike[a:b], seg, 0.0)
            out[k] = float(seg.max())
        return out

    def summary(self):
        kinds = {}
        for e in self.events:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        return kinds


def classify_samples(x, session=None):
    """In-memory mono 16 kHz float32 -> SoundTimeline."""
    session = session or load()
    if session is None:
        return None
    idx = _class_index()
    scores = _run(session, np.asarray(x, np.float32))
    return SoundTimeline({g: scores[:, ids].max(axis=1) if ids else np.zeros(len(scores)) for g, ids in idx.items()})


def _run(session, x):
    name = session.get_inputs()[0].name
    if len(x) < WIN_SAMPLES:
        x = np.pad(x, (0, WIN_SAMPLES - len(x)))
    return session.run(None, {name: x.astype(np.float32)})[0]


def classify_file(path, duration=None, cancel=None, progress=None, chunk_frames=124):
    """Decode `path` (the cached audio16k.wav, or any media) with ffmpeg and classify it chunk by chunk."""
    session = load()
    if session is None:
        return None
    idx = _class_index()
    step = chunk_frames * HOP_SAMPLES                      # samples per chunk, frame-aligned
    need = step + WIN_SAMPLES - HOP_SAMPLES                # + overlap so frames continue seamlessly
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(path), '-vn', '-sn', '-dn',
           '-map', '0:a:0?', '-ac', '1', '-ar', str(SR), '-f', 's16le', 'pipe:1']
    parts = {g: [] for g in idx}
    buf = np.zeros(0, np.float32)
    done = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, creationflags=_flags())
        try:
            eof = False
            while not eof or len(buf) >= WIN_SAMPLES:
                if cancel and cancel():
                    raise JobCancelled()
                while not eof and len(buf) < need:
                    raw = proc.stdout.read(need * 2)
                    if not raw:
                        eof = True
                        break
                    buf = np.concatenate([buf, np.frombuffer(raw[:len(raw) // 2 * 2], np.int16).astype(np.float32) / 32768.0])
                if len(buf) < WIN_SAMPLES:
                    break
                chunk = buf[:need]
                sc = _run(session, chunk)
                keep = sc if (eof and len(buf) <= need) else sc[:chunk_frames]
                for g, ids in idx.items():
                    parts[g].append(keep[:, ids].max(axis=1) if ids else np.zeros(len(keep), np.float32))
                buf = buf[len(keep) * HOP_SAMPLES:]
                done += len(keep) * HOP
                if progress and duration:
                    progress(min(1.0, done / duration))
                if eof and len(buf) < WIN_SAMPLES:
                    break
        finally:
            kill_tree(proc)
    if not parts['speech']:
        return None
    return SoundTimeline({g: np.concatenate(v) for g, v in parts.items()})


def to_arrays(tl):
    return {f'g_{k}': v for k, v in tl.groups.items()}


def from_arrays(d):
    if not d:
        return None
    groups = {k[2:]: v for k, v in d.items() if k.startswith('g_')}
    return SoundTimeline(groups) if groups else None


def status():
    return {'model': model_files.is_ready(), 'path': str(model_files.model_path())}
