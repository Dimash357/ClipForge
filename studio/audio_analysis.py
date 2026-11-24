"""Audio Reaction Analyzer: loudness, raised voice, screams, laughter-like bursts, volume spikes and pauses.

Pure numpy on a 16 kHz mono track decoded by ffmpeg in chunks (memory stays flat for multi-hour streams).
Everything is relative to the video's own *local* baseline (rolling median over ~30 s), so a loud streamer and a
quiet one are judged the same way. These are explainable heuristics, not a neural classifier:

* volume spike  - loudness jumps >= 9 dB within 0.25 s (explosion, shout, sudden reaction);
* scream/shout  - sustained loud (>= +11 dB over baseline) voiced or bright sound with a raised pitch or a lot of
                  high-frequency energy;
* laughter      - a run of short loud bursts with a syllable rhythm of ~2.5-8 per second;
* raised voice  - voiced speech with pitch and loudness clearly above the speaker's norm;
* pause         - >= 0.7 s well below the local baseline; a pause right before a burst is a "dramatic pause".
"""
import logging
import subprocess
import tempfile
import time
import warnings
from bisect import bisect_left

import numpy as np

from .procutil import JobCancelled, _flags, kill_tree

log = logging.getLogger('studio.audio')

SR = 16000
HOP = 0.05                    # seconds per analysis frame
WIN = 1024                    # samples (64 ms) per FFT window
FLOOR_DB = -60.0
CHUNK_S = 60


class AudioProfile:
    """Per-frame features (hop = HOP seconds) plus detected events and pauses for one source file."""

    def __init__(self, db, hf, pitch, voiced, duration):
        self.hop = HOP
        self.duration = float(duration)
        self.db = db.astype(np.float32)
        self.hf = hf.astype(np.float32)
        self.pitch = pitch.astype(np.float32)
        self.voiced = voiced.astype(bool)
        self._derive()

    # ------------------------------------------------------------------ derived signals
    def _derive(self):
        n = len(self.db)
        self.base = _rolling_baseline(self.db, self.hop)
        rel = self.db - self.base
        self.rel = rel
        vp = self.pitch[self.voiced & (self.pitch > 0)]
        self.pitch_med = float(np.median(vp)) if len(vp) >= 10 else 0.0
        active = self.db > max(FLOOR_DB + 10, float(np.percentile(self.db, 20)) if n else FLOOR_DB)
        self.hf_med = float(np.median(self.hf[active])) if active.sum() >= 10 else float(np.median(self.hf)) if n else 0.0
        loud = np.clip(rel / 12.0, 0, 1.5)
        bright = np.clip((self.hf - self.hf_med) / 0.15, 0, 1.0) * (rel > 3)
        if self.pitch_med > 0:
            high = np.clip((self.pitch / self.pitch_med - 1.0) / 0.4, 0, 1.2) * self.voiced
        else:
            high = np.zeros(n, np.float32)
        arousal = 0.55 * loud + 0.2 * bright + 0.25 * high
        self.arousal = _smooth(arousal, max(1, int(round(0.3 / self.hop)))).astype(np.float32)
        self.sound = getattr(self, 'sound', None)
        self.events = detect_events(self)
        self.pauses = detect_pauses(self)
        self._ev_starts = [e['start'] for e in self.events]
        self._cum_ar = np.concatenate([[0.0], np.cumsum(self.arousal, dtype=np.float64)])

    def attach_sound(self, timeline):
        """Use a YAMNet SoundTimeline: the classifier names the events (laughter, scream, explosion, gunshot, cheer),
        the loudness analysis keeps their timing. Without it the heuristic labels stay."""
        self.sound = timeline
        self.events = detect_events(self)
        self._ev_starts = [e['start'] for e in self.events]
        return self

    # ------------------------------------------------------------------ queries
    def idx(self, t):
        return int(max(0, min(len(self.db), round(t / self.hop))))

    def events_in(self, start, end):
        k = bisect_left(self._ev_starts, start - 30)
        out = []
        for e in self.events[k:]:
            if e['start'] >= end:
                break
            if e['end'] > start:
                out.append(e)
        return out

    def summary(self):
        kinds = {}
        for e in self.events:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        return {'frames': len(self.db), 'events': len(self.events), 'kinds': kinds, 'pauses': len(self.pauses),
                'pitch_median': round(self.pitch_med, 1)}


# ------------------------------------------------------------------ signal helpers
def _smooth(x, k):
    if k <= 1 or len(x) == 0:
        return x
    ker = np.ones(k, np.float32) / k
    return np.convolve(x, ker, mode='same')


def _rolling_baseline(db, hop, window_s=30.0):
    """Robust local speech level: median of active frames per second, then a rolling median over ~30 s."""
    n = len(db)
    if n == 0:
        return db.copy()
    per = max(1, int(round(1.0 / hop)))
    nb = (n + per - 1) // per
    pad = np.full(nb * per, np.nan, np.float32)
    pad[:n] = db
    blocks = pad.reshape(nb, per)
    act = np.where(blocks > FLOOR_DB + 10, blocks, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # all-silent seconds -> NaN, handled below
        lvl = np.nanpercentile(act, 60, axis=1)
    glob = float(np.nanmedian(lvl)) if np.isfinite(lvl).any() else FLOOR_DB
    lvl = np.where(np.isfinite(lvl), lvl, np.nan)
    half = int(window_s // 2)
    base = np.empty(nb, np.float32)
    for i in range(nb):
        seg = lvl[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        base[i] = np.median(seg) if len(seg) >= 3 else glob
    return np.repeat(base, per)[:n]


def _runs(mask):
    """[(i0, i1_exclusive)] of True runs."""
    if not len(mask):
        return []
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def frame_features(x, sr=SR, hop_s=HOP, win=WIN):
    """x: mono float32 in [-1, 1] -> (db, hf_ratio, pitch_hz, voiced) per hop."""
    hop = int(round(hop_s * sr))
    n = max(0, 1 + (len(x) - win) // hop) if len(x) >= win else 0
    if n == 0:
        z = np.zeros(0, np.float32)
        return z, z, z, z.astype(bool)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(win).astype(np.float32)[None, :]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = np.maximum(FLOOR_DB, 20 * np.log10(rms + 1e-9) + 3.0)
    spec = np.abs(np.fft.rfft(frames, n=2 * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(2 * win, 1.0 / sr)
    total = spec[:, freqs >= 60].sum(axis=1) + 1e-9
    hf = spec[:, freqs >= 2000].sum(axis=1) / total
    # pitch: normalized autocorrelation (Wiener-Khinchin) in 75..600 Hz
    ac = np.fft.irfft(spec, axis=1)[:, :win]
    lo, hi = int(sr / 600), int(sr / 75)
    r0 = ac[:, :1] + 1e-9
    seg = ac[:, lo:hi] / r0
    k = seg.argmax(axis=1)
    peak = seg[np.arange(n), k]
    pitch = sr / (k + lo)
    voiced = (peak > 0.45) & (db > -45)
    pitch = np.where(voiced, pitch, 0.0)
    return db.astype(np.float32), hf.astype(np.float32), pitch.astype(np.float32), voiced


def profile_from_samples(x, sr=SR):
    """Build a profile from an in-memory mono signal (tests, short files)."""
    x = np.asarray(x, np.float32)
    if sr != SR:
        raise ValueError('expected 16 kHz')
    db, hf, pitch, voiced = frame_features(x)
    return AudioProfile(db, hf, pitch, voiced, len(x) / SR)


# ------------------------------------------------------------------ event detection
def _laugh_like(p, a, b):
    """Syllable rhythm: count loud local maxima separated by dips (>= 4 dB) per second."""
    seg = p.db[a:b]
    if len(seg) < int(0.7 / p.hop):
        return False, 0.0
    peaks = 0
    last_val = None
    trough = None
    for i in range(1, len(seg) - 1):
        v = seg[i]
        if trough is None or v < trough:
            trough = v
        if v >= seg[i - 1] and v > seg[i + 1] and v - trough >= 4.0:
            if last_val is None or i - last_val >= 2:
                peaks += 1
                last_val = i
                trough = v
    rate = peaks / (len(seg) * p.hop)
    return 2.5 <= rate <= 8.5 and peaks >= 3, rate


def detect_events(p, thr=0.55):
    """Contiguous high-arousal regions + instant volume spikes, classified into kinds."""
    n = len(p.db)
    if n == 0:
        return []
    hot = (p.arousal >= thr) & (p.rel >= 5.0)
    # bridge tiny gaps (laughter / shouting has short dips)
    gap = int(round(0.4 / p.hop))
    runs = []
    for a, b in _runs(hot):
        if runs and a - runs[-1][1] <= gap:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    events = []
    for a, b in runs:
        dur = (b - a) * p.hop
        if dur < 0.25:
            continue
        rel = p.rel[a:b]
        peak_i = a + int(np.argmax(p.arousal[a:b]))
        loud = float(np.percentile(rel, 80))
        hf = float(np.mean(p.hf[a:b]))
        vs = p.voiced[a:b]
        pr = p.pitch[a:b][vs]
        pitch_up = (float(np.median(pr)) / p.pitch_med) if (p.pitch_med > 0 and len(pr) >= 3) else 1.0
        laugh, rate = _laugh_like(p, a, b)
        if laugh and loud >= 5:
            kind = 'laughter'
        elif loud >= 11 and dur >= 0.4 and (pitch_up >= 1.3 or hf >= p.hf_med + 0.12):
            kind = 'scream'
        elif pitch_up >= 1.2 and vs.mean() >= 0.4:
            kind = 'raised_voice'
        else:
            kind = 'loud'
        strength = float(min(2.0, np.max(p.arousal[a:b])))
        events.append({'start': round(float(a * p.hop), 2), 'end': round(float(b * p.hop), 2),
                       'peak': round(float(peak_i * p.hop), 2),
                       'kind': kind, 'strength': round(strength, 2), 'loud_db': round(loud, 1)})
    # sudden volume jumps
    k = max(1, int(round(0.25 / p.hop)))
    if n > k:
        jump = p.db[k:] - p.db[:-k]
        cand = np.flatnonzero((jump >= 9.0) & (p.rel[k:] >= 6.0)) + k
        last = -10 ** 9
        for i in cand:
            if i - last < int(1.0 / p.hop):
                continue
            last = i
            t = round(float(i * p.hop), 2)
            if any(e['start'] - 0.3 <= t <= e['end'] for e in events):
                for e in events:
                    if e['start'] - 0.3 <= t <= e['end']:
                        e['spike'] = True
                continue
            events.append({'start': round(max(0, t - 0.15), 2), 'end': round(t + 0.6, 2), 'peak': t, 'kind': 'spike',
                           'strength': round(float(min(2.0, jump[i - k] / 12.0)), 2), 'loud_db': round(float(p.rel[i]), 1)})
    events.sort(key=lambda e: e['start'])
    if getattr(p, 'sound', None) is not None:
        events = _apply_sound_model(p, events)
    # dramatic pause: an event right after >= 0.7 s of near-silence
    quiet = p.rel < -12
    for e in events:
        i = p.idx(e['start'])
        j0 = max(0, i - int(round(1.2 / p.hop)))
        pre = quiet[j0:max(j0, i - 1)]
        if len(pre) and pre.mean() >= 0.6 and len(pre) * p.hop >= 0.7:
            e['after_pause'] = True
    return events


MODEL_PRIORITY = ('laughter', 'scream', 'explosion', 'gunshot', 'cheer')
WEAK_SUPPORT = 0.04     # a laughter/scream found by the loudness rhythm stays if the classifier gives at least this


def _apply_sound_model(p, events):
    """Relabel loudness events with the classifier and add classifier-only events (e.g. quiet laughter)."""
    from .sound_events import LOUD_NEEDED, THRESH
    tl = p.sound
    out = []
    for e in events:
        # only classifier events that passed all its checks (threshold, look-alikes) may rename a loud moment
        near = [m for m in tl.events if m['start'] < e['end'] + 0.3 and e['start'] - 0.3 < m['end']]
        best = max(near, key=lambda m: m['prob'] / THRESH[m['kind']], default=None)
        if best is not None:
            e = dict(e, kind=best['kind'], model=best['prob'])
        elif e['kind'] in ('laughter', 'scream') and tl.max_in(e['start'], e['end']).get(e['kind'], 0.0) >= WEAK_SUPPORT:
            e = dict(e, model=round(tl.max_in(e['start'], e['end'])[e['kind']], 3))   # rhythm + weak model agreement
        elif e['kind'] in ('laughter', 'scream'):
            e = dict(e, kind='loud', model=0.0)        # the classifier does not hear laughter/screaming here
        out.append(e)
    for m in tl.events:
        if any(o['start'] < m['end'] and m['start'] < o['end'] for o in out):
            continue
        a, b = p.idx(m['start']), max(p.idx(m['start']) + 1, p.idx(m['end']))
        if len(p.rel[a:b]) and float(np.max(p.rel[a:b])) < LOUD_NEEDED.get(m['kind'], 0.0):
            continue
        out.append({'start': m['start'], 'end': m['end'], 'peak': m['peak'], 'kind': m['kind'],
                    'strength': round(min(2.0, 0.6 + m['prob']), 2), 'loud_db': None, 'model': m['prob']})
    out.sort(key=lambda e: e['start'])
    return out


def detect_pauses(p, min_len=0.7):
    quiet = (p.rel < -12) | (p.db <= FLOOR_DB + 5)
    out = []
    for a, b in _runs(quiet):
        if (b - a) * p.hop >= min_len:
            out.append((round(float(a * p.hop), 2), round(float(b * p.hop), 2)))
    return out


# ------------------------------------------------------------------ per-span features
def span_features(p, start, end):
    """Features of [start, end) for highlight scoring."""
    a, b = p.idx(start), p.idx(end)
    if b - a < 4:
        return None
    ar = p.arousal[a:b]
    rel = p.rel[a:b]
    top = float(np.mean(np.sort(ar)[-max(1, len(ar) // 5):]))
    mean_ar = float((p._cum_ar[b] - p._cum_ar[a]) / (b - a))
    active = rel > -12
    dyn = float(np.std(rel[active])) if active.sum() > 5 else 0.0
    silence = 1.0 - float(active.mean())
    evs = p.events_in(start, end)
    kinds = {}
    for e in evs:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    strong = [e for e in evs if e['strength'] >= 0.8 or e['kind'] in ('scream', 'laughter', 'explosion')]
    peak_rel = (int(np.argmax(ar)) / max(1, len(ar) - 1)) if len(ar) else 0.5
    # longest stretch of silence inside the span
    longest = 0.0
    for s0, s1 in p.pauses:
        if s1 > start and s0 < end:
            longest = max(longest, min(s1, end) - max(s0, start))
    return {'top': top, 'mean': mean_ar, 'dyn': dyn, 'silence': silence, 'events': evs, 'kinds': kinds,
            'strong': len(strong), 'peak_rel': peak_rel, 'peak_t': start + peak_rel * (end - start),
            'longest_pause': longest, 'dramatic_pause': any(e.get('after_pause') for e in evs),
            'spike': any(e['kind'] == 'spike' or e.get('spike') for e in evs)}


def audio_score(f):
    """0..100 emotional-energy score + facts list from span_features()."""
    if not f:
        return None, []
    facts = []
    s = 34.0
    s += min(30.0, f['top'] * 30.0)
    s += min(18.0, 6.0 * f['strong'])
    k = f['kinds']
    if k.get('laughter'):
        s += 12
        facts.append(('laughter', k['laughter']))
    if k.get('scream'):
        s += 12
        facts.append(('scream', k['scream']))
    if k.get('explosion'):
        s += 8
        facts.append(('explosion', k['explosion']))
    if k.get('gunshot'):
        s += 4
        facts.append(('gunshot', k['gunshot']))
    if k.get('cheer'):
        s += 5
        facts.append(('cheer', k['cheer']))
    if f['spike']:
        s += 6
        facts.append(('volume_spike', None))
    if k.get('raised_voice'):
        s += 5
        facts.append(('raised_voice', k['raised_voice']))
    if f['dramatic_pause']:
        s += 6
        facts.append(('dramatic_pause', None))
    if f['dyn'] < 2.0 and not f['strong']:
        s -= 8
        facts.append(('monotone', None))
    else:
        s += max(-4.0, min(8.0, (f['dyn'] - 4.0) * 1.5))
    if f['silence'] > 0.35:
        s -= min(20.0, (f['silence'] - 0.35) * 60)
        facts.append(('quiet', round(f['silence'], 2)))
    return max(0.0, min(100.0, s)), facts


# ------------------------------------------------------------------ ffmpeg front end
def analyze_file(source, duration=None, cancel=None, progress=None, timeout=3600):
    """Decode the audio track and build a profile. Returns None if the file has no decodable audio."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source), '-vn', '-sn', '-dn',
           '-map', '0:a:0?', '-ac', '1', '-ar', str(SR), '-f', 's16le', 'pipe:1']
    hop = int(round(HOP * SR))
    chunk = CHUNK_S * SR
    parts = []
    tail = np.zeros(0, np.float32)
    total = 0
    t_start = time.monotonic()
    with tempfile.TemporaryFile() as err:
        from .procutil import Watchdog
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                                creationflags=_flags())
        dog = Watchdog(proc, 'Анализ звука')
        try:
            while True:
                if cancel and cancel():
                    raise JobCancelled()
                if time.monotonic() - t_start > timeout:
                    raise RuntimeError('audio analysis timed out')
                buf = proc.stdout.read(chunk * 2)
                if not buf:
                    break
                dog.touch()
                x = np.frombuffer(buf[:len(buf) // 2 * 2], np.int16).astype(np.float32) / 32768.0
                total += len(x)
                x = np.concatenate([tail, x])
                usable = ((len(x) - WIN) // hop + 1) * hop if len(x) >= WIN else 0
                if usable > 0:
                    parts.append(frame_features(x[:usable - hop + WIN]))
                    tail = x[usable:]
                else:
                    tail = x
                if progress and duration:
                    progress(min(1.0, total / SR / duration))
            proc.wait(timeout=30)
        finally:
            dog.stop()
            kill_tree(proc)
        dog.check()
        if proc.returncode not in (0, None) and total == 0:
            err.seek(0)
            msg = err.read().decode('utf-8', 'replace')
            if 'does not contain any stream' in msg:      # optional map '0:a:0?' matched nothing: no audio track
                return None
            raise RuntimeError('ffmpeg audio decode failed: ' + msg[-400:])
    if total < SR:          # no audio stream (map '?' yields nothing) or under a second of sound
        return None
    db = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.float32)
    hf = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.float32)
    pitch = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.float32)
    voiced = np.concatenate([p[3] for p in parts]) if parts else np.zeros(0, bool)
    if len(db) < 20:
        return None
    return AudioProfile(db, hf, pitch, voiced, total / SR)


# ------------------------------------------------------------------ cache (de)serialisation
def to_arrays(p):
    return {'db': p.db, 'hf': p.hf, 'pitch': p.pitch, 'voiced': p.voiced, 'duration': np.array(p.duration)}


def from_arrays(d):
    if not d or 'db' not in d:
        return None
    return AudioProfile(d['db'], d['hf'], d['pitch'], d['voiced'], float(d['duration']))
