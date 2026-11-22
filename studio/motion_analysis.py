"""Gameplay dynamics for highlight scoring: motion intensity, sudden motion bursts and scene cuts.

Decoding a whole multi-hour stream only for this would cost minutes, so the analyzer runs on the *best candidate
spans only* (a capped total duration), at 4 frames/s and 96 px width. Values are normalized against the median
motion of everything analyzed in the same video.
"""
import logging

import numpy as np

from . import scenes as scn

log = logging.getLogger('studio.motion')

FPS = 4.0
WIDTH = 96
MAX_TOTAL_S = 900.0


def merge_spans(spans, gap=2.0):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def color_stats(bgr):
    """(luma 0..255, saturation 0..1, red dominance) of a small BGR frame."""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1.0)))
    luma = float(np.mean(0.114 * b + 0.587 * g + 0.299 * r))
    red = float(np.mean(r - (g + b) / 2.0)) / 255.0
    return luma, sat, red


def screen_events(times, luma, sat, red, cuts, min_gap=1.5):
    """Explainable on-screen 'something happened' markers from per-frame colour statistics.

    fail_screen: picture drops to near-black / loses most of its colour / floods red (death, crash, 'mission
                 failed' screens in most games), flash: sudden bright burst (explosion), cut: hard scene change.
    """
    ev = []
    n = len(times)
    last_t = -1e9
    for k in range(1, n):
        lo = max(0, k - 3)
        ref_l = float(np.median(luma[lo:k]))
        ref_s = float(np.median(sat[lo:k]))
        ref_r = float(np.median(red[lo:k]))
        kind = None
        if ref_l > 40 and luma[k] < 0.35 * ref_l:
            kind = 'fail_screen'          # fade/cut to black
        elif ref_s > 0.25 and sat[k] < 0.35 * ref_s and 25 < luma[k] < 1.3 * ref_l + 10:
            kind = 'fail_screen'          # desaturated "wasted" style screen
        elif red[k] - ref_r > 0.18 and red[k] > 0.15:
            kind = 'fail_screen'          # red damage/death overlay
        elif luma[k] - ref_l > 70:
            kind = 'flash'
        elif cuts[k]:
            kind = 'cut'
        if kind and times[k] - last_t >= min_gap:
            ev.append({'t': round(float(times[k]), 2), 'kind': kind})
            last_t = times[k]
    return ev


def frame_stats(prev, cur):
    """(motion, cut) between two small gray frames."""
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    motion = float(diff.mean())
    h1 = np.histogram(prev, bins=16, range=(0, 256))[0].astype(np.float32)
    h2 = np.histogram(cur, bins=16, range=(0, 256))[0].astype(np.float32)
    hist = float(np.abs(h1 - h2).sum() / (2 * max(1.0, h1.sum())))
    cut = hist >= 0.45 or (motion >= 45 and hist >= 0.25)
    return motion, cut


class MotionProfile:
    def __init__(self, times, motion, cuts, luma=None, sat=None, red=None):
        self.t = np.asarray(times, np.float64)
        self.motion = np.asarray(motion, np.float32)
        self.cuts = np.asarray(cuts, bool)
        n = len(self.t)
        self.luma = np.asarray(luma if luma is not None else np.full(n, 100.0), np.float32)
        self.sat = np.asarray(sat if sat is not None else np.full(n, 0.4), np.float32)
        self.red = np.asarray(red if red is not None else np.zeros(n), np.float32)
        self.events = screen_events(self.t, self.luma, self.sat, self.red, self.cuts) if n else []
        self.ref = float(np.median(self.motion)) if len(self.motion) else 1.0
        self.ref = max(self.ref, 1.0)

    def covers(self, start, end, min_share=0.6):
        if not len(self.t):
            return False
        inside = ((self.t >= start) & (self.t < end)).sum()
        return inside >= min_share * (end - start) * FPS

    def span(self, start, end):
        m = (self.t >= start) & (self.t < end)
        if m.sum() < 4:
            return None
        mo = self.motion[m] / self.ref
        top = float(np.mean(np.sort(mo)[-max(1, len(mo) // 4):]))
        # bursts: motion jumps to >= 2x the running level
        lvl = np.convolve(mo, np.ones(8) / 8, mode='same')
        bursts = int(((mo[1:] >= 2.0 * np.maximum(lvl[:-1], 0.25)) & (mo[1:] >= 1.5)).sum())
        evs = [e for e in self.events if start <= e['t'] < end]
        return {'mean': float(mo.mean()), 'top': top, 'cuts': int(self.cuts[m].sum()), 'bursts': bursts,
                'static': float((mo < 0.2).mean()), 'screen_events': evs,
                'fail_screen': sum(1 for e in evs if e['kind'] == 'fail_screen'),
                'flash': sum(1 for e in evs if e['kind'] == 'flash')}


def analyze_spans(source, spans, src_w, src_h, cancel=None, max_total=MAX_TOTAL_S):
    """Sample the given absolute spans [(start, end)] -> MotionProfile."""
    import cv2
    times, motion, cuts, lum, sat, red = [], [], [], [], [], []
    budget = max_total
    for a, b in merge_spans(spans):
        if budget <= 0:
            break
        b = min(b, a + budget)
        budget -= b - a
        prev = None
        for t, bgr in scn.sample_frames(source, a, b, src_w, src_h, fps=FPS, width=WIDTH, cancel=cancel):
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                mo, cut = frame_stats(prev, g)
                times.append(t)
                motion.append(mo)
                cuts.append(cut)
                lu, sa, rd = color_stats(bgr)
                lum.append(lu)
                sat.append(sa)
                red.append(rd)
            prev = g
    return MotionProfile(times, motion, cuts, lum, sat, red)


def action_score(f):
    """0..100 'something is happening on screen' score + facts."""
    if not f:
        return None, []
    facts = []
    s = 30.0 + min(30.0, 18.0 * f['top']) + min(12.0, 10.0 * f['mean'])
    if f['bursts']:
        s += min(15.0, 5.0 * f['bursts'])
        facts.append(('action_burst', f['bursts']))
    if f['cuts']:
        s += min(10.0, 4.0 * f['cuts'])
        facts.append(('scene_change', f['cuts']))
    if f['static'] > 0.6:
        s -= 15
        facts.append(('static_picture', None))
    if f['top'] >= 1.6:
        facts.append(('fast_action', None))
    if f.get('fail_screen'):
        s += 14
        facts.append(('fail_screen', f['fail_screen']))
    if f.get('flash'):
        s += 5
        facts.append(('flash', f['flash']))
    return max(0.0, min(100.0, s)), facts


# ------------------------------------------------------------------ whole-source keyframe scan
class ScreenProfile:
    """Coarse whole-video screen timeline from keyframes only (cheap: no full decode)."""

    def __init__(self, times, luma, sat, red, hist_diff):
        self.t = np.asarray(times, np.float64)
        self.luma = np.asarray(luma, np.float32)
        self.sat = np.asarray(sat, np.float32)
        self.red = np.asarray(red, np.float32)
        self.hist = np.asarray(hist_diff, np.float32)
        cuts = self.hist >= 0.45
        self.events = screen_events(self.t, self.luma, self.sat, self.red, cuts, min_gap=2.0) if len(self.t) else []

    def events_in(self, a, b):
        return [e for e in self.events if a <= e['t'] < b]


def scan_keyframes(source, duration=None, cancel=None, timeout=1800, width=64):
    """Decode keyframes only (-skip_frame nokey), 64 px wide -> ScreenProfile (or None).

    Streams normally carry a keyframe every 1-4 s. If the file has sparse keyframes (fewer than one per 8 s),
    fall back to a 1 frame/s sampled scan (full decode, slower but still tiny frames)."""
    sp = _scan(source, True, cancel, timeout, width)
    if duration and (sp is None or len(sp.t) < duration / 8.0):
        sp = _scan(source, False, cancel, timeout, width)
    return sp


def _scan(source, keyframes, cancel, timeout, width):
    import re
    import subprocess
    import tempfile
    import time
    from .procutil import JobCancelled, _flags, kill_tree
    h = 36
    pre = ['-skip_frame', 'nokey'] if keyframes else []
    vf = f'scale={width}:{h}:flags=area,showinfo' if keyframes else f'fps=1,scale={width}:{h}:flags=area,showinfo'
    cmd = ['ffmpeg', '-hide_banner', '-nostdin'] + pre + ['-i', str(source), '-an', '-sn', '-dn',
                                                           '-vf', vf, '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24',
                                                           '-f', 'rawvideo', 'pipe:1']
    size = width * h * 3
    frames = []
    t0 = time.monotonic()
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, creationflags=_flags())
        try:
            while True:
                if cancel and cancel():
                    raise JobCancelled()
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError('keyframe scan timed out')
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                frames.append(np.frombuffer(buf, np.uint8).reshape(h, width, 3))
            proc.wait(timeout=30)
        finally:
            kill_tree(proc)
        err.seek(0)
        log_text = err.read().decode('utf-8', 'replace')
    times = [float(m) for m in re.findall(r'pts_time:\s*([\d.]+)', log_text)]
    n = min(len(times), len(frames))
    if n < 3:
        return None
    luma, sat, red, hd = [], [], [], []
    prev_h = None
    for k in range(n):
        lu, sa, rd = color_stats(frames[k])
        luma.append(lu)
        sat.append(sa)
        red.append(rd)
        g = frames[k].mean(axis=2)
        hist = np.histogram(g, bins=16, range=(0, 256))[0].astype(np.float32)
        hd.append(0.0 if prev_h is None else float(np.abs(hist - prev_h).sum() / (2 * max(1.0, hist.sum()))))
        prev_h = hist
    return ScreenProfile(times[:n], luma, sat, red, hd)


def screen_to_arrays(sp):
    return {'t': sp.t, 'luma': sp.luma, 'sat': sp.sat, 'red': sp.red, 'hist': sp.hist}


def screen_from_arrays(d):
    if not d or 't' not in d:
        return None
    return ScreenProfile(d['t'], d['luma'], d['sat'], d['red'], d['hist'])
