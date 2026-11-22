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
    def __init__(self, times, motion, cuts):
        self.t = np.asarray(times, np.float64)
        self.motion = np.asarray(motion, np.float32)
        self.cuts = np.asarray(cuts, bool)
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
        return {'mean': float(mo.mean()), 'top': top, 'cuts': int(self.cuts[m].sum()), 'bursts': bursts,
                'static': float((mo < 0.2).mean())}


def analyze_spans(source, spans, src_w, src_h, cancel=None, max_total=MAX_TOTAL_S):
    """Sample the given absolute spans [(start, end)] -> MotionProfile."""
    import cv2
    times, motion, cuts = [], [], []
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
            prev = g
    return MotionProfile(times, motion, cuts)


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
    return max(0.0, min(100.0, s)), facts
