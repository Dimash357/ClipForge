"""AI Hook Engine: the strongest first 1-3 seconds of a Short.

For one clip it builds several possible openings and gives each a Hook Score 0..100 with reasons:
* original - the clip as chosen by the highlight ranker;
* start    - begin at another sentence (later: skip a weak / slow / context-dependent opening; slightly earlier:
             pull in a strong question or hook line). Only sentence starts are used, the end stays, the clip
             length stays within the project's limits and the clip's main moment (payoff) is never cut away;
* teaser   - a cold open: a 1-3 s peak from inside the clip (laugh, scream, exclamation, fail screen...) cut on word
             gaps and played first, then the whole clip from its start - nothing is removed, so meaning is kept.

Score = speech (35) + emotion (25) + visual dynamics (20) + coherence (20). Inputs are what the pipeline already
has: the transcript (words), the cached audio profile with sound-classifier events, the cached screen profile and,
when given, the gameplay-motion probe. No new models.

A teaser is applied through Timeline IR: prepend_teaser() edits the clip's timeline snapshot and the timeline
renderer renders it (1B), so the classic renderer stays untouched.
"""
import copy
import logging
import re

import numpy as np

from . import highlights as hl
from . import moments as mo
from . import textutil as tx

log = logging.getLogger('studio.hooks')

VERSION = 1
WINDOW = 3.0                 # the hook: the first seconds a viewer sees
MARGIN = 6.0                 # a variant must beat the original by this much to be recommended
EARLIER_S = 4.0              # how far before the clip a strong line may pull the start
TRIM_SHARE = 0.45            # at most this share of the clip may be trimmed away at the start
AUDIO_W = {'laughter': 1.0, 'scream': 1.0, 'cheer': 0.8, 'explosion': 0.8, 'raised_voice': 0.7, 'gunshot': 0.6,
           'spike': 0.5, 'loud': 0.4}
SCREEN_W = {'fail_screen': 1.0, 'flash': 0.6, 'cut': 0.35}
KIND_RU = {'original': 'как есть', 'start': 'другое начало', 'teaser': 'тизер в начале'}
EVENT_RU = {'laughter': 'смех', 'scream': 'крик', 'cheer': 'ликование', 'explosion': 'взрыв', 'raised_voice': 'громкий голос',
            'gunshot': 'выстрел', 'spike': 'резкий звук', 'loud': 'громко', 'fail_screen': 'экран поражения',
            'flash': 'вспышка', 'cut': 'смена сцены'}


def _round(x):
    return round(float(x), 2)


class Signals:
    """Everything the scorer looks at, for one source."""

    def __init__(self, segments, audio=None, screen=None, motion=None):
        self.words = [w for w in tx.flat_words(segments or []) if (w.get('text') or '').strip()]
        self.sents = tx.build_sentences(self.words) if self.words else []
        self.audio = audio
        self.screen = screen
        self.motion = motion            # MotionProfile covering the hook windows, or None

    def words_in(self, a, b):
        return [w for w in self.words if w['end'] > a and w['start'] < b]

    def arousal(self, a, b):
        p = self.audio
        if p is None or not len(p.arousal):
            return None
        i, j = p.idx(a), max(p.idx(a) + 1, p.idx(b))
        return float(np.mean(p.arousal[i:j])) if j <= len(p.arousal) else None


# ------------------------------------------------------------------ scoring
def _speech(sig, a, b, kind, reasons):
    ws = sig.words_in(a, b)
    text = ' '.join(w['text'] for w in ws)
    low = text.lower()
    s = 8.0
    if not ws:
        reasons.append((-8, 'тишина в первые секунды'))
        return 0.0, text
    lead = ws[0]['start'] - a
    if lead > 1.5:
        s -= 8
        reasons.append((-8, f'речь начинается только через {lead:.1f} с'))
    elif lead > 0.8:
        s -= 4
        reasons.append((-4, f'пауза {lead:.1f} с перед первым словом'))
    n_hook = len(hl.HOOK.findall(low))
    if n_hook:
        s += min(2, n_hook) * 7
        reasons.append((min(2, n_hook) * 7, 'цепляющее слово'))
    if '?' in text or hl.Q_START.match(low.strip()):
        s += 9
        reasons.append((9, 'вопрос'))
    cues = mo.sentence_cues(text)
    if cues:
        pts = min(14, sum(mo.CUE_WEIGHT.get(k, 8) for k in cues) // 2)
        s += pts
        reasons.append((pts, ', '.join(mo.CUE_LABEL_RU.get(k, k) for k in cues)))
    if hl.EMO.search(low):
        s += 5
        reasons.append((5, 'эмоциональное слово'))
    if hl.DIGIT.search(text):
        s += 3
        reasons.append((3, 'число'))
    rate = len(ws) / max(0.5, b - a)
    if rate >= 1.6:
        s += 4
        reasons.append((4, 'быстрая живая речь'))
    first = ws[0]['text'].lower().strip('.,!?…«»"\'')
    if kind != 'teaser' and (hl.CONT_START.fullmatch(first) or hl.FILLER.fullmatch(first) or
                             hl.ANAPHORA.fullmatch(first)):
        s -= 10
        reasons.append((-10, f'начинается с «{ws[0]["text"]}» — звучит как продолжение'))
    return max(0.0, min(35.0, s)), text


def _emotion(sig, a, b, ref, reasons):
    s = 0.0
    ar = sig.arousal(a, b)
    if ar is not None and ref is not None:
        lift = ar - ref
        pts = max(0.0, min(12.0, lift * 30 + 4))
        s += pts
        if lift > 0.12:
            reasons.append((round(pts), 'голос/звук заметно эмоциональнее остального клипа'))
    if sig.audio is not None:
        evs = sig.audio.events_in(a, b)
        if evs:
            w = sum(AUDIO_W.get(e['kind'], 0.4) * min(1.5, 0.6 + 0.5 * e.get('strength', 0.5)) for e in evs)
            pts = min(16.0, w * 9)
            s += pts
            kinds = sorted({EVENT_RU.get(e['kind'], e['kind']) for e in evs})
            reasons.append((round(pts), 'звук: ' + ', '.join(kinds)))
    return min(25.0, s)


def _visual(sig, a, b, reasons):
    s = 4.0
    evs = []
    if sig.screen is not None:
        evs += [e for e in sig.screen.events if a <= e['t'] < b]
    span = sig.motion.span(a, b) if sig.motion is not None else None
    if span:
        evs += [e for e in span['screen_events'] if all(abs(e['t'] - x['t']) > 0.5 for x in evs)]
        pts = max(0.0, min(9.0, (span['top'] - 1.0) * 6 + span['bursts'] * 2))
        if pts >= 2:
            s += pts
            reasons.append((round(pts), 'много движения в кадре'))
        if span['static'] > 0.8:
            s -= 3
            reasons.append((-3, 'статичная картинка'))
    if evs:
        w = sum(SCREEN_W.get(e['kind'], 0.3) for e in evs)
        pts = min(10.0, w * 7)
        s += pts
        reasons.append((round(pts), 'на экране: ' + ', '.join(sorted({EVENT_RU.get(e['kind'], e['kind']) for e in evs}))))
    return max(0.0, min(20.0, s))


def _sentence_start(sig, t, tol=0.25):
    return any(abs(s['start'] - t) <= tol for s in sig.sents)


def _coherence(sig, v, clip, payoff, reasons):
    s = 8.0
    if v['kind'] == 'teaser':
        a, b = v['teaser']
        whole = not any(w['start'] < a < w['end'] - 0.05 or w['start'] + 0.05 < b < w['end'] for w in sig.words)
        if whole:
            s += 8
            reasons.append((8, 'тизер не режет слова'))
        s += 4
        reasons.append((4, 'клип остаётся целиком — смысл не теряется'))
        return min(20.0, s)
    if _sentence_start(sig, v['start']):
        s += 8
        reasons.append((8, 'начинается с начала фразы'))
    else:
        s -= 6
        reasons.append((-6, 'начинается посреди фразы'))
    if payoff is not None and v['start'] <= payoff - 0.5:
        s += 4
    lost = v['start'] - clip[0]
    if lost > 0.35 * (clip[1] - clip[0]):
        s -= 5
        reasons.append((-5, f'обрезано {lost:.0f} с начала'))
    return max(0.0, min(20.0, s))


def score_window(sig, v, clip, payoff, ref):
    a, b = v['window']
    reasons = []
    speech, text = _speech(sig, a, b, v['kind'], reasons)
    emo = _emotion(sig, a, b, ref, reasons)
    vis = _visual(sig, a, b, reasons)
    coh = _coherence(sig, v, clip, payoff, reasons)
    parts = {'speech': round(speech, 1), 'emotion': round(emo, 1), 'visual': round(vis, 1), 'coherence': round(coh, 1)}
    v.update(score=round(min(100.0, speech + emo + vis + coh), 1), parts=parts, text=text[:160],
             reasons=[{'points': int(p), 'text': t} for p, t in sorted(reasons, key=lambda r: -abs(r[0]))])
    return v


# ------------------------------------------------------------------ variants
def _payoff(sig, a, b):
    """The clip's strongest moment (what it is about): fused text/audio/screen events."""
    sents = [s for s in sig.sents if s['end'] > a and s['start'] < b]
    cues = [mo.sentence_cues(s['text']) for s in sents]
    evs = [e for e in mo.collect_events(sents, cues, sig.audio, sig.screen) if a <= e['t'] < b]
    return max(evs, key=lambda e: e['w']) if evs else None, evs


def _teasers(sig, clip, evs, n=2):
    a0, b0 = clip
    out = []
    for e in sorted(evs, key=lambda e: -e['w']):
        if e['w'] < 0.6 or e['t'] < a0 + WINDOW + 1.0:
            continue                     # weak, or already in the opening anyway
        t = e['t']
        start = mo._word_gap_before(sig.words, t - 0.3, t - 1.4, t) if sig.words else None
        start = t - 0.3 if start is None else start
        end = mo._safe_end(sig.words, min(start + 2.6, b0), b0) if sig.words else start + 2.0
        end = min(end, start + 3.0, b0)
        if end - start < 1.0:
            continue
        if any(abs(start - x[0]) < 2.0 for x in out):
            continue
        out.append((max(a0, start), end))
        if len(out) >= n:
            break
    return out


def analyze(segments, clip_start, clip_end, clip_min=15, clip_max=60, duration=None, audio=None, screen=None,
            motion=None, max_variants=5):
    """-> {'version', 'variants': [...best first], 'recommended', 'selected': None, 'original'}."""
    sig = segments if isinstance(segments, Signals) else Signals(segments, audio, screen, motion)
    a0, b0 = float(clip_start), float(clip_end)
    L = b0 - a0
    duration = duration or b0
    peak, evs = _payoff(sig, a0, b0)
    payoff = peak['t'] if peak else None
    ref = sig.arousal(a0, b0)
    variants = [{'id': 'orig', 'kind': 'original', 'start': a0, 'end': b0, 'teaser': None,
                 'window': [a0, min(b0, a0 + WINDOW)]}]
    lo, hi = max(0.0, a0 - EARLIER_S), a0 + min(TRIM_SHARE * L, max(0.0, L - clip_min))
    for s in sig.sents:
        t = s['start']
        if not (lo <= t <= hi) or abs(t - a0) < 0.3:
            continue
        if b0 - t > clip_max or b0 - t < min(clip_min, L):
            continue
        if payoff is not None and t > payoff - 0.5:
            continue                     # never cut away the moment the clip is about
        variants.append({'id': f'start-{int(round(t * 1000))}', 'kind': 'start', 'start': _round(t), 'end': b0,
                         'teaser': None, 'window': [t, min(b0, t + WINDOW)]})
    for ta, tb in _teasers(sig, (a0, b0), evs):
        variants.append({'id': f'teaser-{int(round(ta * 1000))}', 'kind': 'teaser', 'start': a0, 'end': b0,
                         'teaser': [_round(ta), _round(tb)], 'window': [ta, tb]})
    for v in variants:
        score_window(sig, v, (a0, b0), payoff, ref)
        v['label'] = KIND_RU[v['kind']]
        v['window'] = [_round(x) for x in v['window']]
    orig = variants[0]
    rest = sorted(variants[1:], key=lambda v: -v['score'])[:max_variants - 1]
    best = max([orig] + rest, key=lambda v: v['score'])
    rec = best['id'] if best is orig or best['score'] - orig['score'] >= MARGIN else 'orig'
    ordered = sorted([orig] + rest, key=lambda v: -v['score'])
    return {'version': VERSION, 'variants': ordered, 'recommended': rec, 'selected': None,
            'original': orig['score'], 'payoff': None if payoff is None else _round(payoff),
            'clip': [_round(a0), _round(b0)]}


def variant(hooks, vid):
    return next((v for v in (hooks or {}).get('variants', []) if v['id'] == vid), None)


# ------------------------------------------------------------------ signals from the pipeline's caches
class _Motions:
    """Several MotionProfiles queried as one (the ranker's candidate probe + a small probe of the rest)."""

    def __init__(self, *profiles):
        self.profiles = [m for m in profiles if m is not None]

    def span(self, a, b):
        for m in self.profiles:
            if m.covers(a, b):
                return m.span(a, b)
        return None


MOTION_BUDGET_S = 30.0       # extra decoding for hook windows the ranker's motion probe did not cover


def load_signals(p, segments=None, source=None, vinfo=None, windows=(), cancel=None, motion=None):
    """Signals for project p from its analysis cache (audio + sound classes + screen). Motion: the ranker's
    profile (`motion`) is reused; only uncovered `windows` are probed, within MOTION_BUDGET_S seconds."""
    from pathlib import Path
    from . import analysis_cache as ac, audio_analysis, motion_analysis, sound_events
    audio = screen = motion = None
    cache = ac.cache_dir(p.source_fingerprint, create=False) if p.source_fingerprint else None
    if cache and Path(cache).is_dir():
        try:
            audio = audio_analysis.from_arrays(ac.load_npz(Path(cache) / 'audio.npz'))
            snd = sound_events.from_arrays(ac.load_npz(Path(cache) / 'sound.npz'))
            if audio is not None and snd is not None:
                audio.attach_sound(snd)
            screen = motion_analysis.screen_from_arrays(ac.load_npz(Path(cache) / 'screen.npz'))
        except Exception as exc:
            log.warning('hook signals from cache failed: %s', exc)
    known = motion
    todo = [w for w in windows if known is None or not known.covers(*w)]
    extra = None
    if source is not None and vinfo and todo:
        try:
            extra = motion_analysis.analyze_spans(source, todo, vinfo['width'], vinfo['height'], cancel=cancel,
                                                  max_total=MOTION_BUDGET_S)
        except Exception as exc:
            from .procutil import JobCancelled
            if isinstance(exc, JobCancelled):
                raise
            log.info('hook motion probe skipped: %s', exc)
    motion = _Motions(known, extra) if (known is not None or extra is not None) else None
    return Signals(segments if segments is not None else p.transcript, audio, screen, motion)


def candidate_windows(segments, clips, clip_min, clip_max):
    """Rough hook windows of a set of clips (for one shared motion probe before scoring)."""
    sig = Signals(segments)
    out = []
    for a0, b0 in clips:
        out.append((a0, a0 + WINDOW))
        for s in sig.sents:
            if max(0.0, a0 - EARLIER_S) <= s['start'] <= a0 + TRIM_SHARE * (b0 - a0):
                out.append((s['start'], s['start'] + WINDOW))
    return out


# ------------------------------------------------------------------ applying a teaser through Timeline IR
def prepend_teaser(tl, teaser, clip_start, src_words=None):
    """A new Timeline: the teaser [a, b] (source seconds, inside the clip) plays first, then the whole clip.
    Picture uses the layout of the scene the teaser comes from, the sound and captions of that moment come along."""
    from .timeline import model as m
    from .timeline.ids import stable_id
    from .timeline.validate import ensure_valid
    a, b = float(teaser[0]), float(teaser[1])
    D = round(b - a, 3)
    rel = a - float(clip_start)                     # teaser start on the clip's own timeline
    out = copy.deepcopy(tl)
    src_item = next((it for t in out.tracks if t.type == 'video' for it in t.items
                     if it.type == 'video' and it.src_in <= a + 1e-3 < it.src_in + it.duration), None)
    if src_item is None:                            # e.g. that moment was cut by Smart Editing: nearest shot
        vids = [it for t in out.tracks if t.type == 'video' for it in t.items if it.type == 'video']
        if not vids:
            raise ValueError('teaser is outside the clip picture')
        src_item = min(vids, key=lambda it: min(abs(it.src_in - a), abs(it.src_in + it.duration - a)))
    for sc in out.scenes:
        sc.start = round(sc.start + D, 3)
    for t in out.tracks:
        for it in t.items:
            it.start = round(it.start + D, 3)
            if it.type == 'caption':
                for w in it.words:
                    w.t0, w.t1 = round(w.t0 + D, 3), round(w.t1 + D, 3)
    sid = stable_id('sc', 'hook', tl.id, a)
    out.scenes.insert(0, m.Scene(id=sid, start=0.0, duration=D, name='hook', status='generated',
                                 generation=m.Generation(provider='hook_engine', params={'teaser': [a, b]})))
    vtrack = next(t for t in out.tracks if t.type == 'video')
    vtrack.items.insert(0, m.VideoItem(id=stable_id('it', 'hook', tl.id, 'v', a), start=0.0, duration=D, scene=sid,
                                       asset=src_item.asset, src_in=round(a, 3), volume=0.0,
                                       layout=copy.deepcopy(src_item.layout)))
    for t in out.tracks:
        if t.type == 'audio':
            first = next((it for it in t.items if it.type == 'audio'), None)
            if first is not None:
                t.items.insert(0, m.AudioItem(id=stable_id('it', 'hook', tl.id, 'a', a, t.id), start=0.0, duration=D,
                                              asset=first.asset, role=first.role, src_in=round(a, 3),
                                              volume=first.volume, ext=copy.deepcopy(first.ext)))
        if t.type == 'caption':
            cap = next((it for it in t.items if it.type == 'caption'), None)
            if cap is not None:
                if src_words is not None:           # source-time words of the teaser (edited clips)
                    words = [m.Word(t0=round(max(0.0, w['start'] - a), 3), t1=round(min(D, w['end'] - a), 3),
                                    text=w['text']) for w in src_words if w['start'] >= a - 0.05 and w['end'] <= b + 0.05]
                else:
                    words = [m.Word(t0=round(max(0.0, w.t0 - D - rel), 3), t1=round(min(D, w.t1 - D - rel), 3),
                                    text=w.text, emphasis=w.emphasis, speaker=w.speaker)
                             for w in cap.words if w.t0 - D >= rel - 0.05 and w.t1 - D <= rel + D + 0.05]
                words = [w for w in words if w.t1 > w.t0]
                if words:
                    t.items.insert(0, m.CaptionItem(id=stable_id('it', 'hook', tl.id, 'c', a), start=0.0, duration=D,
                                                    words=words, style=cap.style))
    out.duration = round(out.duration + D, 3)
    out.meta = dict(out.meta, hook={'kind': 'teaser', 'teaser': [a, b], 'seconds': D})
    return ensure_valid(out)
