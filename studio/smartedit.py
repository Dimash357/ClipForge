"""Smart Editing (2B): tighter, livelier Shorts without boring stretches - and without losing meaning.

plan(): decides what to cut inside a clip (source seconds), from the transcript words and the cached audio /
sound-classifier / screen analysis (via hooks.Signals):
* long pauses and silence are shortened to a natural gap - never to zero, never inside a word;
* emotional pauses are kept longer: before a punchline / reaction, after a question, around laughter;
* pauses that are not silent (laughter, screams, explosions, loud reactions) are not pauses: never cut;
* stutters and restarts ("я я я", "это было это было"), isolated hesitations ("э", "эм") - Dynamic and up;
* low-interest sentences (no cue, no emotion, no event, away from the clip's main moment) - Aggressive only;
* the result never gets shorter than the project's minimum; the main moment (payoff) and its setup stay.
Camera: jump cuts are hidden by alternating a small punch-in; reactions get one smooth push-in. No effect is used
on webcam layouts (it would crop the face panel). Modes: natural / dynamic / aggressive.

to_timeline(): applies a plan to the clip's Timeline IR - video pieces with their scene's layout (crop paths
re-timed), audio pieces with one shared loudness measurement and 12 ms de-click fades, captions re-timed word by
word - so picture, sound and subtitles stay in sync. Rendering is done by the Timeline Renderer (1B).
"""
import copy
import re

from . import highlights as hl
from . import moments as mo

VERSION = 2
MODES = {
    # Pauses: a gap longer than `max` is shortened to `keep` (emotional pauses keep `emo`); a speechless stretch
    # longer than `dead_air` is dead air (shortened to `dead_keep`). The clip never gets shorter than
    # floor x the project's minimum length. Zoom: `zoom` = alternating punch-in that hides jump cuts; punch-ins on
    # peaks reach 1 + punch_min .. 1 + punch (max 1.3), at least `cooldown` s apart, at most `rate` per 10 s.
    'natural': {'label': 'Natural', 'max': 1.1, 'keep': 0.45, 'emo': 0.9, 'dead_air': 3.0, 'dead_keep': 1.0,
                'floor': 0.85, 'repeats': False, 'fillers': False, 'boring': False, 'zoom': 0.0,
                'punch_min': 0.10, 'punch': 0.12, 'cooldown': 6.0, 'rate': 1.0, 'hold': 1.6, 'ease': 0.35,
                'note': 'только длинные паузы и тишина; редкие мягкие приближения'},
    'dynamic': {'label': 'Dynamic', 'max': 0.6, 'keep': 0.25, 'emo': 0.6, 'dead_air': 1.6, 'dead_keep': 0.6,
                'floor': 0.7, 'repeats': True, 'fillers': True, 'boring': False, 'zoom': 0.08,
                'punch_min': 0.12, 'punch': 0.22, 'cooldown': 3.5, 'rate': 2.0, 'hold': 1.4, 'ease': 0.22,
                'note': 'рекомендуется для игровых Shorts: паузы, тишина, повторы, запинки, зум на реакциях'},
    'aggressive': {'label': 'Aggressive', 'max': 0.4, 'keep': 0.15, 'emo': 0.45, 'dead_air': 1.0, 'dead_keep': 0.4,
                   'floor': 0.6, 'repeats': True, 'fillers': True, 'boring': True, 'zoom': 0.12,
                   'punch_min': 0.15, 'punch': 0.30, 'cooldown': 2.5, 'rate': 3.0, 'hold': 1.2, 'ease': 0.15,
                   'note': 'плюс малоинтересные фразы; максимум темпа и приближений'},
}
OVERRIDES = {   # user-tunable on top of a preset: key -> (min, max)
    'max': (0.25, 3.0), 'dead_air': (0.6, 8.0), 'floor': (0.3, 1.0), 'zoom_strength': (0.0, 1.5),
    'zoom_rate': (0.2, 4.0),
}
MAX_ZOOM = 1.3
CAM_LAYOUTS = {'game_cam', 'reaction_cam'}
REASON_RU = {'pause': 'пауза', 'silence': 'тишина', 'dead_air': 'пустой промежуток', 'repeat': 'повтор',
             'filler': 'запинка', 'boring': 'неинтересный фрагмент'}
SILENT = ('pause', 'silence', 'dead_air')
PROTECT = {'laughter', 'scream', 'cheer', 'explosion', 'gunshot', 'raised_voice', 'spike'}
KEEP_EVENTS = PROTECT | {'loud'}     # + loud non-speech sounds: game events (shots, hits, wins) are never cut
HESITATION = re.compile(r'^(э+|эм+|мм+|хм+|uh+|um+|erm+|hmm+)[.,…]*$', re.I)
PAD = 0.06                    # kept around every word at a cut
MIN_PIECE = 0.35              # shorter leftovers are merged into the cut (only if they hold no word)
NO_ZOOM_LAYOUTS = CAM_LAYOUTS     # no jump-cut zoom over a webcam panel (punch-ins there aim at the face)


def _norm(t):
    return re.sub(r'[^\w]+', '', t.lower().replace('ё', 'е'))


def _round(x):
    return round(float(x), 3)


def settings(mode, overrides=None):
    """Preset + validated user overrides (pause threshold, dead-air length, length floor, zoom strength / rate)."""
    cfg = dict(MODES[mode])
    cfg['zoom_strength'], cfg['zoom_rate'] = 1.0, cfg['rate']
    for k, v in (overrides or {}).items():
        if k in OVERRIDES:
            try:
                lo, hi = OVERRIDES[k]
                cfg[k] = max(lo, min(hi, float(v)))
            except (TypeError, ValueError):
                pass
    cfg['keep'] = min(cfg['keep'], cfg['max'] * 0.6)
    return cfg


def edit_args(edit=None, opts=None):
    """plan() keyword arguments from a clip's edit choice (meta['edit'] / API) over the project options."""
    e, o = edit or {}, opts or {}
    sw = lambda k, ok: bool(e[k]) if isinstance(e.get(k), bool) else bool(o.get(ok, True))   # noqa: E731
    ov = {k: v for k, v in (e.get('overrides') or {}).items() if k in OVERRIDES}
    return {'silence': sw('silence', 'smart_silence'), 'zoom': sw('zoom', 'smart_zoom'), 'overrides': ov}


def clean_edit(data):
    """Validated edit choice from the UI: switches and overrides (unknown keys dropped)."""
    out = {}
    for k in ('silence', 'zoom'):
        if k in (data or {}):
            out[k] = data[k] in (True, 1, '1', 'true', 'on')
    ov = {}
    for k, v in ((data or {}).get('overrides') or {}).items():
        if k in OVERRIDES:
            try:
                lo, hi = OVERRIDES[k]
                ov[k] = round(max(lo, min(hi, float(v))), 3)
            except (TypeError, ValueError):
                pass
    if ov:
        out['overrides'] = ov
    return out


# ------------------------------------------------------------------ audio evidence (AudioProfile, 50 ms frames)
def _has_levels(sig):
    p = sig.audio
    return p is not None and getattr(p, 'db', None) is not None and len(p.db) > 0


def _frames(sig, x, y):
    p = sig.audio
    i, j = p.idx(x), p.idx(y)
    return i, max(i, min(j, len(p.db)))


def _quiet_mask(sig, i, j):
    """Silence frames: far below the local level or near the floor, and not voiced."""
    import numpy as np
    from .audio_analysis import FLOOR_DB
    p = sig.audio
    quiet = (p.rel[i:j] < -10.0) | (p.db[i:j] <= FLOOR_DB + 8)
    voiced = getattr(p, 'voiced', None)
    if voiced is not None:
        quiet &= ~np.asarray(voiced[i:j], bool)
    return quiet


def quiet_share(sig, x, y):
    """Share of [x, y) that is silent (None without audio levels: no evidence either way)."""
    if not _has_levels(sig) or y - x < 0.05:
        return None
    i, j = _frames(sig, x, y)
    return float(_quiet_mask(sig, i, j).mean()) if j > i else None


def min_word(w):
    """A word is never shorter than this when its padded edges are trimmed (no syllable is ever cut)."""
    letters = len(re.sub(r'[^\w]', '', w['text']))
    return max(0.18, min(0.7, 0.08 + 0.07 * letters))


def speech_bounds(sig, w):
    """(start, end) of a word, tightened where Whisper stretched it over silence (common at pauses)."""
    s, e = w['start'], w['end']
    if not _has_levels(sig) or e - s <= min_word(w) + 0.25:
        return s, e
    i, j = _frames(sig, s, e)
    if j - i < 4:
        return s, e
    quiet = _quiet_mask(sig, i, j)
    hop = sig.audio.hop
    loud = [k for k, q in enumerate(quiet) if not q]
    if not loud:
        return s, e                                   # all quiet: no evidence where the word is -> untouched
    s2 = max(s, i * hop + loud[0] * hop - 0.05)
    e2 = min(e, i * hop + (loud[-1] + 1) * hop + 0.08)
    # trimmed parts must be really silent, and the word keeps its minimum length
    if e2 < e and quiet[int(round((e2 - s) / hop)):].mean() < 0.9:
        e2 = e
    if s2 > s and quiet[:max(1, int(round((s2 - s) / hop)))].mean() < 0.9:
        s2 = s
    if e2 - s2 < min_word(w):
        e2 = min(e, s2 + min_word(w))
    return s2, e2


def islands(sig, x, y, ref, payoff=None):
    """Parts of a speechless gap that must stay (with context): reactions, laughter, loud game events, screen
    events, high arousal, unrecognised voice (Whisper may have missed it - when unsure, keep). -> [(a, b, why)]"""
    out = []
    if sig.audio is not None:
        for e in sig.audio.events_in(x, y):
            if e['kind'] in KEEP_EVENTS or e.get('spike'):
                tail = 0.9 if e['kind'] in ('laughter', 'cheer', 'scream') else 0.6
                out.append((e['start'] - 0.5, e['end'] + tail, e['kind']))
    if sig.screen is not None:
        for e in sig.screen.events_in(x, y) if hasattr(sig.screen, 'events_in') else \
                [e for e in sig.screen.events if x <= e['t'] < y]:
            out.append((e['t'] - 0.5, e['t'] + 1.0, 'screen'))
    mot = getattr(sig, 'motion', None)
    for prof in getattr(mot, 'profiles', None) or ([mot] if mot is not None else []):
        if not len(getattr(prof, 't', [])):
            continue
        sel = (prof.t >= x) & (prof.t < y)
        for t, v, cut in zip(prof.t[sel], prof.motion[sel], prof.cuts[sel]):
            if cut or v >= 2.5 * prof.ref:
                out.append((float(t) - 0.4, float(t) + 0.8, 'motion'))
    if sig.audio is not None and ref is not None and len(getattr(sig.audio, 'arousal', [])):
        p = sig.audio
        i, j = p.idx(x), min(p.idx(y), len(p.arousal))
        hot = p.arousal[i:j] > ref + 0.25
        for a, b in _runs(hot):
            if b - a >= 4:                                # 0.2 s of clearly higher energy
                out.append(((i + a) * p.hop - 0.4, (i + b) * p.hop + 0.6, 'arousal'))
    if _has_levels(sig) and getattr(sig.audio, 'voiced', None) is not None:
        p = sig.audio
        i, j = _frames(sig, x, y)
        vo = p.voiced[i:j] & (p.rel[i:j] > 4.0)
        for a, b in _runs(vo):
            if b - a >= 5:                                # 0.25 s of a loud voice no word covers
                out.append(((i + a) * p.hop - 0.3, (i + b) * p.hop + 0.4, 'voice'))
    if _has_levels(sig) and y - x >= 0.3:
        import numpy as np
        from .audio_analysis import FLOOR_DB
        p = sig.audio
        i, j = _frames(sig, x, y)
        lv = p.db[i:j]
        if len(lv):
            # a sharp sound over the gap's own background (a hit, a shot, a crash) - even one the event detector
            # did not classify - is kept: when unsure, keep
            burst = (lv > float(np.median(lv)) + 8.0) & (lv > FLOOR_DB + 20)
            for a, b in _runs(burst):
                if b - a >= 3:
                    out.append(((i + a) * p.hop - 0.4, (i + b) * p.hop + 0.6, 'sound'))
    if payoff is not None and x <= payoff < y:
        out.append((payoff - 0.8, payoff + 1.5, 'payoff'))
    return [(max(x, a), min(y, b), why) for a, b, why in out if min(y, b) > max(x, a)]


def _runs(mask):
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        elif not v and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _subtract(x, y, keeps):
    """[x, y) minus the keep ranges -> free ranges."""
    cur, out = x, []
    for a, b in sorted(keeps):
        if a > cur:
            out.append((cur, min(a, y)))
        cur = max(cur, b)
        if cur >= y:
            break
    if cur < y:
        out.append((cur, y))
    return [(a, b) for a, b in out if b - a > 1e-3]


# ------------------------------------------------------------------ the plan
def plan(sig, clip_start, clip_end, mode='dynamic', clip_min=15, restore=(), payoff=None, silence=True, zoom=True,
         overrides=None):
    """-> {'version', 'mode', 'clip', 'cuts': [...], 'keep': [[a, b], ...], 'zooms': [...], 'push', 'stats'}.
    `restore`: cut ids the user wants back (manual control); silence / zoom: the two switches."""
    cfg = settings(mode, overrides)
    a0, b0 = float(clip_start), float(clip_end)
    words = [w for w in sig.words if w['start'] >= a0 - 0.05 and w['end'] <= b0 + 0.05]
    sents = [s for s in sig.sents if s['end'] > a0 and s['start'] < b0]
    if payoff is None:
        from .hooks import _payoff
        peak, _ = _payoff(sig, a0, b0)
        payoff = peak['t'] if peak else None
    ref = sig.arousal(a0, b0)
    cuts = []

    def add(x, y, reason, weight):
        x, y = max(a0, x), min(b0, y)
        if y - x >= 0.08:
            cuts.append({'start': _round(x), 'end': _round(y), 'reason': reason, 'w': weight})

    def emotional(prev_w, next_w, gap_end):
        if prev_w is not None and prev_w['text'].rstrip().endswith('?'):
            return True
        nxt = next((s for s in sents if s['start'] >= gap_end - 0.05), None)
        if nxt is not None and nxt['start'] - gap_end < 0.3 and (mo.sentence_cues(nxt['text']) or
                                                                 hl.EMO.search(nxt['text'].lower())):
            return True
        if sig.audio is not None and any(e['kind'] in PROTECT for e in sig.audio.events_in(gap_end, gap_end + 1.5)):
            return True
        return payoff is not None and 0 <= payoff - gap_end < 1.2

    # ---- pauses, silence and dead air (also before the first / after the last word)
    if silence:
        bounds = [speech_bounds(sig, w) for w in words]
        edges = [(a0, bounds[0][0], None, words[0]) if words else (a0, b0, None, None)]
        edges += [(bounds[k][1], bounds[k + 1][0], words[k], words[k + 1]) for k in range(len(words) - 1)]
        if words:
            edges.append((bounds[-1][1], b0, words[-1], None))
        for x, y, pw, nw in edges:
            gap = y - x
            if gap <= cfg['max']:
                continue
            dead = gap >= cfg['dead_air']
            keep = cfg['emo'] if emotional(pw, nw, y) else (cfg['dead_keep'] if dead else cfg['keep'])
            # what stays: a natural beat next to the speech + every island (reaction, event) with its context
            lead = 0.0 if pw is None else (max(keep, 0.6) if nw is None else keep / 2)
            tail = 0.0 if nw is None else (min(keep, 0.25) if pw is None else keep / 2)
            stay = [(x, x + lead), (y - tail, y)] + [(a, b) for a, b, _ in islands(sig, x, y, ref, payoff)]
            for fa, fb in _subtract(x, y, stay):
                if fb - fa < max(0.2, cfg['max'] - keep):
                    continue                              # too little to be worth a jump cut
                q = quiet_share(sig, fa, fb)
                if q is not None and q >= 0.85:
                    reason = 'silence'
                elif dead:
                    reason = 'dead_air'                   # no speech, no event: game noise / nothing happening
                else:
                    reason = 'silence' if pw is None or nw is None else 'pause'
                add(fa, fb, reason, gap)
    # ---- stutters / restarts: drop the first take, keep the last
    if cfg['repeats']:
        toks = [_norm(w['text']) for w in words]
        i = 0
        while i < len(words):
            hit = 0
            for n in (4, 3, 2, 1):
                if i + 2 * n <= len(words) and toks[i:i + n] == toks[i + n:i + 2 * n] and all(toks[i:i + n]):
                    if n == 1 and len(toks[i]) > 6:
                        continue                         # "очень очень" style emphasis is kept
                    hit = n
                    break
            if hit:
                add(words[i]['start'] - PAD / 2, words[i + hit]['start'] - PAD, 'repeat', 3.0)
                i += hit
            else:
                i += 1
    # ---- isolated hesitations
    if cfg['fillers']:
        for k, w in enumerate(words):
            if not HESITATION.match(w['text'].strip()):
                continue
            before = w['start'] - (words[k - 1]['end'] if k else a0)
            after = (words[k + 1]['start'] if k + 1 < len(words) else b0) - w['end']
            if before >= 0.1 and after >= 0.1:
                add(w['start'] - min(before, 0.2) + 0.02, w['end'] + min(after, 0.2) - 0.02, 'filler', 2.0)
    # ---- low-interest sentences
    if cfg['boring'] and len(sents) > 2:
        for k, s in enumerate(sents[1:-1], start=1):
            if s['start'] < a0 or s['end'] > b0 or s['end'] - s['start'] < 1.2:
                continue
            if payoff is not None and payoff - 6.0 <= s['end'] and s['start'] <= payoff + 3.0:
                continue                                 # the setup and the main moment stay
            low = s['text'].lower()
            if (mo.sentence_cues(s['text']) or hl.EMO.search(low) or hl.HOOK.search(low) or '?' in s['text']
                    or islands(sig, s['start'], s['end'], ref)):
                continue
            if sig.screen is not None and any(s['start'] <= e['t'] < s['end'] for e in sig.screen.events):
                continue
            add(s['start'] - PAD, s['end'] + PAD, 'boring', 1.0 + (s['end'] - s['start']) / 10)
    # ---- ids, manual restore, length floor, merge
    for c in cuts:
        c['id'] = f'{c["reason"]}-{int(round(c["start"] * 1000))}'
    floor = max(3.0, min(b0 - a0, clip_min * cfg['floor']))
    chosen, removed = [], 0.0
    for c in sorted(cuts, key=lambda c: -c['w']):
        c['seconds'] = _round(c['end'] - c['start'])
        c['label'] = REASON_RU[c['reason']]
        if c['id'] in restore:
            c['applied'] = False
            continue
        extra = _new_seconds(chosen, c)
        room = (b0 - a0) - removed - floor
        if extra > room and c['reason'] in SILENT and room >= 0.3 and abs(extra - c['seconds']) < 0.01:
            mid, half = (c['start'] + c['end']) / 2, room / 2          # shorten the pause instead of keeping it all
            c['start'], c['end'] = _round(mid - half), _round(mid + half)
            c['seconds'], c['shortened'] = _round(c['end'] - c['start']), True
            extra = _new_seconds(chosen, c)
        if extra > room + 1e-6:
            c['applied'] = False
            c['skipped'] = 'клип стал бы короче минимума'
            continue
        c['applied'] = True
        chosen.append(c)
        removed += extra
    keep = _keep_ranges(a0, b0, chosen, words, floor)
    result = sum(b - a for a, b in keep)
    zooms = plan_zooms(sig, a0, b0, keep, sents, payoff, ref, cfg) if zoom else []
    by = {}
    for c in chosen:
        by[c['reason']] = round(by.get(c['reason'], 0) + c['seconds'], 2)
    out_cuts = sorted(cuts, key=lambda c: c['start'])
    for c in out_cuts:
        c.pop('w', None)
    return {'version': VERSION, 'mode': mode, 'clip': [_round(a0), _round(b0)], 'cuts': out_cuts,
            'keep': [[_round(a), _round(b)] for a, b in keep], 'zooms': zooms, 'push': [z['t'] for z in zooms],
            'payoff': None if payoff is None else _round(payoff),
            'switches': {'silence': bool(silence), 'zoom': bool(zoom)},
            'settings': {k: cfg[k] for k in ('max', 'dead_air', 'floor', 'zoom_strength', 'zoom_rate')},
            'stats': {'original': _round(b0 - a0), 'result': _round(result), 'removed': _round(b0 - a0 - result),
                      'by_reason': by, 'cuts': len(keep) - 1,
                      'pauses': sum(1 for c in chosen if c['reason'] in SILENT), 'zooms': len(zooms)}}


# ------------------------------------------------------------------ Smart Auto Zoom
ZOOM_RU = {'laughter': 'смех', 'scream': 'крик', 'cheer': 'ликование', 'explosion': 'взрыв', 'gunshot': 'выстрел',
           'raised_voice': 'громкая реакция', 'spike': 'резкий звук', 'payoff': 'главный момент',
           'loud': 'громкое событие', 'phrase': 'яркая фраза', 'screen': 'событие на экране', 'arousal': 'эмоциональный пик'}


def plan_zooms(sig, a0, b0, keep, sents, payoff, ref, cfg):
    """Punch-ins on peaks: reactions and events, the payoff, emotional / funny phrases, energy peaks, screen events.
    Best first, at least `cooldown` s apart on the edited clip's clock, at most `rate` per 10 s. -> [zoom]."""
    def kept(t):
        return any(a <= t < b - 0.3 for a, b in keep)
    cands = []
    if sig.audio is not None:
        for e in sig.audio.events_in(a0, b0):
            if e['kind'] in KEEP_EVENTS:
                st = float(e.get('strength', 0.7) or 0.7) * (0.8 if e['kind'] == 'loud' else 1.0)
                cands.append((e['start'], 0.55 + 0.35 * min(1.0, st) + (0.1 if e['kind'] in ('laughter', 'scream')
                                                                       else 0.0), e['kind'], e['end']))
    if payoff is not None:
        cands.append((payoff, 0.95, 'payoff', payoff + cfg['hold']))
    for s in sents:
        low = s['text'].lower()
        cues = mo.sentence_cues(s['text'])
        score = (0.25 * bool(cues) + 0.2 * bool(hl.EMO.search(low)) + 0.15 * ('!' in s['text']) +
                 0.1 * bool(hl.HOOK.search(low)))
        if score >= 0.2:
            cands.append((s['start'], 0.35 + score, 'phrase', s['end']))
    if sig.screen is not None:
        for e in sig.screen.events:
            if a0 <= e['t'] < b0:
                cands.append((e['t'], 0.45, 'screen', e['t'] + 1.0))
    if sig.audio is not None and ref is not None and len(getattr(sig.audio, 'arousal', [])):
        p = sig.audio
        i, j = p.idx(a0), min(p.idx(b0), len(p.arousal))
        for a, b in _runs(p.arousal[i:j] > ref + 0.35):
            if b - a >= 3:
                k = i + a + int(max(range(a, b), key=lambda q: p.arousal[i + q]) - a)
                cands.append((k * p.hop - 0.15, 0.4 + min(0.3, float(p.arousal[k]) - ref - 0.35), 'arousal',
                              (i + b) * p.hop))
    if _has_levels(sig) and getattr(sig.audio, 'voiced', None) is not None:
        import numpy as np
        from .audio_analysis import FLOOR_DB
        p = sig.audio
        i, j = _frames(sig, a0, b0)
        lv = p.db[i:j]
        if len(lv):
            hit = (lv > float(np.median(lv)) + 8.0) & (lv > FLOOR_DB + 20) & ~np.asarray(p.voiced[i:j], bool)
            for a, b in _runs(hit):                       # a sharp non-voice sound: a shot, a hit, a crash
                if b - a >= 3:
                    cands.append(((i + a) * p.hop, 0.55, 'loud', (i + b) * p.hop + 0.6))
    total = sum(b - a for a, b in keep)
    limit = max(1, int(round(total / 10.0 * cfg['zoom_rate'])))
    chosen = []
    for t, score, kind, end in sorted(cands, key=lambda c: -c[1]):
        t = max(a0, t)
        if not kept(t):
            continue
        o = source_to_output({'keep': keep}, t)
        if o is None or any(abs(o - z['out']) < cfg['cooldown'] for z in chosen):
            continue
        strength = cfg['zoom_strength']
        level = 1.0 + (cfg['punch_min'] + (cfg['punch'] - cfg['punch_min']) * min(1.0, score)) * strength
        if level < 1.03:
            continue
        hold = max(0.7, min(cfg['hold'] * 1.6, end - t)) if end and end > t + 0.3 else cfg['hold']
        chosen.append({'t': _round(t), 'out': _round(o), 'level': round(min(MAX_ZOOM, level), 3),
                       'hold': round(hold, 2), 'ease': cfg['ease'],
                       'kind': kind, 'label': ZOOM_RU.get(kind, kind), 'score': round(score, 2)})
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda z: z['t'])


def _new_seconds(chosen, c):
    """Seconds c removes beyond the cuts already chosen."""
    lo, hi, extra = c['start'], c['end'], 0.0
    pts = sorted((x['start'], x['end']) for x in chosen if x['end'] > lo and x['start'] < hi)
    cur = lo
    for a, b in pts:
        if a > cur:
            extra += a - cur
        cur = max(cur, b)
    return extra + max(0.0, hi - cur)


def _keep_ranges(a0, b0, cuts, words, floor=0.0):
    merged = []
    for c in sorted(cuts, key=lambda c: c['start']):
        if merged and c['start'] <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], c['end'])
        else:
            merged.append([c['start'], c['end']])
    keep, cur = [], a0
    for x, y in merged:
        if x > cur:
            keep.append([cur, x])
        cur = max(cur, y)
    if cur < b0:
        keep.append([cur, b0])
    has_word = lambda a, b: any(a <= (w['start'] + w['end']) / 2 < b for w in words)     # noqa: E731
    tidy = [k for k in keep if k[1] - k[0] >= MIN_PIECE or has_word(*k)]
    if sum(b - a for a, b in tidy) >= floor - 1e-6:      # slivers go only if the length floor still holds
        keep = tidy
    return keep or [[a0, b0]]


# ------------------------------------------------------------------ Timeline IR
def _shift_path(params, offset):
    p = dict(params)
    path = p.get('path')
    if not path:
        return p
    pts = [list(map(float, q)) for q in path]
    out = []
    for k, (t, x, y) in enumerate(pts):
        if t >= offset:
            if not out and k > 0:                        # value at the new start, interpolated
                t0, x0, y0 = pts[k - 1]
                f = (offset - t0) / max(1e-6, t - t0)
                out.append([0.0, round(x0 + f * (x - x0), 1), round(y0 + f * (y - y0), 1)])
            out.append([round(t - offset, 3), x, y])
    if not out:
        out = [[0.0, pts[-1][1], pts[-1][2]]]
    elif out[0][0] > 0:
        out.insert(0, [0.0, out[0][1], out[0][2]])
    p['path'] = out
    return p


def to_timeline(tl, edit, clip_start, mode='dynamic'):
    """Apply an edit plan (source-time keep ranges) to a clip timeline. -> new validated Timeline."""
    from .timeline import model as m
    from .timeline.ids import stable_id
    from .timeline.validate import ensure_valid
    cfg = MODES[mode]
    c0 = float(clip_start)
    keep = [(float(a), float(b)) for a, b in edit['keep']]
    out = copy.deepcopy(tl)
    vtrack = next(t for t in out.tracks if t.type == 'video')
    vids = [it for it in vtrack.items if it.type == 'video']
    fade = next((it.duration for it in vtrack.items if it.type == 'transition' and it.kind != 'cut'), 0.0)
    scene_of = {s.id: s for s in tl.scenes}

    def scene_range(it):                                 # the scene's own source range (without a fade tail)
        sc = scene_of.get(it.scene)
        if sc is None:
            return it.src_in, it.src_in + it.duration
        return c0 + sc.start, c0 + sc.start + sc.duration

    pieces = []                                          # (src_a, src_b, video item, out_start)
    t = 0.0
    for a, b in keep:
        for it in vids:
            sa, sb = scene_range(it)
            x, y = max(a, sa), min(b, sb)
            if y - x >= 0.04:
                pieces.append([x, y, it, t])
                t += y - x
    total = round(t, 3)
    # sound and captions follow exactly the same pieces as the picture (contiguous source pieces merged)
    spans = []
    for x, y, _, ts in pieces:
        if spans and abs(spans[-1][1] - x) < 1e-6:
            spans[-1][1] = y
        else:
            spans.append([x, y, None, ts])
    new_items, scenes, transitions = [], [], []
    zoom_on = False
    zooms = edit.get('zooms')
    if zooms is None:                                    # plan of version 1: reaction push-ins only
        zooms = [{'t': t, 'level': 1.0 + cfg['punch_min'], 'hold': cfg['hold'], 'ease': 0.45} for t in edit.get('push') or []]
    use_cuts_zoom = (edit.get('switches') or {}).get('zoom', True) and cfg['zoom']
    W, H = tl.canvas.width, tl.canvas.height
    by_scene = {}
    for k, (x, y, it, ts) in enumerate(pieces):
        by_scene.setdefault(it.scene, []).append(k)
    sid_map = {}
    for old_sid, ks in by_scene.items():
        sid = stable_id('sc', 'edit', tl.id, old_sid)
        sid_map[old_sid] = sid
        first, last = pieces[ks[0]], pieces[ks[-1]]
        sc = scene_of.get(old_sid)
        scenes.append(m.Scene(id=sid, start=round(first[3], 3), duration=round(last[3] + last[1] - last[0] - first[3], 3),
                              name=(sc.name if sc else 'edit'), status='edited',
                              generation=copy.deepcopy(sc.generation) if sc else None))
    prev = None
    applied_zooms = 0
    for k, (x, y, it, ts) in enumerate(pieces):
        dur = y - x
        lay = copy.deepcopy(it.layout)
        if lay is not None:
            lay.params = _shift_path(lay.params, x - it.src_in)
        cam = lay is not None and lay.type in NO_ZOOM_LAYOUTS
        jump = prev is not None and prev[2] is it and abs(prev[1] - x) > 0.05      # a cut inside one shot
        if jump and use_cuts_zoom and not cam:
            zoom_on = not zoom_on
        base = 1.0 + (cfg['zoom'] if zoom_on and use_cuts_zoom and not cam else 0.0)
        mine = [z for z in zooms if x <= z['t'] < y - 0.3]
        kf = _zoom_keyframes(m, base, mine, x, dur)
        anims = []
        if kf is not None:
            anims.append(m.Animation(property='zoom', keyframes=kf))
            applied_zooms += len(mine)
            fx, fy = focus(lay, W, H, tl.asset(it.asset))
            if abs(fx - 0.5) > 0.01:
                anims.append(m.Animation(property='pan_x', keyframes=[m.Keyframe(t=0.0, value=round(fx - 0.5, 3))]))
            if abs(fy - 0.5) > 0.01:
                anims.append(m.Animation(property='pan_y', keyframes=[m.Keyframe(t=0.0, value=round(fy - 0.5, 3))]))
        vi = m.VideoItem(id=stable_id('it', 'edit', tl.id, 'v', round(x, 3)), start=round(ts, 3),
                         duration=round(dur, 3), scene=sid_map[it.scene], asset=it.asset, src_in=round(x, 3),
                         volume=0.0, layout=lay, animations=anims)
        if prev is not None and fade and prev[2] is not it and abs(prev[1] - x) < 0.05:
            # a scene change of the original edit (not a cut): keep its cross-fade
            pv = new_items[-1]
            pv.duration = round(pv.duration + fade, 3)
            transitions.append(m.TransitionItem(id=stable_id('it', 'edit', tl.id, 'f', round(x, 3)),
                                                start=round(ts, 3), duration=fade, kind='fade', from_item=pv.id,
                                                to_item=vi.id))
        new_items.append(vi)
        prev = (x, y, it)
    vtrack.items = new_items + transitions
    out.scenes = scenes
    for tr in out.tracks:
        if tr.type == 'audio':
            src = [it for it in tr.items if it.type == 'audio']
            if not src:
                continue
            a = src[0]
            ext = copy.deepcopy(a.ext)
            if ext.get('clipforge.loudness', {}).get('target') is not None:
                ext['clipforge.loudness']['measure'] = [round(a.src_in, 3), round(a.src_in + a.duration, 3)]
            items = []
            for k, (x, y, it, ts) in enumerate(spans):
                items.append(m.AudioItem(id=stable_id('it', 'edit', tl.id, 'a', tr.id, round(x, 3)),
                                         start=round(ts, 3), duration=round(y - x, 3), asset=a.asset, role=a.role,
                                         src_in=round(x, 3), volume=a.volume, fade_in=0.012, fade_out=0.012,
                                         ext=ext))
            tr.items = items
        elif tr.type == 'caption':
            for it in tr.items:
                if it.type != 'caption':
                    continue
                words = []
                for w in it.words:
                    s0, s1 = c0 + w.t0, c0 + w.t1
                    for x, y, _, ts in spans:
                        if s0 >= x - 0.02 and s1 <= y + 0.02:
                            nw = copy.deepcopy(w)
                            nw.t0 = round(max(0.0, ts + max(0.0, s0 - x)), 3)
                            nw.t1 = round(min(total, ts + min(y - x, s1 - x)), 3)
                            if nw.t1 > nw.t0:
                                words.append(nw)
                            break
                it.start, it.duration, it.words = 0.0, total, words
            tr.items = [it for it in tr.items if it.type != 'caption' or it.words]
    out.duration = total
    out.meta = dict(out.meta, edit={'mode': mode, 'removed': edit['stats']['removed'], 'cuts': edit['stats']['cuts'],
                                    'pauses': edit['stats'].get('pauses', 0), 'zooms': applied_zooms,
                                    'original': edit['stats']['original'], 'result': edit['stats']['result']})
    return ensure_valid(out)


def focus(lay, W, H, asset=None):
    """Where a punch-in aims on the output canvas (fractions): the face in a webcam panel, the reframed subject
    (crop paths follow the face / action), else the centre."""
    if lay is None:
        return 0.5, 0.5
    p = lay.params or {}
    if lay.type == 'game_cam' and p.get('cam') and asset is not None and asset.width and asset.height:
        from .layouts import cam_geometry
        try:
            g = cam_geometry(W, H, asset.width, asset.height, p['cam'], p.get('fit', 'balanced'))
        except Exception:
            return 0.5, 0.5
        hc = g['Hc'] / H
        fy = float((p.get('face_rel') or (0.5, 0.5))[1])
        y = (1 - hc + hc * fy) if p.get('cam_position', 'bottom') == 'bottom' else hc * fy
        return 0.5, min(0.95, max(0.05, y))
    if lay.type == 'reaction_cam':
        from .layouts import REACTION_CAM_SHARE
        return 0.5, REACTION_CAM_SHARE * 0.42             # eyes on the panel's upper third, face just below
    if lay.type in ('talking', 'smart', 'center'):
        return 0.5, 0.42                                  # the crop follows the face; faces sit above the middle
    return 0.5, 0.5


def _zoom_keyframes(m, base, zs, x, dur):
    """One zoom animation for a piece: punch in (ease-out), hold, smooth return (ease-in-out); a cut that comes
    first is the return. Never above MAX_ZOOM; None when the piece has no zoom at all."""
    if not zs:
        return [m.Keyframe(t=0.0, value=round(base, 3))] if base > 1.0 else None
    kf = [m.Keyframe(t=0.0, value=round(base, 3))]
    for z in sorted(zs, key=lambda z: z['t']):
        peak = round(min(MAX_ZOOM, base + (z['level'] - 1.0)), 3)
        ease = float(z.get('ease') or 0.25)
        t0 = max(kf[-1].t + 0.05, z['t'] - x - 0.08)
        if t0 >= dur - 0.15:
            break
        t1 = min(dur, t0 + ease)
        if t0 > kf[-1].t + 1e-3:
            kf.append(m.Keyframe(t=round(t0, 3), value=kf[-1].value))
        kf.append(m.Keyframe(t=round(t1, 3), value=peak, easing='ease_out'))
        t2 = min(dur, t1 + float(z.get('hold') or 1.2))
        ret = 0.4
        if t2 + ret <= dur - 0.02:
            kf.append(m.Keyframe(t=round(t2, 3), value=peak))
            kf.append(m.Keyframe(t=round(t2 + ret, 3), value=round(base, 3), easing='ease_in_out'))
        else:
            break                                       # the jump cut at the piece end returns the framing
    out = []
    for k in kf:                                        # strictly increasing times
        if out and k.t <= out[-1].t + 1e-6:
            continue
        out.append(k)
    return out


def _audio_pieces(keep):
    t = 0.0
    out = []
    for a, b in keep:
        out.append((a, b, None, t))
        t += b - a
    return out


def source_to_output(edit, t):
    """Map a source time of the clip to the edited clip's time (None if that moment was cut)."""
    acc = 0.0
    for a, b in edit['keep']:
        if a - 0.02 <= t <= b + 0.02:
            return acc + max(0.0, t - a)
        acc += b - a
    return None
