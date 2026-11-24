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

VERSION = 1
MODES = {
    # pause longer than `max` is shortened to `keep`; emotional pauses keep `emo`; zoom = punch-in on jump cuts
    'natural': {'label': 'Natural', 'max': 1.1, 'keep': 0.45, 'emo': 0.9, 'repeats': False, 'fillers': False,
                'boring': False, 'zoom': 0.0, 'push': 0.05, 'note': 'только длинные паузы, без приближений'},
    'dynamic': {'label': 'Dynamic', 'max': 0.6, 'keep': 0.25, 'emo': 0.6, 'repeats': True, 'fillers': True,
                'boring': False, 'zoom': 0.08, 'push': 0.07, 'note': 'паузы, повторы, запинки; лёгкие приближения'},
    'aggressive': {'label': 'Aggressive', 'max': 0.4, 'keep': 0.15, 'emo': 0.45, 'repeats': True, 'fillers': True,
                   'boring': True, 'zoom': 0.12, 'push': 0.08,
                   'note': 'плюс малоинтересные фразы; максимум темпа'},
}
REASON_RU = {'pause': 'пауза', 'silence': 'тишина', 'repeat': 'повтор', 'filler': 'запинка', 'boring': 'неинтересный фрагмент'}
PROTECT = {'laughter', 'scream', 'cheer', 'explosion', 'gunshot', 'raised_voice', 'spike'}
HESITATION = re.compile(r'^(э+|эм+|мм+|хм+|uh+|um+|erm+|hmm+)[.,…]*$', re.I)
PAD = 0.06                    # kept around every word at a cut
MIN_PIECE = 0.35              # shorter leftovers are merged into the cut (only if they hold no word)
NO_ZOOM_LAYOUTS = {'game_cam', 'reaction_cam'}


def _norm(t):
    return re.sub(r'[^\w]+', '', t.lower().replace('ё', 'е'))


def _round(x):
    return round(float(x), 3)


def plan(sig, clip_start, clip_end, mode='dynamic', clip_min=15, restore=(), payoff=None):
    """-> {'version', 'mode', 'clip', 'cuts': [...], 'keep': [[a, b], ...], 'push': [t...], 'stats'}.
    `restore`: cut ids the user wants back (manual control)."""
    cfg = MODES[mode]
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

    def protected(x, y):
        if sig.audio is not None and any(e['kind'] in PROTECT for e in sig.audio.events_in(x, y)):
            return True
        ar = sig.arousal(x, y)
        return ar is not None and ref is not None and ar > ref + 0.15

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

    # ---- pauses and silence (also before the first / after the last word)
    edges = [(a0, words[0]['start'], None, words[0]) if words else (a0, b0, None, None)]
    edges += [(w['end'], n['start'], w, n) for w, n in zip(words, words[1:])]
    if words:
        edges.append((words[-1]['end'], b0, words[-1], None))
    for x, y, pw, nw in edges:
        gap = y - x
        if gap <= cfg['max'] or protected(x, y):
            continue
        keep = cfg['emo'] if emotional(pw, nw, y) else cfg['keep']
        if pw is None:                                   # leading silence: start shortly before speech
            add(x, y - min(keep, 0.25), 'silence', gap)
        elif nw is None:                                 # trailing: keep a beat for the reaction to land
            add(x + max(keep, 0.6), y, 'silence', gap)
        else:
            add(x + keep / 2, y - keep / 2, 'pause', gap)
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
                    or protected(s['start'], s['end'])):
                continue
            if sig.screen is not None and any(s['start'] <= e['t'] < s['end'] for e in sig.screen.events):
                continue
            add(s['start'] - PAD, s['end'] + PAD, 'boring', 1.0 + (s['end'] - s['start']) / 10)
    # ---- ids, manual restore, length floor, merge
    for c in cuts:
        c['id'] = f'{c["reason"]}-{int(round(c["start"] * 1000))}'
    floor = max(3.0, min(b0 - a0, clip_min * 0.85))
    chosen, removed = [], 0.0
    for c in sorted(cuts, key=lambda c: -c['w']):
        c['seconds'] = _round(c['end'] - c['start'])
        c['label'] = REASON_RU[c['reason']]
        if c['id'] in restore:
            c['applied'] = False
            continue
        extra = _new_seconds(chosen, c)
        if (b0 - a0) - removed - extra < floor:
            c['applied'] = False
            c['skipped'] = 'клип стал бы короче минимума'
            continue
        c['applied'] = True
        chosen.append(c)
        removed += extra
    keep = _keep_ranges(a0, b0, chosen, words, floor)
    result = sum(b - a for a, b in keep)
    push = []
    if cfg['push']:
        evs = [e['start'] for e in (sig.audio.events_in(a0, b0) if sig.audio is not None else [])
               if e['kind'] in PROTECT] + ([payoff] if payoff is not None else [])
        for t in sorted(set(round(t, 2) for t in evs)):
            if any(a <= t < b for a, b in keep) and all(abs(t - p) >= 5.0 for p in push):
                push.append(t)
    by = {}
    for c in chosen:
        by[c['reason']] = round(by.get(c['reason'], 0) + c['seconds'], 2)
    out_cuts = sorted(cuts, key=lambda c: c['start'])
    for c in out_cuts:
        c.pop('w', None)
    return {'version': VERSION, 'mode': mode, 'clip': [_round(a0), _round(b0)], 'cuts': out_cuts,
            'keep': [[_round(a), _round(b)] for a, b in keep], 'push': push[:4],
            'payoff': None if payoff is None else _round(payoff),
            'stats': {'original': _round(b0 - a0), 'result': _round(result), 'removed': _round(b0 - a0 - result),
                      'by_reason': by, 'cuts': len(keep) - 1}}


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
    pushes = list(edit.get('push') or [])
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
    for k, (x, y, it, ts) in enumerate(pieces):
        dur = y - x
        lay = copy.deepcopy(it.layout)
        if lay is not None:
            lay.params = _shift_path(lay.params, x - it.src_in)
        anims = []
        zoomable = cfg['zoom'] or cfg['push']
        if zoomable and (lay is None or lay.type not in NO_ZOOM_LAYOUTS):
            jump = prev is not None and prev[2] is it and abs(prev[1] - x) > 0.05      # a cut inside one shot
            if jump and cfg['zoom']:
                zoom_on = not zoom_on
            base = 1.0 + (cfg['zoom'] if zoom_on else 0.0)
            push = next((p for p in pushes if x <= p < y), None)
            if push is not None:
                tp = max(0.0, push - x - 0.15)
                kf = [m.Keyframe(t=round(tp, 3), value=base),
                      m.Keyframe(t=round(min(dur, tp + 0.45), 3), value=round(base + cfg['push'], 3),
                                 easing='ease_out')]
                if tp > 0:
                    kf.insert(0, m.Keyframe(t=0.0, value=base))
                anims.append(m.Animation(property='zoom', keyframes=kf))
            elif base > 1.0:
                anims.append(m.Animation(property='zoom', keyframes=[m.Keyframe(t=0.0, value=round(base, 3))]))
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
    out.meta = dict(out.meta, edit={'mode': mode, 'removed': edit['stats']['removed'], 'cuts': edit['stats']['cuts']})
    return ensure_valid(out)


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
