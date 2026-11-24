"""AI Viral Director (2C): which Shorts are the most promising, and in which version.

For every clip it scores combinations of an opening (Hook Engine variant) and an edit (Smart Editing mode) from
analysis that already exists - the hook scores, the edit plans (computed, not rendered), the ranker's scores, the
cached audio / sound / screen events and the transcript - and picks the best one. Nothing is rendered to score.

Viral Score 0..100 = hook 25 + emotion 20 + dynamics 15 + coherence 15 + originality 10 + retention 15.
It is a heuristic estimate of how well a clip is built for Shorts (strong opening, reactions, pace, no dead air,
a clear payoff), NOT a prediction of views: platforms, topic, audience and timing are outside of it.

rank(): duplicates (overlapping time or the same words) are filtered - the stronger clip stays; the rest are
ordered by score; select() keeps the best N for automatic selection.
"""
import logging

from . import highlights as hl
from . import moments as mo
from . import smartedit
from . import textutil as tx

log = logging.getLogger('studio.viral')

VERSION = 1
DISCLAIMER = ('Viral Score — эвристическая оценка того, насколько клип собран под Shorts (сильное начало, реакции, '
              'темп, без «пустот», понятная кульминация). Это не прогноз просмотров: результат зависит от темы, '
              'аудитории, времени публикации и алгоритмов платформы.')
WEIGHTS = {'hook': 25, 'emotion': 20, 'dynamics': 15, 'coherence': 15, 'originality': 10, 'retention': 15}
EDIT_MODES = ('off', 'natural', 'dynamic', 'aggressive')
PREFER_SIMPLE = 2.0          # a fancier combination must win by this much
SWEET = (20.0, 45.0)         # seconds: the length range that usually holds viewers best
DUP_OVERLAP = 0.3            # share of the shorter clip that overlaps -> the same moment
DUP_TEXT = 0.55              # word-set similarity -> the same story


def _clamp(x, hi):
    return max(0.0, min(float(hi), float(x)))


def _words(text):
    return set(t for t in tx.tokens(text or '') if len(t) > 2)


def similarity(a, b):
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def overlap(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


# ------------------------------------------------------------------ one combination
def _events(sig, a, b):
    out = []
    if sig.audio is not None:
        out += [(e['start'], e['kind']) for e in sig.audio.events_in(a, b) if e['kind'] in smartedit.PROTECT]
    if sig.screen is not None:
        out += [(e['t'], e['kind']) for e in sig.screen.events if a <= e['t'] < b and e['kind'] != 'cut']
    for s in sig.sents:
        if a <= s['start'] < b and mo.sentence_cues(s['text']):
            out.append((s['start'], 'cue'))
    return sorted(out)


def evaluate(sig, clip, hook, edit, scores=None, others=(), clip_min=15, clip_max=60, payoff=None):
    """Score one combination. clip=(a, b) source seconds; hook=Hook variant dict (or None); edit=Smart Editing plan
    (or None = no cuts); others=[text of the other clips] for originality. -> dict(score, parts, reasons, tips)."""
    a0, b0 = clip
    if hook and hook['kind'] == 'start':
        a0, b0 = hook['start'], hook['end']
    keep = [tuple(k) for k in edit['keep']] if edit else [(a0, b0)]
    teaser = hook['teaser'] if hook and hook['kind'] == 'teaser' else None
    length = sum(b - a for a, b in keep) + (teaser[1] - teaser[0] if teaser else 0.0)
    reasons, tips = [], []
    sc = scores or {}

    # hook (25): the opening's own score
    hook_score = hook['score'] if hook else 45.0
    p_hook = _clamp(hook_score / 100 * 25, 25)
    reasons.append((round(p_hook), f'начало {hook_score:.0f}/100' + (f' ({hook["label"]})' if hook else '')))

    # emotion (20): reactions per minute of the result + the ranker's reaction/engagement view
    evs = [(t, k) for t, k in _events(sig, a0, b0) if any(a <= t < b for a, b in keep)]
    per_min = len(evs) / max(0.25, length / 60)
    p_emo = _clamp(per_min * 2.2 + (sc.get('reaction', 40) - 40) * 0.12 + (sc.get('engagement', 50) - 50) * 0.08, 20)
    reasons.append((round(p_emo), f'{len(evs)} реакц./событий ({per_min:.1f} в минуту)'))
    if per_min < 2:
        tips.append('Мало реакций: выберите момент с более яркой реакцией или укоротите клип вокруг неё.')

    # dynamics (15): speech density of the result + what is left of dead air
    words = [w for w in sig.words if any(a <= w['start'] and w['end'] <= b + 0.02 for a, b in keep)]
    wps = len(words) / max(1.0, length)
    dead = 0.0                                   # silence > 1 s that is still in the result (pieces join seamlessly)
    for a, b in keep:
        ws = [w for w in words if a <= w['start'] < b]
        for x, y in zip(ws, ws[1:]):
            if y['start'] - x['end'] > 1.0:
                dead += y['start'] - x['end'] - 1.0
    p_dyn = _clamp(4 + min(7.0, wps * 3.0) + min(4.0, len(keep) - 1) * 0.6 - dead * 0.8, 15)
    reasons.append((round(p_dyn), f'темп {wps:.1f} слова/с' + (f', пустоты {dead:.1f} с' if dead >= 0.5 else '')))
    if dead >= 2 and not edit:
        tips.append(f'В клипе {dead:.0f} с пауз: включите Smart Editing (Dynamic).')

    # coherence (15): the ranker's context / completeness; heavy cutting is a risk
    ctx = sc.get('context', 55)
    comp = sc.get('completeness', 55)
    cut_share = 1 - sum(b - a for a, b in keep) / max(1e-6, b0 - a0)
    p_coh = _clamp(15 * (0.5 * ctx + 0.5 * comp) / 100 - max(0.0, cut_share - 0.22) * 20, 15)
    if hook and hook['kind'] == 'start' and any(r['points'] < 0 and 'продолжение' in r['text']
                                                for r in hook.get('reasons', [])):
        p_coh -= 3
    reasons.append((round(p_coh), f'смысл и завершённость {ctx:.0f}/{comp:.0f}' +
                    (f', вырезано {cut_share * 100:.0f}%' if cut_share > 0.02 else '')))
    if comp < 45:
        tips.append('Клип обрывается: продлите конец до завершения мысли или реакции.')

    # originality (10): not the same words as the other clips, not repetitive inside
    text = ' '.join(w['text'] for w in words)
    sim = max([similarity(text, o) for o in others] or [0.0])
    rep = hl._repetition(text)
    p_orig = _clamp(10 * (1 - sim) * min(1.0, rep / 0.55), 10)
    reasons.append((round(p_orig), 'уникальный фрагмент' if sim < 0.25 else f'похож на другой клип ({sim * 100:.0f}%)'))

    # retention (15): length in the sweet spot, payoff in the first ~70%, events spread, strong first seconds
    p_ret = 5.0
    if SWEET[0] <= length <= SWEET[1]:
        p_ret += 4
    elif length > SWEET[1]:
        p_ret -= min(4.0, (length - SWEET[1]) / 5)
        tips.append(f'Клип длинноват ({length:.0f} с): Shorts лучше удерживают до {SWEET[1]:.0f} с.')
    payoff = (edit or {}).get('payoff') or payoff
    if payoff is not None and any(a - 0.05 <= payoff <= b + 0.05 for a, b in keep):
        rel = (smartedit.source_to_output({'keep': keep}, payoff) or 0) + (teaser[1] - teaser[0] if teaser else 0)
        share = rel / max(1e-6, length)
        if 0.1 <= share <= 0.75:
            p_ret += 3
        elif share > 0.85:
            p_ret -= 2
            tips.append('Кульминация в самом конце: зритель может не досмотреть — попробуйте тизер в начале.')
    if teaser:
        p_ret += 2
    times = sorted(smartedit.source_to_output({'keep': keep}, t) or 0 for t, _ in evs)
    gaps = [y - x for x, y in zip([0.0] + times, times + [length])]
    if gaps and max(gaps) > 12:
        p_ret -= 2
        tips.append(f'Провал без реакций {max(gaps):.0f} с — сократите середину (Smart Editing Aggressive).')
    p_ret += 2 if hook_score >= 60 else 0
    p_ret = _clamp(p_ret, 15)
    reasons.append((round(p_ret), f'длина {length:.0f} с, удержание'))

    parts = {'hook': round(p_hook, 1), 'emotion': round(p_emo, 1), 'dynamics': round(p_dyn, 1),
             'coherence': round(max(0.0, p_coh), 1), 'originality': round(p_orig, 1), 'retention': round(p_ret, 1)}
    if hook_score < 50:
        tips.insert(0, 'Слабое начало: выберите другой вариант Hook или тизер.')
    return {'score': round(min(100.0, sum(parts.values())), 1), 'parts': parts, 'length': round(length, 1),
            'reasons': [{'points': p, 'text': t} for p, t in reasons], 'tips': tips[:4]}


# ------------------------------------------------------------------ best combination of one clip
def direct(sig, clip, hooks_data, scores=None, others=(), clip_min=15, clip_max=60, modes=EDIT_MODES,
           restore=()):
    """Compare Hook variants x Smart Editing modes (plans only, nothing rendered). -> report dict."""
    from . import hooks as hk
    a0, b0 = clip
    variants = (hooks_data or {}).get('variants') or [None]
    variants = sorted(variants, key=lambda v: -(v or {}).get('score', 0))[:3]
    if hooks_data and hk.variant(hooks_data, 'orig') not in variants:
        variants.append(hk.variant(hooks_data, 'orig'))
    combos = []
    for v in variants:
        ca, cb = (v['start'], v['end']) if v and v['kind'] == 'start' else (a0, b0)
        for mode in modes:
            plan = None if mode == 'off' else smartedit.plan(sig, ca, cb, mode, clip_min, restore=restore)
            if plan is not None and plan['stats']['removed'] < 0.3:
                continue                                       # this mode changes nothing: same as off
            r = evaluate(sig, (a0, b0), v, plan, scores, others, clip_min, clip_max,
                         payoff=(hooks_data or {}).get('payoff'))
            complexity = (0 if not v or v['kind'] == 'original' else 1) + EDIT_MODES.index(mode) * 0.5
            combos.append(dict(r, hook=v['id'] if v else None, hook_kind=v['kind'] if v else None,
                               edit=mode, removed=plan['stats']['removed'] if plan else 0.0,
                               complexity=complexity))
    best = max(combos, key=lambda c: c['score'])
    simple = min((c for c in combos if c['score'] >= best['score'] - PREFER_SIMPLE), key=lambda c: c['complexity'])
    plain = next((c for c in combos if c['edit'] == 'off' and c['hook_kind'] in (None, 'original')), combos[0])
    ordered = sorted(combos, key=lambda c: -c['score'])[:6]
    return {'version': VERSION, 'score': simple['score'], 'parts': simple['parts'], 'reasons': simple['reasons'],
            'tips': simple['tips'], 'combo': {'hook': simple['hook'], 'hook_kind': simple['hook_kind'],
                                               'edit': simple['edit']},
            'gain': round(simple['score'] - plain['score'], 1), 'plain': plain['score'],
            'combos': [{k: c[k] for k in ('score', 'hook', 'hook_kind', 'edit', 'removed', 'length')} for c in ordered],
            'disclaimer': DISCLAIMER}


# ------------------------------------------------------------------ the project: duplicates, ranking, selection
def rank(items):
    """items: [{'id', 'range': (a, b), 'text', 'score'}] -> same items with 'rank' and 'duplicate_of' (or None),
    best first. A clip overlapping a stronger one in time, or telling the same words, is marked as its duplicate."""
    order = sorted(items, key=lambda x: -x['score'])
    kept = []
    for it in order:
        dup = next((k for k in kept if overlap(it['range'], k['range']) >= DUP_OVERLAP or
                    similarity(it['text'], k['text']) >= DUP_TEXT), None)
        it['duplicate_of'] = dup['id'] if dup else None
        if dup is None:
            kept.append(it)
    n = 0
    for it in order:
        if it['duplicate_of'] is None:
            n += 1
            it['rank'] = n
        else:
            it['rank'] = None
    return order


def select(ranked, count):
    """The ids to keep in automatic mode: the best `count` that are not duplicates."""
    return [it['id'] for it in ranked if it['duplicate_of'] is None][:count]
