"""Stream moment detection: speech cues + event fusion + event-anchored clip windows.

A "moment" is a short time span where at least one signal fires:
  * speech cues (this module): exclamations ("ЧТО?!", "ДА ЛАДНО", "ТЫ ВИДЕЛ?!"), swearing, fails/deaths,
    donations, arguing with chat, absurd/bug situations, laughter written by Whisper;
  * audio events (audio_analysis): scream, laughter, raised voice, volume spike, loud;
  * screen events (motion_analysis): fail screens (black/grey/red), flashes, scene cuts.
Signals within a few seconds are fused; a moment confirmed by several modalities is worth more than a loud one.
Clip windows are then built *around* the moment: a short setup before it (so the cause is visible and the hook
lands in the first 1-2 s) and the reaction/resolution after it, snapped to word boundaries.
"""
import re

from . import textutil as tx


def _rx(*parts):
    return re.compile(r'(?<!\w)(?:' + '|'.join(parts) + r')(?!\w)', re.I | re.U)


CUES = {
    # surprise / disbelief exclamations
    'exclaim': _rx(r'что\?!', r'чего\?!', r'как\?!', r'да ладно\w*', r'ты видел\w*', r'вы видели', r'видели\?',
                   r'серь[её]зно', r'не может быть', r'какого\s+\w+', r'ни фига', r'ничего себе', r'офигеть',
                   r'обалдеть', r'вот это да', r'ну нет', r'нет нет нет\w*', r'стоп стоп', r'what\?!', r'no way',
                   r'are you kidding( me)?', r'did you see( that)?', r'you saw that', r'oh my god', r'omg',
                   r'what the hell', r'wtf', r'holy (shit|cow|moly)', r'oh no', r'no no no', r'are you serious',
                   r'бро', r'жесть', r'капец', r'пипец'),
    # swearing (incl. Whisper's starred-out words)
    'profanity': _rx(r'бля\w*', r'бляд\w*', r'хуй\w*', r'хуе\w*', r'хуё\w*', r'пизд\w*', r'ёб\w*', r'еба\w*',
                     r'ебл\w*', r'ебу\w*', r'заеб\w*', r'наеб\w*', r'уеб\w*', r'сук[аи]', r'сучк\w*', r'нахер',
                     r'нахрен', r'охуе\w*', r'fuck\w*', r'shit\w*', r'bitch\w*', r'damn', r'motherfuck\w*',
                     r'\w*\*{2,}\w*'),
    # fails, deaths, crashes, failed missions
    'fail': _rx(r'умер\w*', r'умира\w*', r'сдох\w*', r'убил[иа]?', r'убили меня', r'меня убили', r'слил\w*',
                r'проиграл\w*', r'провал\w*', r'разбил\w*', r'авари\w*', r'врезал\w*', r'упал\w*', r'потрачено',
                r'вайп\w*', r'опять умер', r'game over', r'гейм овер', r'died', r'dead', r'killed', r'i\'?m dead',
                r'crash\w*', r'failed', r'mission failed', r'wasted', r'we lost', r'i lost', r'oops', r'упс'),
    # donation / subscription reactions
    'donation': _rx(r'донат\w*', r'задонат\w*', r'донатер\w*', r'спасибо за (донат|подписк\w*|поддержк\w*)',
                    r'\d+\s*(руб\w*|рублей|доллар\w*|бакс\w*)', r'подписк\w*', r'сабк\w*', r'donat\w*',
                    r'thanks? (you )?for the (sub|bits|dono|donation)\w*', r'super ?chat', r'bits', r'subscri\w*'),
    # talking to / arguing with chat
    'chat': _rx(r'чат\w*', r'чатик\w*', r'чатлане', r'ребят\w*', r'вы (что|чё|че)', r'нет,? чат', r'chat',
                r'guys', r'you guys'),
    # absurd / broken game situations
    'absurd': _rx(r'что происходит', r'что это было', r'как это вообще', r'это баг\w*', r'баг\w*', r'глюк\w*',
                  r'лаг\w*', r'летает', r'застрял\w*', r'what is happening', r'what just happened', r'bug\w*',
                  r'glitch\w*', r'how is that possible', r'stuck', r'бред', r'абсурд\w*', r'дичь'),
    # laughter that Whisper wrote down
    'laugh_text': _rx(r'(ха){2,}\w*', r'(ах){2,}\w*', r'хех\w*', r'(ha){2,}\w*', r'lol', r'lmao', r'rofl',
                      r'\[смех\]', r'\(смеется\)', r'\(смеётся\)', r'\[laugh\w*\]', r'\(laugh\w*\)'),
}
CUE_WEIGHT = {'exclaim': 14, 'profanity': 11, 'fail': 18, 'donation': 12, 'chat': 8, 'absurd': 13, 'laugh_text': 14}
CUE_LABEL_RU = {'exclaim': 'эмоциональное восклицание', 'profanity': 'неожиданный мат', 'fail': 'фейл / смерть / провал',
                'donation': 'реакция на донат', 'chat': 'разговор/спор с чатом', 'absurd': 'странная ситуация в игре',
                'laugh_text': 'смех в речи'}
CUE_LABEL_EN = {'exclaim': 'emotional exclamation', 'profanity': 'unexpected swearing', 'fail': 'fail / death',
                'donation': 'donation reaction', 'chat': 'talking to chat', 'absurd': 'absurd game situation',
                'laugh_text': 'laughter in speech'}


def sentence_cues(text):
    """{category: count} of stream-reaction cues in one sentence."""
    raw = text.lower()
    low = raw.replace('ё', 'е')
    out = {}
    for k, rx in CUES.items():
        n = max(len(rx.findall(raw)), len(rx.findall(low)))
        if n:
            out[k] = min(3, n)
    words = text.split()
    stripped = text.rstrip('"»”) ')
    if ('?!' in text or '!?' in text or '!!' in text or
            (len(words) <= 4 and stripped.endswith('!'))):
        out['exclaim'] = max(out.get('exclaim', 0), 1)
    caps = [w for w in words if len(w) >= 3 and w.isupper() and any(c.isalpha() for c in w)]
    if len(caps) >= 2:
        out['exclaim'] = max(out.get('exclaim', 0), 1)
    return out


def reaction_score(cue_counts):
    """0..100 'the streamer is reacting' score from summed cue counts of a span."""
    s = 30.0
    for k, n in cue_counts.items():
        s += CUE_WEIGHT.get(k, 8) * min(2.5, n ** 0.8)
    if len(cue_counts) >= 2:
        s += 6
    return max(0.0, min(100.0, s))


# ------------------------------------------------------------------ event fusion
def collect_events(sents, cues, audio=None, screen=None):
    """-> sorted list of {'t', 'end', 'src', 'kind', 'w'} from all modalities."""
    ev = []
    for s, c in zip(sents, cues):
        if not c:
            continue
        kind = max(c, key=lambda k: CUE_WEIGHT.get(k, 0) * c[k])
        if kind == 'chat' and len(c) == 1:
            w = 0.35                                  # just addressing chat is weak on its own
        else:
            w = min(1.6, sum(CUE_WEIGHT.get(k, 8) * n for k, n in c.items()) / 18.0)
        ev.append({'t': s['start'], 'end': s['end'], 'src': 'text', 'kind': kind, 'w': round(w, 2)})
    if audio is not None:
        for e in audio.events:
            base = {'scream': 1.2, 'laughter': 1.2, 'raised_voice': 0.7, 'spike': 0.6, 'loud': 0.45}.get(e['kind'], 0.5)
            ev.append({'t': e['start'], 'end': e['end'], 'src': 'audio', 'kind': e['kind'],
                       'w': round(base * min(1.5, 0.6 + 0.5 * e['strength']), 2)})
    if screen is not None:
        for e in screen.events:
            w = {'fail_screen': 1.0, 'flash': 0.5, 'cut': 0.25}.get(e['kind'], 0.3)
            ev.append({'t': e['t'], 'end': e['t'] + 1.0, 'src': 'screen', 'kind': e['kind'], 'w': w})
    ev.sort(key=lambda e: e['t'])
    return ev


def fuse(events, window=3.5):
    """Cluster events that happen together. A cluster confirmed by several modalities gets a bonus;
    a cluster made only of plain loudness ('loud'/'spike'/'cut') is down-weighted (volume alone is not a moment)."""
    clusters = []
    for e in events:
        if clusters and e['t'] - clusters[-1]['end'] <= window:
            c = clusters[-1]
            c['end'] = max(c['end'], e['end'])
            c['items'].append(e)
        else:
            clusters.append({'t': e['t'], 'end': e['end'], 'items': [e]})
    for c in clusters:
        srcs = {e['src'] for e in c['items']}
        kinds = {e['kind'] for e in c['items']}
        w = sum(e['w'] for e in c['items'])
        if len(srcs) >= 2:
            w *= 1.0 + 0.35 * (len(srcs) - 1)
        if kinds <= {'loud', 'spike', 'cut', 'flash'}:
            w *= 0.5
        peak = max(c['items'], key=lambda e: e['w'])
        c.update(srcs=sorted(srcs), kinds=sorted(kinds), w=round(w, 2), peak=peak['t'])
    return clusters


# ------------------------------------------------------------------ event-anchored windows
def _word_gap_before(words, t, lo, hi):
    """Latest moment in [lo, hi] that is not inside a word (prefers real pauses). None if words cover it all."""
    best = None
    for k in range(len(words) - 1):
        a, b = words[k]['end'], words[k + 1]['start']
        if b < lo or a > hi:
            continue
        g = min(b, hi) - 0.02
        if g >= max(a, lo):
            score = (b - a) * 2 - abs(g - t) * 0.3
            if best is None or score > best[0]:
                best = (score, g)
    if words and words[0]['start'] > lo:
        g = min(words[0]['start'] - 0.02, hi)
        if g >= lo:
            score = 5 - abs(g - t) * 0.3
            if best is None or score > best[0]:
                best = (score, g)
    return best[1] if best else None


def _safe_end(words, t, duration):
    """Move t forward to the end of the word it falls into (never stop mid-word), +0.3 s tail before the next."""
    for k, w in enumerate(words):
        if w['start'] < t < w['end']:
            t = w['end']
        if w['start'] >= t:
            return min(t + 0.3, w['start'] - 0.02, duration)
    return min(t + 0.3, duration)


def anchored_windows(cluster, sents, words, duration, minlen, maxlen, setup=2.5):
    """Clip window around a fused moment: [setup before it ... reaction/resolution after it]."""
    t0 = cluster['t']
    lo, hi = max(0.0, t0 - 8.0), max(0.0, t0 - 0.8)
    # prefer the start of a sentence that begins shortly before the moment
    starts = [s['start'] for s in sents if lo <= s['start'] <= hi]
    target = t0 - setup
    if starts:
        start = min(starts, key=lambda x: (abs(x - target) + (0 if x <= t0 - 1.5 else 2.0)))
        start = max(0.0, start - 0.2)
        prev_end = max((w['end'] for w in words if w['end'] <= start + 0.2 + 1e-6), default=0.0)
        start = max(start, prev_end + 0.02 if prev_end else 0.0)
    else:
        g = _word_gap_before(words, target, max(0.0, t0 - 5.0), hi)
        start = g if g is not None else max(0.0, t0 - 2.0)
    start = round(max(0.0, min(start, t0)), 2)
    # end: after the reaction (cluster end) and the climax, then close the current sentence
    want = max(cluster['end'] + 1.5, cluster['peak'] + 2.0, start + minlen)
    containing = next((s for s in sents if s['start'] < want < s['end'] + 0.25), None)
    if containing and containing['end'] - start <= maxlen:
        want = containing['end']
    if want - start > maxlen:
        # longest finished sentence that still keeps the climax
        ok = [s['end'] for s in sents if cluster['peak'] + 1.0 <= s['end'] <= start + maxlen - 0.3]
        want = max(ok) if ok else start + maxlen - 0.3
    end = round(_safe_end(words, want, duration), 2)
    if end - start > maxlen + 1.0:
        end = round(start + maxlen, 2)
    if end - start < min(minlen, 6.0):
        return None
    inside = [k for k, s in enumerate(sents) if s['end'] > start + 0.05 and s['start'] < end - 0.05]
    return {'start': start, 'end': end, 'sent_range': (inside[0], inside[-1]) if inside else None}


def first_signal(events, start, end):
    """Seconds from clip start to its first meaningful signal (None if nothing fires inside)."""
    ts = [e['t'] for e in events if start - 0.5 <= e['t'] < end and e['w'] >= 0.5]
    if not ts:
        return None
    return max(0.0, min(ts) - start)


def span_cues(cues, i, j):
    out = {}
    for c in cues[i:j + 1]:
        for k, n in c.items():
            out[k] = out.get(k, 0) + n
    return out


def explain_cues(counts, lang='ru'):
    lab = CUE_LABEL_RU if lang == 'ru' else CUE_LABEL_EN
    order = sorted(counts, key=lambda k: -CUE_WEIGHT.get(k, 0) * counts[k])
    return [lab[k] for k in order if k in lab]


def has_cues(text):
    return bool(sentence_cues(text))


def key_words(text):
    """Words worth highlighting in captions: reaction cues, emotion/hook words, numbers, all-caps shouting."""
    out = set()
    low = text.lower()
    for cat, rx in CUES.items():                  # whole cue phrases ("да ладно", "ты видел")
        for m in rx.finditer(low):
            for w in m.group(0).split():
                t = w.strip('.,!?…:;"«»()')
                # exclamations are coloured as a whole; elsewhere ("меня убили") only the meaningful words
                if t and (cat == 'exclaim' or t.replace('ё', 'е') not in tx.STOP):
                    out.add(t)
    for w in text.split():
        t = w.strip('.,!?…:;"«»()').lower()
        if not t:
            continue
        if sentence_cues(w) or any(ch.isdigit() for ch in t) or (len(w) >= 3 and w.isupper()):
            out.add(t)
    return out


__all__ = ['sentence_cues', 'reaction_score', 'collect_events', 'fuse', 'anchored_windows', 'first_signal',
           'span_cues', 'explain_cues', 'key_words', 'tx']
