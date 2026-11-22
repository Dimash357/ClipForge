"""Highlight Ranker v2: sentence-aligned candidates scored on Hook / Context / Engagement / Completeness.

Scores are heuristics for ranking *inside one video*; they are not a prediction of virality.
The legacy `pipeline.candidates()` stays as the fallback (see pipeline.pick_highlights).
"""
import logging
import re

from . import audio_analysis as audio_mod
from . import motion_analysis as motion_mod
from . import textutil as tx

log = logging.getLogger('studio.highlights')


def _rx(*parts):
    return re.compile(r'(?<!\w)(?:' + '|'.join(parts) + r')(?!\w)', re.I | re.U)


Q_START = _rx(r'why', r'how', r'what', r'who', r'when', r'where', r'did you', r'do you', r'have you', r'is it', r'can you',
              r'what if', r'почему', r'как', r'что', r'кто', r'когда', r'зачем', r'где', r'а вы', r'что если', r'представь\w*',
              r'знаешь\w*', r'знаете')
HOOK = _rx(r'secret\w*', r'truth', r'mistake\w*', r'never', r'always', r'biggest', r'worst', r'best', r'crazy', r'insane',
           r'shocking', r'nobody', r'imagine', r'unbelievable', r'wait', r'stop', r'look', r'listen', r'watch',
           r'секрет\w*', r'правд\w+', r'ошибк\w+', r'никогда', r'всегда', r'самы[йеая]\w*', r'безумн\w+', r'невероятн\w+',
           r'никто', r'представь\w*', r'смотри\w*', r'слушай\w*', r'подожди\w*', r'стой', r'главн\w+', r'шок\w*')
EMO = _rx(r'wow', r'omg', r'oh my god', r'holy', r'insane', r'crazy', r'incredible', r'amazing', r'no way', r'what the',
          r'damn', r'lol', r'haha+', r'hell', r'awesome', r'terrible', r'love', r'hate', r'scream\w*', r'laugh\w*',
          r'вау', r'ничего себе', r'офигет\w+', r'жесть', r'капец', r'невероятн\w+', r'ужас\w*', r'кошмар\w*', r'блин',
          r'хаха\w*', r'ахах\w*', r'прикол\w*', r'смешн\w+', r'круто', r'класс', r'обожаю', r'ненавижу', r'бешен\w+',
          r'пипец', r'сумасшедш\w+')
SURPRISE = _rx(r'but', r'however', r'suddenly', r'actually', r'turns out', r'wait', r'except', r'unexpectedly',
               r'but then', r'но', r'однако', r'вдруг', r'внезапно', r'оказывается', r'на самом деле', r'неожиданно',
               r'подожди\w*', r'стоп', r'а потом')
CONCL = _rx(r"that'?s why", r'so basically', r'in the end', r'the point is', r'lesson', r'that is it', r"that'?s it",
            r'вот почему', r'поэтому', r'в итоге', r'короче', r'вот так', r'вывод', r'главное', r'итого', r'вот и все')
CONT_START = _rx(r'and', r'but', r'so', r'because', r'then', r'or', r'also', r'which', r'that', r'while',
                 r'и', r'а', r'но', r'потому', r'потому что', r'поэтому', r'тогда', r'или', r'также', r'который\w*',
                 r'что', r'чтобы', r'зато', r'ведь')
ANAPHORA = _rx(r'it', r'this', r'that', r'they', r'he', r'she', r'them', r'those', r'он', r'она', r'они', r'это', r'этот',
               r'эта', r'эти', r'его', r'её', r'их', r'там', r'тут', r'такой', r'такая', r'такие')
FILLER = _rx(r'um+', r'uh+', r'like', r'you know', r'well', r'так', r'ну', r'э+', r'эм+', r'типа', r'значит')
TRAIL_BAD = {'and', 'but', 'so', 'because', 'the', 'a', 'an', 'to', 'of', 'in', 'that', 'or', 'with', 'for', 'is', 'are',
             'и', 'а', 'но', 'что', 'чтобы', 'потому', 'в', 'на', 'с', 'к', 'по', 'у', 'о', 'как', 'если', 'когда',
             'или', 'не', 'за', 'из', 'от', 'для', 'это', 'то'}
DIGIT = re.compile(r'\d')

W_HOOK, W_CTX, W_ENG, W_COMP = 0.28, 0.20, 0.32, 0.20


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _first_words(text, n=4):
    return ' '.join(text.split()[:n])


def _sentence_features(s):
    t = s['text']
    low = t.lower()
    first = _first_words(low)
    n = len(s['words'])
    return {
        'n': n,
        'dur': max(0.01, s['end'] - s['start']),
        'q': t.rstrip('"»”) ').endswith('?') or bool(Q_START.match(first)) and '?' in t,
        'q_start': bool(Q_START.match(first)),
        'excl': t.count('!'),
        'hook': len(HOOK.findall(low)),
        'emo': len(EMO.findall(low)) + t.count('!'),
        'surprise': len(SURPRISE.findall(low)),
        'digit': bool(DIGIT.search(t)),
        'conclusion': bool(CONCL.search(low)),
        'cont_start': bool(CONT_START.match(first)),
        'anaphora': bool(ANAPHORA.match(first)),
        'filler': bool(FILLER.match(first)),
        'trail_bad': tx.tokens(t)[-1:] and tx.tokens(t)[-1] in TRAIL_BAD,
        'terminal': t.rstrip('"»”) ').endswith(tx.TERMINAL),
        'ends_comma': t.rstrip().endswith((',', ';', ':', '—', '-')),
        'kw': tx.keyword_set(t),
    }


def _gen_candidates(sents, feats, words_total, minlen, maxlen, min_words=15):
    out = []
    n = len(sents)
    limit = maxlen - 0.5
    for i in range(n):
        start = sents[i]['start']
        nw = 0
        for j in range(i, n):
            end = sents[j]['end']
            length = end - start
            if length > limit:
                break
            nw += feats[j]['n']
            if length < minlen or nw < min_words:
                continue
            out.append((i, j, length, nw))
    return out


def _score(i, j, length, nw, sents, feats, minlen, maxlen):
    F = feats[i:j + 1]
    first, last = feats[i], feats[j]
    facts = []
    # ---- hook -------------------------------------------------------
    hook = 30.0
    if first['q'] or first['q_start']:
        hook += 25
        facts.append(('hook_question', None))
    if first['hook']:
        hook += 20
        facts.append(('hook_word', None))
    if first['digit']:
        hook += 8
    if first['excl']:
        hook += 8
    if first['n'] <= 12:
        hook += 10
    elif first['n'] > 30:
        hook -= 10
    weak = first['cont_start'] or first['filler']
    if weak:
        hook -= 25
    gap_before = sents[i]['start'] - sents[i - 1]['end'] if i > 0 else 5
    if gap_before >= 0.8:
        hook += 8
    # ---- context ----------------------------------------------------
    ctx = 55.0
    if first['cont_start'] or first['anaphora']:
        ctx -= 25
        facts.append(('weak_start', None))
    if first['filler']:
        ctx -= 10
    if (first['q'] or first['q_start']) and len(F) >= 3:
        ctx += 15
        facts.append(('question_answered', None))
    half = max(1, len(F) // 2)
    k1, k2 = set().union(*[f['kw'] for f in F[:half]]), set().union(*[f['kw'] for f in F[half:]] or [set()])
    coh = tx.jaccard(k1, k2)
    ctx += min(20, coh * 120)
    if coh >= 0.12:
        facts.append(('coherent', None))
    ctx += 10 if nw >= 40 else (-15 if nw < 25 else 0)
    # ---- engagement -------------------------------------------------
    emo = sum(f['emo'] for f in F)
    qn = sum(1 for f in F if f['q'])
    excl = sum(f['excl'] for f in F)
    sur = sum(f['surprise'] for f in F)
    dig = sum(1 for f in F if f['digit'])
    eng = 35.0 + min(30, emo * 8) + min(15, qn * 5) + min(15, excl * 4) + min(10, sur * 5) + min(6, dig * 3)
    rate = nw / max(length, 1)
    if 2.2 <= rate <= 4.2:
        eng += 10
    elif rate < 1.6:
        eng -= 15
    gaps = [sents[k + 1]['start'] - sents[k]['end'] for k in range(i, j)]
    max_gap = max(gaps) if gaps else 0
    speech = sum(sents[k]['end'] - sents[k]['start'] for k in range(i, j + 1))
    silence_ratio = 1 - speech / length
    if max_gap > 2.0:
        eng -= 15
        facts.append(('long_pause', round(max_gap, 1)))
    if silence_ratio > 0.25:
        eng -= 10
    if emo >= 2:
        facts.append(('emotion', emo))
    if sur:
        facts.append(('twist', sur))
    if qn:
        facts.append(('questions', qn))
    # climax: most intense sentence sits in the later part of the clip
    inten = [f['emo'] + f['surprise'] * 1.5 + (2 if f['q'] else 0) for f in F]
    climax = False
    if len(F) >= 3 and max(inten) > 0:
        pos = inten.index(max(inten)) / (len(F) - 1)
        climax = pos >= 0.35
    # ---- completeness ----------------------------------------------
    comp = 40.0
    next_gap = sents[j + 1]['start'] - sents[j]['end'] if j + 1 < len(sents) else 5
    if last['terminal'] or next_gap >= 0.6:
        comp += 35
        facts.append(('complete_end', None))
    if last['conclusion']:
        comp += 15
        facts.append(('conclusion', None))
    if last['trail_bad']:
        comp -= 30
        facts.append(('trailing_word', None))
    if last['ends_comma']:
        comp -= 15
    if climax:
        comp += 10
        facts.append(('climax', None))
    overall = (W_HOOK * _clamp(hook) + W_CTX * _clamp(ctx) + W_ENG * _clamp(eng) + W_COMP * _clamp(comp))
    target = (minlen + maxlen) / 2
    overall -= 0.06 * abs(length - target)
    return {
        'hook': round(_clamp(hook)), 'context': round(_clamp(ctx)), 'engagement': round(_clamp(eng)),
        'completeness': round(_clamp(comp)), 'overall': round(_clamp(overall), 1),
    }, facts


def explain(scores, facts, lang='ru'):
    """Human-readable reasons built only from detected features."""
    ru = lang == 'ru'
    f = {k: v for k, v in facts}
    parts = []
    if 'hook_question' in f:
        parts.append('открывается вопросом' if ru else 'opens with a question')
    if 'hook_word' in f:
        parts.append('в первой фразе есть цепляющее слово' if ru else 'the first sentence has a strong hook word')
    if 'question_answered' in f:
        parts.append('вопрос получает развитие внутри клипа' if ru else 'the opening question is developed inside the clip')
    if 'emotion' in f:
        parts.append(f'эмоциональные маркеры речи ({f["emotion"]})' if ru else f'emotional speech markers ({f["emotion"]})')
    if 'twist' in f:
        parts.append('есть поворот («но», «оказывается» и т.п.)' if ru else 'contains a turn ("but", "turns out", ...)')
    if 'climax' in f:
        parts.append('кульминация во второй части' if ru else 'the peak sits in the later part')
    if 'complete_end' in f:
        parts.append('мысль завершена на границе фразы' if ru else 'ends on a finished sentence')
    if 'conclusion' in f:
        parts.append('есть вывод в конце' if ru else 'closes with a conclusion')
    if 'coherent' in f:
        parts.append('единая тема от начала до конца' if ru else 'one consistent topic throughout')
    if 'laughter' in f:
        parts.append('смех в звуке' if ru else 'laughter in the audio')
    if 'scream' in f:
        parts.append('крик / громкая реакция' if ru else 'a scream / loud reaction')
    if 'raised_voice' in f:
        parts.append('голос повышается' if ru else 'raised voice')
    if 'volume_spike' in f:
        parts.append('резкий всплеск громкости' if ru else 'a sudden volume spike')
    if 'dramatic_pause' in f:
        parts.append('пауза перед взрывом эмоций' if ru else 'a dramatic pause before the burst')
    if 'fast_action' in f or 'action_burst' in f:
        parts.append('активный геймплей / резкие движения' if ru else 'intense gameplay / sudden motion')
    if 'scene_change' in f:
        parts.append(f'смены сцен ({f["scene_change"]})' if ru else f'scene changes ({f["scene_change"]})')
    if 'kept_reaction' in f:
        parts.append('реакция сохранена целиком' if ru else 'the full reaction is kept')
    if 'kept_lead_in' in f:
        parts.append('сохранён момент перед реакцией' if ru else 'the lead-in before the reaction is kept')
    warns = []
    if 'weak_start' in f:
        warns.append('начало опирается на предыдущий контекст' if ru else 'the opening leans on earlier context')
    if 'trailing_word' in f:
        warns.append('конец может обрываться на союзе' if ru else 'may end on a dangling word')
    if 'long_pause' in f:
        warns.append(f'внутри есть пауза {f["long_pause"]} с' if ru else f'contains a {f["long_pause"]}s pause')
    if 'monotone' in f:
        warns.append('ровная громкость без всплесков' if ru else 'flat loudness, no bursts')
    if 'static_picture' in f:
        warns.append('картинка почти статична' if ru else 'the picture is mostly static')
    if 'repetitive' in f:
        warns.append('много повторов в речи' if ru else 'a lot of repetition in speech')
    head = ('Выбран, потому что: ' if ru else 'Picked because: ') + ('; '.join(parts) if parts else
            ('нет ярких маркеров, лучший вариант по общей оценке' if ru else 'no standout markers; best overall candidate'))
    if warns:
        head += ('. Обратите внимание: ' if ru else '. Note: ') + '; '.join(warns)
    return head + '.'


def _pad_bounds(words, i0, i1, sents, i, j, duration):
    s, e = sents[i]['start'], sents[j]['end']
    first_idx = sents[i]['words'][0]
    # find neighbouring words for safe padding
    prev_end = 0.0
    next_start = duration
    for w in words:
        if w['end'] <= s + 1e-6:
            prev_end = max(prev_end, w['end'])
        if w['start'] >= e - 1e-6 and w['start'] < next_start:
            next_start = w['start']
            break
    start = max(prev_end + 0.02, s - 0.25, 0.0)
    start = min(start, s)
    end = min(e + 0.35, next_start - 0.02, duration)
    end = max(end, e + 0.05 if e + 0.05 <= duration else e)
    return round(start, 2), round(end, 2)


# weights of the combined interest score (text = the v5 Hook/Context/Engagement/Completeness overall)
MM_WEIGHTS = {
    (True, True): (0.50, 0.33, 0.17),     # (audio?, action?) -> (text, audio, action)
    (True, False): (0.60, 0.40, 0.0),
    (False, True): (0.80, 0.0, 0.20),
    (False, False): (1.0, 0.0, 0.0),
}


def combine(text, audio=None, action=None):
    wt, wa, wv = MM_WEIGHTS[(audio is not None, action is not None)]
    return wt * text + wa * (audio or 0.0) + wv * (action or 0.0)


def _repetition(text):
    toks = tx.tokens(text)
    if len(toks) < 20:
        return 1.0
    return len(set(toks)) / len(toks)


def _strong_audio(af):
    return bool(af) and (af['strong'] > 0 or af['top'] >= 0.9)


def _refine_bounds(c, sents, words, audio, duration, maxlen):
    """Never cut a reaction or its cause: extend the end over a running reaction / a late peak, and the start
    back over the lead-in when the clip opens on a burst. Words are never cut in half."""
    i, j = c['sent_range']
    start, end = c['start'], c['end']
    facts = []
    evs = audio.events_in(start - 2.0, end + 6.0)
    # ---- end: reaction still running, or the loudest moment sits at the very end
    want_end = end
    for e in evs:
        if e['start'] < end and e['end'] > end - 0.3:
            want_end = max(want_end, min(e['end'] + 0.4, end + 4.0))
    af = c.get('_af')
    if af and af['strong'] and af['peak_t'] >= end - 1.2:
        want_end = max(want_end, min(af['peak_t'] + 1.5, end + 4.0))
    if want_end > end + 0.05:
        nxt = next((w['start'] for w in words if w['start'] >= end - 1e-6), duration)
        if nxt < want_end and j + 1 < len(sents) and sents[j + 1]['end'] + 0.3 - start <= maxlen + 3:
            # the reaction is spoken ("OH MY GOD!") -> keep the whole next sentence
            j += 1
            want_end = max(want_end, sents[j]['end'] + 0.3)
            nxt = next((w['start'] for w in words if w['start'] >= sents[j]['end'] - 1e-6), duration)
        new_end = round(min(want_end, nxt - 0.02, duration), 2)
        if new_end > end + 0.05:
            end = new_end
            facts.append(('kept_reaction', None))
    # ---- start: the clip opens on a burst -> keep what caused it (the gameplay moment right before)
    first_burst = next((e for e in evs if e['end'] > start and e['start'] <= start + 0.8 and
                        (e['strength'] >= 0.8 or e['kind'] in ('scream', 'laughter', 'spike'))), None)
    if first_burst:
        want = max(first_burst['start'] - 1.5, start - 4.0, 0.0)
        prv = max((w['end'] for w in words if w['end'] <= start + 1e-6), default=0.0)
        new_start = round(max(want, prv + 0.02 if prv > 0 else 0.0), 2)
        if new_start < start - 0.05:
            start = new_start
            facts.append(('kept_lead_in', None))
    if j != c['sent_range'][1]:
        c['text'] = ' '.join(s['text'] for s in sents[i:j + 1])
    c['start'], c['end'] = start, end
    c['sent_range'] = (i, j)
    c['facts'] = list(c['facts']) + facts


def find_highlights(segments, duration, minlen=20, maxlen=55, count=3, pool_size=400, diversity=0.35, rerank=None,
                    audio=None, motion=None, motion_top=12):
    """Returns a list of selected highlight dicts, best first. Empty list => caller falls back to legacy.

    audio:  optional audio_analysis.AudioProfile of the whole source (laughter, screams, spikes, pauses...)
    motion: optional callable([(start, end)]) -> motion_analysis.MotionProfile, run only on the best candidates.
    Without them the ranking is exactly the transcript-only v5 ranking.
    """
    words = tx.flat_words(segments)
    if len(words) < 15:
        return []
    sents = tx.build_sentences(words)
    if len(sents) < 2:
        return []
    feats = [_sentence_features(s) for s in sents]
    # reactions are often short on words (screams, laughter): with audio, sparse spans may qualify if they are loud
    gen = _gen_candidates(sents, feats, len(words), minlen, maxlen, min_words=8 if audio is not None else 15)
    if not gen:
        return []
    scored = []
    for (i, j, length, nw) in gen:
        af = None
        if audio is not None:
            # a reaction right after the last words (laughing, screaming) belongs to the moment
            nxt = sents[j + 1]['start'] if j + 1 < len(sents) else duration
            tail = max(0.0, min(2.5, nxt - sents[j]['end'] - 0.05))
            af = audio_mod.span_features(audio, max(0.0, sents[i]['start'] - 0.3), sents[j]['end'] + tail)
            if nw < 15 and not _strong_audio(af):
                continue
        sc, facts = _score(i, j, length, nw, sents, feats, minlen, maxlen)
        text_overall = sc['overall']
        overall = text_overall
        if af is not None:
            a_sc, a_facts = audio_mod.audio_score(af)
            if a_sc is not None:
                facts = list(facts) + a_facts
                if af['longest_pause'] < 1.0 and any(k == 'long_pause' for k, _ in facts):
                    # the transcript has a gap, but the audio is not silent there (game sound, laughter...)
                    facts = [x for x in facts if x[0] != 'long_pause']
                    text_overall = min(100.0, text_overall + 4)
                sc = dict(sc, audio=round(a_sc))
                overall = combine(text_overall, a_sc)
                if a_sc < 35 and sc['engagement'] < 45:
                    overall -= 6
                    facts.append(('boring', None))
        scored.append([overall, i, j, length, nw, sc, facts, af, text_overall])
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = scored[:pool_size]
    lang = tx.detect_lang(' '.join(s['text'] for s in sents[:200]))

    cands = []
    for overall, i, j, length, nw, sc, facts, af, text_overall in pool:
        start, end = _pad_bounds(words, 0, 0, sents, i, j, duration)
        text = ' '.join(s['text'] for s in sents[i:j + 1])
        rep = _repetition(text)
        if rep < 0.4:
            overall -= 8
            facts = list(facts) + [('repetitive', round(rep, 2))]
        cands.append({'start': start, 'end': end, 'text': text, 'score': round(overall, 1), 'scores': sc, 'facts': facts,
                      'kw': set().union(*[feats[k]['kw'] for k in range(i, j + 1)]), 'sent_range': (i, j),
                      '_af': af, '_text': text_overall, '_audio': sc.get('audio')})
    if motion is not None:
        _apply_motion(cands, motion, motion_top)
    for c in cands:
        if 'audio' in c['scores'] or 'action' in c['scores']:
            c['scores'] = dict(c['scores'], overall=round(_clamp(c['score']), 1))
    cands.sort(key=lambda c: c['score'], reverse=True)
    if rerank:
        try:
            cands = rerank(cands)
        except Exception:
            log.exception('rerank failed')
    chosen = _select_diverse(cands, count, diversity, lang, finalize=False)
    if audio is not None:
        for c in chosen:
            try:
                _refine_bounds(c, sents, words, audio, duration, maxlen)
            except Exception:
                log.exception('boundary refinement failed')
    for c in chosen:
        c['reason'] = explain(c['scores'], c['facts'], lang)
        c['lang'] = lang
        for k in ('kw', '_af', '_text', '_audio'):
            c.pop(k, None)
    return chosen


def _apply_motion(cands, motion, top_n):
    """Measure gameplay dynamics on the best (non-overlapping) candidates and fold it into their score."""
    probe = []
    for c in cands:
        if len(probe) >= top_n:
            break
        if any(min(c['end'], p['end']) - max(c['start'], p['start']) > 0.5 * (c['end'] - c['start']) for p in probe):
            continue
        probe.append(c)
    if not probe:
        return
    try:
        mp = motion([(p['start'], p['end']) for p in probe])
    except Exception as exc:
        from .procutil import JobCancelled
        if isinstance(exc, JobCancelled):
            raise
        log.warning('motion analysis failed (%s); ranking without gameplay dynamics', str(exc)[:200])
        return
    if mp is None:
        return
    for c in cands:
        if not mp.covers(c['start'], c['end']):
            continue
        f = mp.span(c['start'], c['end'])
        v_sc, v_facts = motion_mod.action_score(f)
        if v_sc is None:
            continue
        c['scores'] = dict(c['scores'], action=round(v_sc))
        c['facts'] = list(c['facts']) + v_facts
        bonus = c['score'] - combine(c['_text'], c['_audio'])       # repetition / boring adjustments
        c['score'] = round(combine(c['_text'], c['_audio'], v_sc) + bonus, 1)


def _select_diverse(cands, count, diversity, lang, finalize=True):
    chosen = []
    remaining = list(cands)
    while remaining and len(chosen) < count:
        best, best_adj = None, -1e9
        for c in remaining:
            overlap_bad = any(
                max(0, min(c['end'], y['end']) - max(c['start'], y['start'])) >
                min(c['end'] - c['start'], y['end'] - y['start']) * 0.2 for y in chosen)
            if overlap_bad:
                continue
            sim = max((tx.jaccard(c['kw'], y['kw']) for y in chosen), default=0.0)
            if sim > 0.6:
                continue
            adj = c['score'] - diversity * 100 * sim
            if adj > best_adj:
                best, best_adj = c, adj
        if best is None:
            break
        best['diversity_penalty'] = round(best['score'] - best_adj, 1)
        chosen.append(best)
        remaining = [c for c in remaining if c is not best]
    if finalize:
        for c in chosen:
            c['reason'] = explain(c['scores'], c['facts'], lang)
            c['lang'] = lang
            c.pop('kw', None)
    return chosen
