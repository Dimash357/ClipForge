"""Metadata Generator: titles / description / hashtags built from the clip's own transcript.

Extractive by default (nothing is invented). With USE_OLLAMA=1 it can also produce RU+EN variants from the same
excerpt; the output is validated and silently falls back to the extractive version.
"""
import json
import logging
import os
import re

from . import textutil as tx
from . import highlights as hl

log = logging.getLogger('studio.meta')


def _trim(text, limit=70):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,;:.\-–—\s]+', '', text)
    if len(text) <= limit:
        return text.rstrip(',;:-–— ')
    cut = text[:limit].rsplit(' ', 1)[0]
    return cut.rstrip(',;:-–— ') + '…'


def _sentences(text):
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def _hashtag(word):
    w = re.sub(r'[^\w]', '', word, flags=re.U)
    return '#' + w.capitalize() if w else ''


def extractive(text, lang=None, scene_types=()):
    lang = lang or tx.detect_lang(text)
    sents = _sentences(text) or [text]
    kws = tx.keywords(text, 6)
    titles = [_trim(sents[0])]
    scored = []
    for s in sents:
        f = hl._sentence_features({'text': s, 'words': s.split(), 'start': 0, 'end': 1})
        scored.append((f['emo'] * 2 + f['hook'] * 2 + (3 if f['q'] else 0) + f['surprise'] + (1 if f['digit'] else 0), s))
    best = max(scored, key=lambda x: x[0])[1] if scored else sents[0]
    titles.append(_trim(best))
    q = next((s for s in sents if s.rstrip().endswith('?')), None)
    if q:
        titles.append(_trim(q))
    if len(sents) > 2 and sents[-1].rstrip().endswith(tx.TERMINAL):
        titles.append(_trim(sents[-1]))
    if len(kws) >= 2:
        titles.append(f'Про {kws[0]} и {kws[1]}' if lang == 'ru' else f'About {kws[0]} and {kws[1]}')
    seen, uniq = set(), []
    for t in titles:
        key = t.lower().rstrip('.!?…')
        if t and key not in seen:
            seen.add(key)
            uniq.append(t)
    uniq = uniq[:5]
    tags = ['#shorts'] + [h for h in (_hashtag(k) for k in kws[:4]) if h]
    if {'gameplay', 'game_cam'} & set(scene_types):
        tags.append('#gaming')
    desc = _trim(best, 160)
    return {'titles': uniq, 'description': desc, 'hashtags': tags[:7], 'lang': lang, 'bilingual': False,
            'source': 'extractive'}


def _ollama(text, lang):
    from . import ai_provider
    if os.getenv('USE_OLLAMA', '0') != '1' or ai_provider.paused():
        return None
    try:
        import requests
        model = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
        prompt = ('You write titles for a short video using ONLY facts present in the transcript excerpt below. '
                  'Do not invent names, numbers or events. Return ONLY JSON: {"titles_ru":[3-5 strings],'
                  '"titles_en":[3-5 strings],"description_ru":"1-2 sentences","description_en":"1-2 sentences"}.\n'
                  'Transcript excerpt:\n' + text[:1500])
        r = requests.post(os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/') + '/api/generate',
                          json={'model': model, 'prompt': prompt, 'stream': False, 'format': 'json'},
                          timeout=(3.05, 90))
        r.raise_for_status()
        raw = json.loads(r.json()['response'])

        def lst(key):
            v = raw.get(key)
            if not isinstance(v, list):
                return []
            return [_trim(str(x), 90) for x in v if isinstance(x, str) and 3 <= len(x.strip()) <= 140][:5]

        ru, en = lst('titles_ru'), lst('titles_en')
        if len(ru) < 2 or len(en) < 2:
            return None
        return {'titles_ru': ru, 'titles_en': en, 'description_ru': _trim(str(raw.get('description_ru', '')), 220),
                'description_en': _trim(str(raw.get('description_en', '')), 220)}
    except Exception as exc:
        log.info('ollama metadata skipped: %s', exc)
        if not isinstance(exc, (ValueError, KeyError)):
            ai_provider.trip(exc)
        return None


def generate(text, scene_types=()):
    base = extractive(text, scene_types=scene_types)
    extra = _ollama(text, base['lang'])
    if extra:
        base.update(extra)
        base['bilingual'] = True
        base['source'] = 'ollama'
        base['titles'] = extra['titles_ru'] if base['lang'] == 'ru' else extra['titles_en']
        base['description'] = (extra['description_ru'] if base['lang'] == 'ru' else extra['description_en']) or base['description']
    return base
