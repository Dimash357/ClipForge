"""AI Script Generator (3A): scripts for Faceless Shorts, stories, educational videos and cartoons.

A script document (Script.data, version 1):
    {version, title, logline, format, language, duration, genre, style, audience, provider, note,
     characters: [{id, name, description, voice}],
     scenes: [{id, part: hook|development|climax|ending, start, duration, title,
               lines: [{speaker: 'narrator' | character id, text}],
               visual, shot, mood, sfx, music, on_screen_text,
               storyboard: {prompt, negative, camera, style, frames: []},      <- filled by 3B (AI Storyboard)
               revision, locked}]}
Scene ids are stable (ids.stable_id) and survive edits and regenerations, so later stages (storyboard, voice,
render) can redo one scene only.

Text comes from the configured LLM (ai_provider: Ollama / LM Studio / any OpenAI-compatible URL - local models
need no paid API). Without a model, or when its answer is unusable, a template draft is built from the idea, and
the script says so (provider 'template'). Local models run on this computer's GPU: the generation holds a VRAM
lease (resource manager) so it never collides with Whisper or NVENC jobs.
"""
import copy
import logging
import re
import time

from . import ai_provider
from .timeline.ids import stable_id

log = logging.getLogger('studio.scripts')

VERSION = 1
FORMATS = {'faceless': 'Faceless Shorts', 'story': 'История', 'education': 'Обучающее видео', 'cartoon': 'Мультфильм'}
PARTS = ('hook', 'development', 'climax', 'ending')
PART_RU = {'hook': 'Хук', 'development': 'Развитие', 'climax': 'Кульминация', 'ending': 'Завершение'}
LANGS = {'ru': 'русский', 'en': 'English', 'uk': 'українська', 'kk': 'қазақ тілі', 'es': 'español', 'de': 'Deutsch'}
WPS = {'ru': 2.3, 'uk': 2.3, 'kk': 2.2, 'en': 2.6, 'es': 2.7, 'de': 2.4}      # spoken words per second
MIN_SCENE, MAX_SCENE = 2.5, 15.0
LLM_TIMEOUT = 240
SCENE_FIELDS = ('title', 'visual', 'shot', 'mood', 'sfx', 'music', 'on_screen_text')


# ------------------------------------------------------------------ settings
def settings_of(sc):
    """Generation settings of a Script model (or a dict with the same keys)."""
    g = (lambda k, d=None: getattr(sc, k, d)) if not isinstance(sc, dict) else (lambda k, d=None: sc.get(k, d))
    fmt = g('format') if g('format') in FORMATS else 'faceless'
    lang = (g('language') or 'ru')[:8]
    return {'idea': (g('idea') or '').strip()[:2000], 'format': fmt, 'language': lang,
            'duration': max(10, min(600, int(g('duration') or 45))), 'genre': (g('genre') or '')[:60],
            'style': (g('style') or '')[:60], 'audience': (g('audience') or '')[:120],
            'title': (g('title') or '')[:250], 'key': f'script-{getattr(sc, "pk", None) or g("id") or "x"}'}


def scene_count(duration):
    return max(4, min(20, int(round(duration / 7.0))))


def plan_parts(n):
    """Which part each of n scenes belongs to: 1 hook, ~20 % climax, 1 ending, the rest development."""
    climax = max(1, round(n * 0.2))
    return ['hook'] + ['development'] * (n - 2 - climax) + ['climax'] * climax + ['ending']


# ------------------------------------------------------------------ prompts
SYSTEM = ('You are a professional short-form video scriptwriter. You write tight, vivid scripts for vertical videos '
          '(YouTube Shorts, TikTok, Reels): a hook in the first 2 seconds, rising development, one clear climax, '
          'a satisfying ending. Every scene has spoken lines, a concrete visual description a storyboard artist can '
          'draw, and a duration. Answer with ONE JSON object only, no prose.')


def _brief(st):
    lang = LANGS.get(st['language'], st['language'])
    return (f'Format: {FORMATS[st["format"]]}. Language of all spoken lines and titles: {lang}. '
            f'Total duration: {st["duration"]} seconds. Genre: {st["genre"] or "any"}. Style: {st["style"] or "any"}. '
            f'Audience: {st["audience"] or "general"}.\nIdea / topic: {st["idea"]}')


SCHEMA_HINT = ('{"title": str, "logline": str, "characters": [{"name": str, "description": str, "voice": str}], '
               '"scenes": [{"part": "hook|development|climax|ending", "title": str, "duration": number (seconds), '
               '"lines": [{"speaker": "narrator" or a character name, "text": str}], "visual": str (what we see), '
               '"shot": str (camera framing/movement), "mood": str, "sfx": str, "music": str, '
               '"on_screen_text": str}]}')


def prompt_full(st):
    n = scene_count(st['duration'])
    extra = ('Use 2-3 recurring characters with distinct voices; lines are their dialogue.' if st['format'] == 'cartoon'
             else 'Use a narrator voice-over; characters only if the story needs them.' if st['format'] != 'story'
             else 'Narrator plus characters as the story needs.')
    return (f'{_brief(st)}\n\nWrite the script as {n} scenes (1 hook, development, 1-3 climax scenes, 1 ending). '
            f'Scene durations must add up to about {st["duration"]} s; spoken text must fit its scene '
            f'(about {WPS.get(st["language"], 2.4)} words per second). {extra} Visual descriptions are in English '
            f'(they become image prompts). JSON shape: {SCHEMA_HINT}')


def prompt_scene(doc, scene, instruction):
    others = '\n'.join(f'- [{s["part"]}] {s["title"]}: ' + ' '.join(l['text'] for l in s['lines'])[:200]
                       for s in doc['scenes'] if s['id'] != scene['id'])
    return (f'{_brief(doc)}\n\nScript "{doc["title"]}". Other scenes:\n{others}\n\nRewrite ONLY this scene '
            f'(part: {scene["part"]}, about {scene["duration"]:.0f} s): {scene["title"]} - '
            f'{" ".join(l["text"] for l in scene["lines"])}\nWish: {instruction or "make it stronger"}.\n'
            f'Keep it consistent with the other scenes. JSON: one scene object of this shape: '
            f'{SCHEMA_HINT.split(chr(34) + "scenes" + chr(34) + ": [")[1].rstrip("]}")}')


# ------------------------------------------------------------------ normalisation (whatever the model returns)
def _s(v, n=400):
    return re.sub(r'\s+', ' ', str(v or '')).strip()[:n]


def _speaker(name, chars):
    n = _s(name, 60).lower()
    if not n or n in ('narrator', 'рассказчик', 'диктор', 'voice', 'voiceover', 'voice-over'):
        return 'narrator'
    for c in chars:
        if c['name'].lower() == n or c['id'] == name:
            return c['id']
    return 'narrator'


def _scene(raw, doc_key, idx, chars, part=None):
    lines = []
    for l in raw.get('lines') or []:
        if isinstance(l, dict) and _s(l.get('text')):
            lines.append({'speaker': _speaker(l.get('speaker'), chars), 'text': _s(l.get('text'), 600)})
        elif isinstance(l, str) and _s(l):
            lines.append({'speaker': 'narrator', 'text': _s(l, 600)})
    if not lines and _s(raw.get('text') or raw.get('narration')):
        lines.append({'speaker': 'narrator', 'text': _s(raw.get('text') or raw.get('narration'), 600)})
    p = raw.get('part') if raw.get('part') in PARTS else part or 'development'
    try:
        dur = float(raw.get('duration') or 0)
    except (TypeError, ValueError):
        dur = 0.0
    sc = {'id': raw.get('id') or stable_id('sc', doc_key, idx), 'part': p, 'start': 0.0, 'duration': dur,
          'lines': lines, 'revision': int(raw.get('revision') or 1), 'locked': bool(raw.get('locked'))}
    for k in SCENE_FIELDS:
        sc[k] = _s(raw.get(k), 600)
    sc['title'] = sc['title'] or PART_RU[p]
    sb = raw.get('storyboard') if isinstance(raw.get('storyboard'), dict) else {}
    sc['storyboard'] = {'prompt': _s(sb.get('prompt') or sc['visual'], 800), 'negative': _s(sb.get('negative'), 300),
                        'camera': _s(sb.get('camera') or sc['shot'], 200), 'style': _s(sb.get('style'), 200),
                        'frames': list(sb.get('frames') or [])}
    return sc


def spoken_seconds(scene, lang):
    words = sum(len(l['text'].split()) for l in scene['lines'])
    return words / WPS.get(lang, 2.4) + 0.6


def retime(doc):
    """Scene durations: at least what their lines need, scaled towards the target total; starts recomputed."""
    lang = doc['language']
    need = [max(MIN_SCENE, spoken_seconds(s, lang)) for s in doc['scenes']]
    want = [max(n, s['duration'] or 0) for n, s in zip(need, doc['scenes'])]
    total = sum(want) or 1.0
    k = doc['duration'] / total
    t = 0.0
    for s, w, n in zip(doc['scenes'], want, need):
        d = w * k if k >= 1 else max(n, w * k)        # never squeeze speech below its natural pace
        s['duration'] = round(min(MAX_SCENE, max(MIN_SCENE, d)), 2)
        s['start'] = round(t, 2)
        t += s['duration']
    doc['total'] = round(t, 2)
    return doc


def normalize(raw, st, provider='template', note=''):
    """Model answer (or template) -> a valid script document."""
    key = st['key']
    chars = []
    for i, c in enumerate(raw.get('characters') or []):
        if isinstance(c, dict) and _s(c.get('name')):
            chars.append({'id': stable_id('ch', key, _s(c['name'], 60).lower()), 'name': _s(c['name'], 60),
                          'description': _s(c.get('description'), 400), 'voice': _s(c.get('voice'), 80)})
    scenes = [_scene(s, key, i, chars) for i, s in enumerate(raw.get('scenes') or []) if isinstance(s, dict)]
    scenes = [s for s in scenes if s['lines'] or s['visual']]
    if not scenes:
        raise ValueError('the answer has no usable scenes')
    if scenes[0]['part'] != 'hook':
        scenes[0]['part'] = 'hook'
    if len(scenes) > 1 and scenes[-1]['part'] != 'ending':
        scenes[-1]['part'] = 'ending'
    if len(scenes) > 2 and not any(s['part'] == 'climax' for s in scenes):
        scenes[-2]['part'] = 'climax'
    doc = {'version': VERSION, 'title': _s(raw.get('title') or st['title'] or st['idea'][:60], 200),
           'logline': _s(raw.get('logline'), 400), 'format': st['format'], 'language': st['language'],
           'duration': st['duration'], 'genre': st['genre'], 'style': st['style'], 'audience': st['audience'],
           'idea': st['idea'], 'provider': provider, 'note': note, 'characters': chars, 'scenes': scenes,
           'generated_at': int(time.time())}
    return retime(doc)


# ------------------------------------------------------------------ template drafts (no model needed)
T = {
    'ru': {
        'faceless': ['Вы точно не знали этого про {t}.', 'Начнём с простого: {t} устроено совсем не так, как кажется.',
                     'Вот первый факт, который всё меняет.', 'А теперь — деталь, о которой молчат.',
                     'И самое неожиданное: {t} может удивить даже тех, кто разбирается.', 'Сохраните, чтобы не забыть, и напишите, что знали вы.'],
        'story': ['Никто не думал, что всё начнётся с {t}.', 'Сначала всё шло как обычно.', 'Но потом случилось странное.',
                  'Становилось только хуже — выхода не было видно.', 'И в этот момент всё перевернулось.', 'С тех пор {t} — уже совсем другая история.'],
        'education': ['Почему {t} работает именно так? Разберёмся за минуту.', 'Сначала главное понятие.',
                      'Шаг второй: как это связано между собой.', 'Пример из жизни, чтобы точно запомнить.',
                      'Ключевая мысль: вот что на самом деле важно в {t}.', 'Коротко: повторим главное. Сохраняйте и делитесь.'],
        'cartoon': ['Смотри, что я сегодня узнал: {t}!', 'Ты уверен, что это хорошая идея?', 'Конечно! Что может пойти не так?',
                    'Кажется, всё идёт не по плану…', 'Держись! Сейчас мы всё исправим!', 'В следующий раз сначала думаем, ладно?'],
        'cast': [('Герой', 'смелый и немного неуклюжий', 'бодрый голос'), ('Друг', 'осторожный и ироничный', 'спокойный голос')],
        'note': 'Шаблонный черновик: локальная LLM не подключена или не ответила. Отредактируйте текст или подключите Ollama.',
    },
    'en': {
        'faceless': ['You definitely did not know this about {t}.', 'Start simple: {t} is not what it seems.',
                     'Here is the first fact that changes everything.', 'Now the detail nobody talks about.',
                     'And the biggest surprise: {t} can shock even experts.', 'Save this and tell me what you knew.'],
        'story': ['Nobody expected it to start with {t}.', 'At first, everything was normal.', 'Then something strange happened.',
                  'It only got worse - there was no way out.', 'And in that moment, everything flipped.', 'Since then, {t} is a different story.'],
        'education': ['Why does {t} work like this? One minute.', 'First, the key idea.', 'Step two: how it connects.',
                      'A real-life example so it sticks.', 'The key insight: what really matters about {t}.', 'Quick recap. Save and share.'],
        'cartoon': ['Look what I found out today: {t}!', 'Are you sure this is a good idea?', 'Of course! What could go wrong?',
                    'This is not going as planned...', 'Hold on! We will fix it!', 'Next time we think first, okay?'],
        'cast': [('Hero', 'brave and a bit clumsy', 'energetic voice'), ('Buddy', 'careful and witty', 'calm voice')],
        'note': 'Template draft: no local LLM is connected or it did not answer. Edit the text or connect Ollama.',
    },
}


def _tlang(lang):
    return lang if lang in T else 'ru' if lang in ('uk', 'kk') else 'en'


def template(st):
    lang = _tlang(st['language'])
    L = T[lang]
    topic = st['idea'].rstrip('.!?') or ('эта тема' if lang == 'ru' else 'this topic')
    topic_short = ' '.join(topic.split()[:8])
    lines = L[st['format']]
    n = scene_count(st['duration'])
    parts = plan_parts(n)
    chars = [{'name': nm, 'description': d, 'voice': v} for nm, d, v in L['cast']] if st['format'] == 'cartoon' else []
    pick = lambda i: lines[min(len(lines) - 1, round(i * (len(lines) - 1) / max(1, n - 1)))]      # noqa: E731
    scenes = []
    for i, part in enumerate(parts):
        text = pick(i).format(t=topic_short)
        speaker = chars[i % 2]['name'] if chars else 'narrator'
        scenes.append({'part': part, 'title': PART_RU[part] if lang == 'ru' else part.title(),
                       'lines': [{'speaker': speaker, 'text': text}],
                       'visual': f'{part} scene about {topic_short}, vertical 9:16, {st["style"] or "clean modern"} style',
                       'shot': 'close-up, quick push-in' if part in ('hook', 'climax') else 'medium shot',
                       'mood': {'hook': 'intriguing', 'development': 'curious', 'climax': 'intense',
                                'ending': 'satisfying'}[part],
                       'on_screen_text': text.upper()[:40] if part == 'hook' else ''})
    return {'title': st['title'] or topic_short[:60], 'logline': topic, 'characters': chars, 'scenes': scenes}, L['note']


# ------------------------------------------------------------------ generation (LLM with a VRAM lease, else template)
def _ask(system, user, cancel=None):
    """The model's JSON answer or (None, reason). A local model holds a VRAM lease while it runs."""
    from . import resources
    if not ai_provider.available():
        return None, 'no model configured'
    lease = None
    if ai_provider.is_local():
        mgr = resources.manager()
        vram = int(__import__('os').getenv('CLIPFORGE_LLM_VRAM_MB', '5000')) if mgr.facts()['vram_total'] else 0
        try:
            lease = mgr.acquire(label='LLM: сценарий', vram_mb=min(vram, mgr.capacity()['vram_mb']), cpu=1,
                                timeout=600, cancel=cancel)
        except resources.Unavailable as exc:
            return None, f'GPU busy: {exc}'
    try:
        return ai_provider.complete_json(system, user, temperature=0.85, timeout=LLM_TIMEOUT)
    finally:
        if lease is not None:
            lease.release()


def generate(st, cancel=None):
    """-> script document (LLM if available, else template)."""
    raw, how = _ask(SYSTEM, prompt_full(st), cancel)
    if raw:
        try:
            return normalize(raw, st, provider=how)
        except (ValueError, TypeError, AttributeError) as exc:
            log.warning('script answer unusable (%s); template draft', exc)
            how = f'unusable answer: {exc}'
    raw, note = template(st)
    return normalize(raw, st, provider='template', note=f'{note} ({how})')


def regenerate_scene(doc, scene_id, instruction='', cancel=None):
    """New version of one scene (same id, same part, revision+1); the rest of the script is untouched."""
    doc = copy.deepcopy(doc)
    i = next((k for k, s in enumerate(doc['scenes']) if s['id'] == scene_id), None)
    if i is None:
        raise KeyError(scene_id)
    old = doc['scenes'][i]
    st = settings_of(doc)
    st['key'] = old['id']
    raw, how = _ask(SYSTEM, prompt_scene(doc, old, instruction), cancel)
    new = None
    if raw:
        try:
            cand = raw['scenes'][0] if isinstance(raw.get('scenes'), list) and raw['scenes'] else raw
            new = _scene(cand, old['id'], 0, doc['characters'], part=old['part'])
            if not new['lines']:
                new = None
        except (TypeError, AttributeError, KeyError):
            new = None
    if new is None:                                       # template: another phrasing from the same pool
        pool = T[_tlang(doc['language'])][doc['format']]
        text = pool[(PARTS.index(old['part']) * 2 + old['revision']) % len(pool)].format(
            t=' '.join(doc['idea'].split()[:8]))
        new = dict(copy.deepcopy(old), lines=[{'speaker': old['lines'][0]['speaker'] if old['lines'] else 'narrator',
                                               'text': text}])
        how = 'template'
    new.update(id=old['id'], part=old['part'], revision=old['revision'] + 1, locked=False,
               duration=old['duration'], storyboard=dict(old['storyboard'], prompt=new['storyboard']['prompt'] or
                                                         old['storyboard']['prompt'], frames=[]))
    new['history'] = (old.get('history') or [])[-4:] + [{k: old[k] for k in ('lines', 'visual', 'revision')}]
    doc['scenes'][i] = new
    doc['provider_last'] = how
    return retime(doc)


def edit_scene(doc, scene_id, changes):
    """Manual edit of one scene (lines / texts / duration / lock). Unknown keys are ignored."""
    doc = copy.deepcopy(doc)
    s = next((x for x in doc['scenes'] if x['id'] == scene_id), None)
    if s is None:
        raise KeyError(scene_id)
    if isinstance(changes.get('lines'), list):
        ids = {c['id'] for c in doc['characters']}
        s['lines'] = [{'speaker': l['speaker'] if l.get('speaker') in ids else _speaker(l.get('speaker'),
                                                                                         doc['characters']),
                       'text': _s(l.get('text'), 600)}
                      for l in changes['lines'] if isinstance(l, dict) and _s(l.get('text'))]
    for k in SCENE_FIELDS:
        if k in changes:
            s[k] = _s(changes[k], 600)
    if 'visual' in changes:
        s['storyboard']['prompt'] = s['visual']
    if 'duration' in changes:
        try:
            s['duration'] = max(MIN_SCENE, min(MAX_SCENE, float(changes['duration'])))
        except (TypeError, ValueError):
            pass
    if 'locked' in changes:
        s['locked'] = bool(changes['locked'])
    if changes.get('part') in PARTS:
        s['part'] = changes['part']
    s['revision'] += 1
    return retime(doc)


# ------------------------------------------------------------------ Timeline IR
def to_timeline(doc, script_id=None, storyboard=True):
    """The script as a Timeline IR (no media yet): scenes with timings and generation prompts, characters, the
    spoken text as caption words (evenly timed until a voice exists), on-screen text as overlay items.
    The storyboard (3B) and voice stages fill in assets scene by scene."""
    from .timeline import model as m
    from .timeline.validate import ensure_valid
    key = f'script-{script_id or doc.get("generated_at")}'
    tl_id = stable_id('tl', key)
    kind = {'faceless': 'faceless', 'cartoon': 'cartoon'}.get(doc['format'], 'generic')
    chars = [m.Character(id=c['id'], name=c['name'], description=c['description'], voice=c['voice'])
             for c in doc['characters']]
    style = m.Style(id=stable_id('st', key, 'captions'), font='Montserrat Black', size=90, uppercase=True, max_words=2,
                    position='middle')
    hook = m.Style(id=stable_id('st', key, 'title'), font='Montserrat Black', size=110, uppercase=True, position='custom')
    scenes, words, texts = [], [], []
    for s in doc['scenes']:
        scenes.append(m.Scene(id=s['id'], start=s['start'], duration=s['duration'], name=s['title'][:80],
                              status='draft',
                              characters=sorted({l['speaker'] for l in s['lines'] if l['speaker'] != 'narrator'}),
                              generation=m.Generation(provider='script', prompt=s['storyboard']['prompt'],
                                                      negative=s['storyboard']['negative'],
                                                      params={'part': s['part'], 'shot': s['shot'], 'mood': s['mood'],
                                                              'sfx': s['sfx'], 'music': s['music']},
                                                      revision=s['revision']),
                              notes=s['visual'][:400], ext={'clipforge.script': {'part': s['part'], 'lines': s['lines']}}))
        toks = [w for l in s['lines'] for w in l['text'].split()]
        if toks:
            step = (s['duration'] - 0.4) / len(toks)
            for k, w in enumerate(toks):
                words.append(m.Word(t0=round(s['start'] + 0.2 + k * step, 3),
                                    t1=round(s['start'] + 0.2 + (k + 1) * step - 0.02, 3), text=w))
        if s['on_screen_text']:
            texts.append(m.TextItem(id=stable_id('it', key, 'txt', s['id']), start=s['start'],
                                    duration=min(s['duration'], 2.5), scene=s['id'], text=s['on_screen_text'][:80],
                                    style=hook.id, y=0.25))
    total = round(doc['scenes'][-1]['start'] + doc['scenes'][-1]['duration'], 3)
    tracks = [m.Track(id=stable_id('tr', key, 'v'), type='video', name='Видео (будет из раскадровки)')]
    if texts:
        tracks.append(m.Track(id=stable_id('tr', key, 'ov'), type='overlay', name='Текст на экране', items=texts))
    if words:
        tracks.append(m.Track(id=stable_id('tr', key, 'cap'), type='caption', name='Субтитры', items=[
            m.CaptionItem(id=stable_id('it', key, 'cap'), start=0.0, duration=total, words=words, style=style.id)]))
    tl = m.Timeline(id=tl_id, kind=kind, duration=total, scenes=scenes, tracks=tracks, styles=[style, hook],
                    characters=chars, meta={'title': doc['title'], 'source': 'script', 'script_id': script_id,
                                            'format': doc['format'], 'language': doc['language']},
                    ext={'clipforge.script': {'version': doc['version'], 'provider': doc['provider']}})
    tl = ensure_valid(tl)
    if storyboard and doc.get('storyboard'):
        from .storyboard import apply_to_timeline
        tl = apply_to_timeline(tl, doc)
    return tl
