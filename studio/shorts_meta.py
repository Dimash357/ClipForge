"""YouTube Shorts title + description for a rendered clip.

Evidence used (all already computed and cached, nothing is decoded again):
  * the clip's transcript (Whisper) and its language;
  * speech cues (moments.py): fails, exclamations, swearing, donations, chat, absurd situations, laughter;
  * audio events in the clip span (audio_analysis cache): screams, laughter, raised voice, volume spikes;
  * screen events (keyframe scan cache): death/fail screens, flashes; scene types of the rendered layout;
  * a streamer name ONLY when it is verified (typed by the user, or present in the video's own YouTube
    title/channel and listed in studio/data/streamers.txt).

Generation: the configured AI provider (ai_provider.py) when available, strictly validated (CAPS, 1-3 emoji,
language, no unknown names, no numbers that are not in the speech); otherwise a local, template-free extractive
generator that quotes what was actually said. The result is stored in Clip.meta['short'].
"""
import hashlib
import logging
import re
import time
from pathlib import Path

from . import ai_provider, moments
from . import textutil as tx

log = logging.getLogger('studio.shorts')
VERSION = 1
TITLE_MAX = 100          # YouTube limit
HASHTAGS_MAX = 8

# category -> (emoji, RU tag phrase, EN tag phrase, RU hashtags, EN hashtags)
CATS = {
    'fail': ('💀', 'ФЕЙЛ', 'FAIL', ['#фейл', '#смешныемоменты'], ['#fail', '#funnymoments']),
    'laughter': ('😂', 'СМЕХ ДО СЛЁЗ', "COULDN'T STOP LAUGHING", ['#смешно', '#ржака'], ['#funny', '#lol']),
    'scream': ('😱', 'ЭТО НАДО СЛЫШАТЬ', 'LISTEN TO THIS', ['#реакция', '#крик'], ['#reaction', '#scream']),
    'absurd': ('🤯', 'ЧТО ПРОИСХОДИТ', 'WHAT IS HAPPENING', ['#баг', '#игровыемоменты'], ['#glitch', '#gamingmoments']),
    'donation': ('💸', 'РЕАКЦИЯ НА ДОНАТ', 'DONATION REACTION', ['#донат', '#стрим'], ['#donation', '#stream']),
    'exclaim': ('😳', 'РЕАКЦИЯ', 'REACTION', ['#реакция'], ['#reaction']),
    'profanity': ('🤬', 'НЕ СДЕРЖАЛСЯ', 'LOST IT', ['#эмоции'], ['#rage']),
    'chat': ('💬', 'РАЗГОВОР С ЧАТОМ', 'TALKING TO CHAT', ['#чат', '#стрим'], ['#chat', '#stream']),
    'action': ('🔥', 'ЖАРКИЙ МОМЕНТ', 'INTENSE MOMENT', ['#эпик'], ['#epic']),
}
ORDER = ['fail', 'laughter', 'scream', 'absurd', 'donation', 'exclaim', 'profanity', 'chat', 'action']
CUE_TO_CAT = {'fail': 'fail', 'laugh_text': 'laughter', 'exclaim': 'exclaim', 'profanity': 'profanity',
              'donation': 'donation', 'chat': 'chat', 'absurd': 'absurd'}
# for choosing what the clip is ABOUT: exclamations / swearing / talking to chat are the tone, not the event
CAT_WEIGHT = {'exclaim': 6, 'profanity': 6, 'chat': 4}
EMOJI_RX = re.compile('[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]️?')


# ------------------------------------------------------------------ streamer names
def _load_streamers():
    path = Path(__file__).parent / 'data' / 'streamers.txt'
    out = []
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                variants = [v.strip() for v in line.split('|') if v.strip()]
                if variants:
                    out.append(variants)
    except OSError:
        pass
    return out


STREAMERS = _load_streamers()


def _name_rx(name):
    return re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I | re.U)


def find_known_names(text):
    """Canonical names of known streamers mentioned in `text`."""
    found = []
    for variants in STREAMERS:
        if any(_name_rx(v).search(text or '') for v in variants):
            found.append(variants[0])
    return found


def verified_streamer(user_value='', source_meta=None):
    """A name we are allowed to use: typed by the user, or in the video's own YouTube title/channel."""
    user_value = re.sub(r'[^\w .\-]', '', (user_value or ''), flags=re.U).strip()[:40]
    if user_value:
        return user_value
    meta = source_meta or {}
    names = find_known_names(f"{meta.get('title', '')} {meta.get('author', '')}")
    return names[0] if names else None


# ------------------------------------------------------------------ evidence
def build_context(text, lang=None, audio_events=(), screen_events=(), scene_types=(), scores=None, streamer=None,
                  duration=None):
    text = re.sub(r'\s+', ' ', text or '').strip()
    lang = lang if lang in ('ru', 'en') else tx.detect_lang(text)
    sents = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text) if s.strip()]
    cue_sents = [(moments.sentence_cues(s), s) for s in sents]
    cues = {}
    for c, _ in cue_sents:
        for k, n in c.items():
            cues[k] = cues.get(k, 0) + n
    audio = {}
    for e in audio_events:
        audio[e['kind']] = audio.get(e['kind'], 0) + 1
    screen = {}
    for e in screen_events:
        screen[e['kind']] = screen.get(e['kind'], 0) + 1
    scores = scores or {}
    cats = {}
    for k, n in cues.items():
        cat = CUE_TO_CAT.get(k)
        if cat:
            cats[cat] = cats.get(cat, 0) + n * CAT_WEIGHT.get(k, moments.CUE_WEIGHT.get(k, 8))
    if audio.get('laughter'):
        cats['laughter'] = cats.get('laughter', 0) + 16 * audio['laughter']
    if audio.get('scream'):
        cats['scream'] = cats.get('scream', 0) + 15 * audio['scream']
    if screen.get('fail_screen'):
        cats['fail'] = cats.get('fail', 0) + 14 * screen['fail_screen']
    if float(scores.get('action') or 0) >= 65 or screen.get('flash'):
        cats['action'] = cats.get('action', 0) + 10
    if cues.get('fail') and screen.get('fail_screen'):
        cats['fail'] += 20                            # said AND seen: the clip is about this fail
    if cues.get('laugh_text') and audio.get('laughter'):
        cats['laughter'] += 12
    gaming = bool({'gameplay', 'game_cam', 'reaction_cam', 'action'} & set(scene_types))
    return {'text': text, 'lang': lang, 'sentences': cue_sents, 'cues': cues, 'audio': audio, 'screen': screen,
            'cats': sorted(cats, key=lambda k: (-cats[k], ORDER.index(k))), 'gaming': gaming,
            'streamer': streamer, 'duration': duration, 'keywords': tx.keywords(text, 4)}


def context_key(ctx):
    raw = '|'.join([str(VERSION), ctx['text'], ctx['lang'], str(ctx.get('streamer')), ','.join(ctx['cats'])])
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


# ------------------------------------------------------------------ helpers
def _clean_quote(s, limit=64):
    s = re.sub(r'^[\-–—,.;:\s]+', '', s).strip()
    s = re.sub(r'\s+', ' ', s)
    if len(s) > limit:
        s = s[:limit].rsplit(' ', 1)[0].rstrip(',;:-–— ')
    return s.rstrip(',;:-–— ')


def emoji_count(s):
    return len(EMOJI_RX.findall(s or ''))


def _emojis(ctx, n=2):
    em = [CATS[c][0] for c in ctx['cats'] if c in CATS]
    if ctx['gaming'] and '🎮' not in em:
        em.append('🎮')
    if not em:
        em = ['🔥']
    out = []
    for e in em:
        if e not in out:
            out.append(e)
    return out[:max(1, min(3, n))]


def _quotes(ctx):
    """Spoken lines worth quoting, best first: the clip's main event, reaction cues, short, emotional.
    A one/two-word exclamation ("ЧТО?!") is joined with the next sentence ("ЧТО?! МЕНЯ ОПЯТЬ УБИЛИ!")."""
    sents = ctx['sentences']
    main = ctx['cats'][0] if ctx['cats'] else None
    units = []
    for k, (cues, s) in enumerate(sents):
        units.append((cues, s))
        if len(s.split()) <= 2 and k + 1 < len(sents) and len(sents[k + 1][1].split()) <= 10:
            merged = dict(cues)
            for c, n in sents[k + 1][0].items():
                merged[c] = merged.get(c, 0) + n
            units.append((merged, s + ' ' + sents[k + 1][1]))
    cand = []
    for cues, s in units:
        words = s.split()
        if len(words) < 2 or len(words) > 14:
            continue
        w = sum(moments.CUE_WEIGHT.get(k, 8) * n for k, n in cues.items())
        if main and any(CUE_TO_CAT.get(c) == main for c in cues):
            w += 12                                   # quotes about the clip's main event first
        w += 6 if s.rstrip().endswith(('!', '?!', '!?')) else 0
        w += 3 if s.rstrip().endswith('?') else 0
        w -= max(0, len(words) - 8)
        cand.append((w, s))
    cand.sort(key=lambda x: -x[0])
    out = []
    for _, s in cand:
        q = _clean_quote(s)
        if q and q.upper() not in {o.upper() for o in out}:
            out.append(q)
    return out


def _fit_title(text, emojis):
    text = re.sub(r'\s+', ' ', EMOJI_RX.sub('', text)).strip().upper()
    tail = (' ' + ''.join(emojis)) if emojis else ''
    limit = TITLE_MAX - len(tail)
    if len(text) > limit:
        text = text[:limit].rsplit(' ', 1)[0].rstrip(',;:-–— ')
    return (text + tail).strip()


def _hashtag(word):
    w = re.sub(r'[^\w]', '', word or '', flags=re.U)
    return ('#' + w.lower()) if w else ''


def _hashtags(ctx, extra=()):
    ru = ctx['lang'] == 'ru'
    tags = ['#shorts']
    if ctx.get('streamer'):
        tags.append(_hashtag(ctx['streamer']))
    if ctx['gaming']:
        tags += ['#игры', '#стрим'] if ru else ['#gaming', '#stream']
    for c in ctx['cats'][:3]:
        tags += CATS[c][3] if ru else CATS[c][4]
    tags += list(extra)
    tags += [_hashtag(k) for k in ctx['keywords'][:2]]
    out = []
    for t in tags:
        if t and len(t) > 2 and t.lower() not in {o.lower() for o in out}:
            out.append(t)
    return out[:HASHTAGS_MAX]


def _summary(ctx):
    ru = ctx['lang'] == 'ru'
    parts = moments.explain_cues({k: 1 for k in ctx['cues']}, ctx['lang'])
    if ctx['audio'].get('laughter') and not ctx['cues'].get('laugh_text'):
        parts.append('смех' if ru else 'laughter')
    if ctx['audio'].get('scream'):
        parts.append('громкая реакция' if ru else 'a loud reaction')
    if ctx['screen'].get('fail_screen'):
        parts.append('экран смерти' if ru else 'a death screen')
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return seen[:3]


# ------------------------------------------------------------------ local generator
def local_candidates(ctx):
    ru = ctx['lang'] == 'ru'
    em = _emojis(ctx)
    tag = CATS[ctx['cats'][0]][1 if ru else 2] if ctx['cats'] else None
    who = ctx.get('streamer')
    quotes = _quotes(ctx)
    titles = []
    for q in quotes[:3]:
        if who:
            titles.append(f'{who}: «{q}»' if ru else f'{who}: "{q}"')
        titles.append(f'«{q}»' if ru else f'"{q}"')
        if tag:
            titles.append(f'{tag}: «{q}»' if ru else f'{tag}: "{q}"')
    kws = ctx['keywords']
    if tag and kws:
        titles.append(f'{tag}' + (f' {who}' if who else '') + ': ' + ' И '.join(k for k in kws[:2]) if ru
                      else f'{tag}' + (f' {who}' if who else '') + ': ' + ' AND '.join(kws[:2]))
    if not titles:
        base = ' '.join(ctx['text'].split()[:8]) or ('МОМЕНТ СО СТРИМА' if ru else 'STREAM MOMENT')
        titles.append(_clean_quote(base))
    out = []
    for t in titles:
        ft = _fit_title(t, em)
        if ft not in out:
            out.append(ft)
    return out


def local_description(ctx, quote_idx=0):
    ru = ctx['lang'] == 'ru'
    what = _summary(ctx)
    lead = (('Игровой момент' if ctx['gaming'] else 'Момент из видео') if ru else
            ('Gaming moment' if ctx['gaming'] else 'A moment from the video'))
    if ctx.get('streamer'):
        lead = (f'{ctx["streamer"]} на стриме' if ru else f'{ctx["streamer"]} on stream') if ctx['gaming'] else ctx['streamer']
    body = lead + (': ' + ', '.join(what) if what else '') + '.'
    quotes = _quotes(ctx)
    if quotes:
        q = quotes[min(quote_idx, len(quotes) - 1)]
        body += f' «{q}»' if ru else f' "{q}"'
    body += ' Досмотри до конца!' if ru else ' Watch till the end!'
    return body, _hashtags(ctx)


def generate_local(ctx, variant=0, avoid=()):
    cands = local_candidates(ctx)
    fresh = [t for t in cands if t not in set(avoid)] or cands
    title = fresh[variant % len(fresh)]
    desc, tags = local_description(ctx, variant)
    return {'title': title, 'description': desc, 'hashtags': tags}


# ------------------------------------------------------------------ AI generator
SYSTEM = (
    'You write the title and description for one YouTube Shorts clip cut from a gaming/stream video. '
    'Use ONLY facts from the transcript and the detected events you are given; never invent people, names, numbers, '
    'places or events that are not there. Be catchy and funny when the moment is funny, but truthful. '
    'Title: ALL CAPS, at most 80 characters, ends with 1-3 fitting emoji. Description: 1-2 short sentences. '
    'Hashtags: 4-8, first one #shorts. Write everything in the requested language. '
    'Answer with JSON only: {"title": "...", "description": "...", "hashtags": ["#shorts", ...]}')


def _prompt(ctx, avoid):
    ev = []
    for k, n in ctx['audio'].items():
        ev.append(f'audio:{k} x{n}')
    for k, n in ctx['screen'].items():
        ev.append(f'screen:{k} x{n}')
    for k, n in ctx['cues'].items():
        ev.append(f'speech:{k} x{n}')
    lines = [f'Language: {"Russian" if ctx["lang"] == "ru" else "English"}',
             f'Genre: {"gaming stream" if ctx["gaming"] else "video"}',
             f'Detected events: {", ".join(ev) or "none"}',
             f'Allowed person names: {ctx["streamer"] or "none - do not name anybody"}',
             f'Clip length: {round(ctx["duration"] or 0)} s',
             'Transcript of the clip:', ctx['text'][:1500]]
    if avoid:
        lines.append('Do NOT repeat these previous titles: ' + ' | '.join(avoid))
    return '\n'.join(lines)


def validate_ai(raw, ctx, avoid=()):
    """Cleaned {'title','description','hashtags'} or None if the answer breaks a rule."""
    if not isinstance(raw, dict):
        return None
    title = str(raw.get('title') or '').strip().strip('"«»')
    desc = re.sub(r'\s+', ' ', str(raw.get('description') or '')).strip()
    if len(title) < 6 or len(desc) < 10:
        return None
    letters = re.sub(r'[^\w]|[\d_]', '', EMOJI_RX.sub('', title), flags=re.U)
    if not letters:
        return None
    if tx.detect_lang(letters) != ctx['lang']:
        return None                          # wrong language
    allowed = {ctx['streamer'].lower()} if ctx.get('streamer') else set()
    for name in find_known_names(title + ' ' + desc):
        if name.lower() not in allowed:
            return None                      # a real person who is not verified for this video
    spoken = set(re.findall(r'\d+', ctx['text']))
    if any(d not in spoken for d in re.findall(r'\d+', title)):
        return None                          # numbers must come from the speech
    em = EMOJI_RX.findall(title)[:3] or _emojis(ctx, 1)
    title = _fit_title(title, em)
    if title in set(avoid):
        return None
    tags = raw.get('hashtags') or []
    if isinstance(tags, str):
        tags = tags.split()
    tags = [t if str(t).startswith('#') else '#' + str(t) for t in tags if isinstance(t, str)]
    tags = [_hashtag(t[1:]) for t in tags]
    tags = [t for t in tags if t and len(t) > 2]
    if '#shorts' not in [t.lower() for t in tags]:
        tags.insert(0, '#shorts')
    seen, clean = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            clean.append(t)
    return {'title': title, 'description': desc[:300], 'hashtags': clean[:HASHTAGS_MAX]}


def generate(ctx, variant=0, avoid=(), use_ai=True):
    """-> stored dict: title, description, hashtags, text (description + hashtags), provider, variant, key..."""
    avoid = [a for a in avoid if a][-6:]
    res, provider = None, 'local'
    if use_ai and ai_provider.available():
        raw, info = ai_provider.complete_json(SYSTEM, _prompt(ctx, avoid), temperature=0.7 + 0.1 * min(variant, 3))
        res = validate_ai(raw, ctx, avoid) if raw is not None else None
        if res:
            provider = info
        else:
            log.info('AI title rejected or unavailable (%s); using local generator', info if raw is None else 'invalid')
    if not res:
        res = generate_local(ctx, variant, avoid)
    full = res['description'] + '\n\n' + ' '.join(res['hashtags'])
    return {'title': res['title'], 'description': res['description'], 'hashtags': res['hashtags'], 'text': full,
            'lang': ctx['lang'], 'provider': provider, 'variant': variant, 'key': context_key(ctx),
            'streamer': ctx.get('streamer'), 'history': (list(avoid) + [res['title']])[-6:],
            'generated_at': int(time.time()), 'version': VERSION}


# ------------------------------------------------------------------ project / clip glue
_profiles = {}


def _cached_profiles(fp):
    """Audio + screen profiles from the analysis cache (loaded once per process and fingerprint)."""
    if not fp:
        return None, None
    if fp in _profiles:
        return _profiles[fp]
    from . import analysis_cache as ac, audio_analysis, motion_analysis
    d = ac.cache_dir(fp, create=False)
    audio = screen = None
    if d and d.exists():
        try:
            audio = audio_analysis.from_arrays(ac.load_npz(d / 'audio.npz'))
        except Exception as exc:
            log.warning('cached audio profile unreadable: %s', exc)
        try:
            screen = motion_analysis.screen_from_arrays(ac.load_npz(d / 'screen.npz'))
        except Exception as exc:
            log.warning('cached screen scan unreadable: %s', exc)
    if len(_profiles) > 4:
        _profiles.clear()
    _profiles[fp] = (audio, screen)
    return audio, screen


def context_for_clip(project, clip, text=None):
    analysis = project.analysis or {}
    audio, screen = _cached_profiles(project.source_fingerprint)
    a_ev = audio.events_in(clip.start, clip.end) if audio is not None else []
    s_ev = screen.events_in(clip.start, clip.end) if screen is not None else []
    info = clip.render_info or {}
    types = [s.get('type') for s in info.get('segments', [])]
    lang = (project.transcript_meta or {}).get('detected_language') or analysis.get('language')
    opts = project.options or {}
    who = verified_streamer(opts.get('streamer', ''), analysis.get('source_meta'))
    return build_context(text if text is not None else clip.text, lang, a_ev, s_ev, types, clip.scores, who,
                         clip.end - clip.start)


def ensure_for_clip(project, clip, force=False, use_ai=True):
    """Generate (or reuse) clip.meta['short']; never raises. Returns the stored dict (or None)."""
    try:
        ctx = context_for_clip(project, clip)
        old = (clip.meta or {}).get('short') or {}
        if old and not force and old.get('key') == context_key(ctx) and old.get('version') == VERSION:
            return old
        variant = (old.get('variant', -1) + 1) if force else 0
        new = generate(ctx, variant=variant, avoid=old.get('history', []) if force else (), use_ai=use_ai)
        meta = dict(clip.meta or {})
        meta['short'] = new
        clip.meta = meta
        return new
    except Exception:
        log.exception('shorts title/description failed for clip %s', getattr(clip, 'pk', None))
        return None
