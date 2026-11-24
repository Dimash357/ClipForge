"""AI Visual & Voice Engine (3C): images (ComfyUI) and voices (local TTS) for a storyboard (3B).

Phases run one after another so heavy models never share the 8 GB card: [free the local LLM] -> character sheets ->
frames (ComfyUI, VRAM lease) -> ComfyUI /free -> voices (TTS) -> exact word timings (Whisper) -> scene timings
updated from the real audio. Every result is saved right away with a content key (prompt, negative, seed, model,
size, references / text, voice), so an interrupted job continues where it stopped and unchanged items are never
generated twice. Hand-made replacements (uploads) are kept until the user asks for a new version.

Files: MEDIA_ROOT/generated/scripts/<script>/{characters,frames,voice}/...; the storyboard keeps
    character['image'], frame['assets']['image'], storyboard['voice'][scene_id][line] =
    {path, key, provider, model, seed, width, height | duration, words, start, manual, created, error}
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from django.conf import settings

log = logging.getLogger('studio.genmedia')

DEFAULTS = {
    'comfy_url': 'http://127.0.0.1:8188', 'image_model': 'sdxl', 'checkpoint': '', 'steps': 28, 'cfg': 6.0,
    'sampler': 'dpmpp_2m', 'scheduler': 'karras', 'ip_adapter': True, 'ip_preset': 'PLUS (high strength)',
    'ip_weight': 0.7, 'tts_provider': 'piper', 'piper_dir': '', 'tts_url': '', 'tts_model': '', 'tts_voices': '',
    'tts_speed': 1.0, 'voice_ru': 'ru_RU-dmitri-medium', 'voice_en': 'en_US-ryan-high', 'align': 'whisper',
}
RECOMMENDED = {
    'images': [
        {'name': 'Juggernaut XL v9 (SDXL)', 'file': 'juggernautXL_v9Rdphoto2Lightning.safetensors', 'size': '6.6 ГБ',
         'for': 'faceless, истории — фотореализм', 'vram': '≈6.5 ГБ (768×1344)'},
        {'name': 'SDXL base 1.0', 'file': 'sd_xl_base_1.0.safetensors', 'size': '6.9 ГБ', 'for': 'универсально',
         'vram': '≈6.5 ГБ'},
        {'name': 'DreamShaper 8 (SD 1.5)', 'file': 'dreamshaper_8.safetensors', 'size': '2.1 ГБ',
         'for': 'мультфильмы, иллюстрации; быстрее и легче', 'vram': '≈3.5 ГБ (512×912)'},
        {'name': 'IP-Adapter Plus (SDXL) + CLIP-ViT-H', 'file': 'ip-adapter-plus_sdxl_vit-h.safetensors + '
         'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', 'size': '0.8 + 2.5 ГБ', 'for': 'постоянная внешность героев',
         'vram': '+≈1 ГБ'},
    ],
    'voices': [
        {'name': 'ru_RU-dmitri-medium', 'for': 'рассказчик, мужской', 'size': '63 МБ'},
        {'name': 'ru_RU-irina-medium', 'for': 'женский', 'size': '63 МБ'},
        {'name': 'ru_RU-denis-medium / ru_RU-ruslan-medium', 'for': 'персонажи', 'size': '63 МБ'},
        {'name': 'en_US-ryan-high / en_US-amy-medium', 'for': 'English', 'size': '63–120 МБ'},
    ],
}


# ------------------------------------------------------------------ settings (clipforge_settings.json -> "media")
def _settings_path():
    from .resources import manager
    return manager()._settings_file()


def config():
    cfg = dict(DEFAULTS, piper_dir=str(Path(settings.BASE_DIR) / 'models' / 'piper'))
    try:
        cfg.update(json.loads(_settings_path().read_text(encoding='utf-8')).get('media') or {})
    except Exception:
        pass
    return cfg


def save_config(changes):
    allowed = {k: type(v) for k, v in DEFAULTS.items()}
    path = _settings_path()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        data = {}
    cur = dict(data.get('media') or {})
    for k, v in (changes or {}).items():
        if k not in allowed:
            continue
        try:
            cur[k] = (v in (True, 'true', 'on', '1', 1)) if allowed[k] is bool else allowed[k](v)
        except (TypeError, ValueError):
            continue
    if cur.get('image_model') not in ('sdxl', 'sd15'):
        cur['image_model'] = 'sdxl'
    if cur.get('tts_provider') not in ('piper', 'openai_tts', 'off'):
        cur['tts_provider'] = 'piper'
    data['media'] = cur
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)
    return config()


def client(cfg=None):
    from . import comfy
    cfg = cfg or config()
    return comfy.Client(cfg['comfy_url'])


def status(cfg=None):
    """Availability of everything 3C needs, for the settings page."""
    from . import tts
    cfg = cfg or config()
    c = client(cfg).status()
    c['checkpoint_ok'] = bool(cfg['checkpoint']) and cfg['checkpoint'] in c.get('checkpoints', [])
    return {'comfy': c, 'tts': tts.status(cfg), 'config': cfg, 'recommended': RECOMMENDED}


# ------------------------------------------------------------------ files and keys
def root(script_id):
    return Path(settings.MEDIA_ROOT) / 'generated' / 'scripts' / str(script_id)


def rel(path):
    return Path(path).relative_to(settings.MEDIA_ROOT).as_posix()


def key_of(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def valid(asset, key=None):
    """An asset that exists on disk (and matches `key` unless it was made by hand)."""
    if not asset or asset.get('error') or not asset.get('path'):
        return False
    if not (Path(settings.MEDIA_ROOT) / asset['path']).is_file():
        return False
    return key is None or asset.get('manual') or asset.get('key') == key


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ images
def image_key(cfg, prompt, negative, seed, refs=()):
    from .comfy import size_for
    return key_of('img', cfg['image_model'], cfg['checkpoint'], size_for(cfg['image_model'], cfg.get('aspect', '9:16')),
                  cfg['steps'], cfg['cfg'],
                  cfg['sampler'], cfg['scheduler'], prompt, negative, seed,
                  [(r, cfg['ip_preset'], cfg['ip_weight']) for r in refs])


def _render_image(cl, cfg, prompt, negative, seed, dest, refs=(), cancel=None):
    """One ComfyUI run (OOM: free VRAM, one retry smaller). -> (width, height)."""
    from . import comfy, resources
    w, h = comfy.size_for(cfg['image_model'], cfg.get('aspect', '9:16'))
    for attempt in (0, 1):
        g = comfy.Client.workflow(checkpoint=cfg['checkpoint'], prompt=prompt, negative=negative, seed=seed,
                                  width=w, height=h, steps=cfg['steps'], cfg=cfg['cfg'], sampler=cfg['sampler'],
                                  scheduler=cfg['scheduler'], prefix=f'clipforge_{dest.stem}', refs=refs,
                                  ip_preset=cfg['ip_preset'], ip_weight=cfg['ip_weight'])
        try:
            png = cl.run(g, cancel=cancel)
            break
        except comfy.ComfyOOM as exc:
            cl.free()
            resources.manager().on_oom('ComfyUI', exc)
            if attempt:
                raise
            w, h = comfy.size_for(cfg['image_model'], cfg.get('aspect', '9:16'), smaller=True)
            log.warning('ComfyUI out of memory; retrying at %dx%d', w, h)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.part.png')
    tmp.write_bytes(png)
    from PIL import Image
    with Image.open(tmp) as im:
        size = im.size
    os.replace(tmp, dest)
    return size


def _asset(path, key, cfg, seed, size, **extra):
    return dict({'path': rel(path), 'key': key, 'provider': 'comfyui', 'model': cfg['checkpoint'], 'seed': seed,
                 'width': size[0], 'height': size[1], 'manual': False, 'created': int(time.time())}, **extra)


def preflight_images(cfg, cl):
    st = cl.status()
    if not st['ok']:
        raise RuntimeError(f'ComfyUI недоступен по адресу {cfg["comfy_url"]}: {st.get("error")}. Запустите ComfyUI '
                           f'(см. «Генерация» → инструкция) и проверьте адрес.')
    if not cfg['checkpoint'] or cfg['checkpoint'] not in st['checkpoints']:
        raise RuntimeError(f'Модель «{cfg["checkpoint"] or "не выбрана"}» не найдена в ComfyUI '
                           f'(models/checkpoints). Доступны: {", ".join(st["checkpoints"][:8]) or "ничего"}.')
    return st


def gen_images(doc, sid, cfg, frame_ids=None, characters=None, force=False, cancel=None, progress=None, client_=None):
    """Character sheets, then frames. Writes into doc (the caller saves it after every item via `progress`).
    -> {'done', 'cached', 'errors', 'ip_adapter'}."""
    from . import ai_provider, resources
    from .comfy import VRAM_MB
    sb = doc['storyboard']
    cl = client_ or client(cfg)
    st = preflight_images(cfg, cl)
    ipa = bool(cfg['ip_adapter'] and st.get('ip_adapter'))
    stats = {'done': 0, 'cached': 0, 'errors': [], 'ip_adapter': ipa,
             'ip_note': '' if ipa or not cfg['ip_adapter'] else 'IP-Adapter не установлен в ComfyUI: внешность '
                                                                'держится только описанием и seed'}
    frames = [(s, f) for s in doc['scenes'] for f in sb['scenes'].get(s['id'], {}).get('frames', [])
              if (frame_ids is None and characters is None) or (frame_ids is not None and f['id'] in frame_ids)]
    chars = [cid for cid in sb['characters'] if (characters is None and frame_ids is None) or
             (characters and cid in characters) or (frame_ids is not None and ipa and
                                                   any(cid in f['characters'] for _, f in frames))]
    ai_provider.unload_local()                     # a local LLM would hold VRAM ComfyUI needs
    mgr = resources.manager()
    vram = VRAM_MB[cfg['image_model']] if mgr.facts()['vram_total'] else 0
    total = len(chars) + len(frames)
    n = 0
    uploaded = {}
    with mgr.acquire(label=f'ComfyUI {cfg["image_model"].upper()}', vram_mb=min(vram, mgr.capacity()['vram_mb']),
                     cpu=1, timeout=3600, cancel=cancel):
        try:
            for cid in chars:
                ch = sb['characters'][cid]
                cfg_ch = dict(cfg, aspect='9:16')          # reference sheets keep one size for every format
                key = image_key(cfg_ch, ch['ref_prompt'], sb['style'].get('negative', ''), ch['seed'])
                n += 1
                force_ch = force and (characters is not None or frame_ids is None)   # a frame's redo keeps sheets
                if not force_ch and valid(ch.get('image'), key):
                    stats['cached'] += 1
                    continue
                dest = root(sid) / 'characters' / f'{cid}_{key}.png'
                try:
                    size = _render_image(cl, cfg_ch, ch['ref_prompt'], sb['style'].get('negative', ''), ch['seed'], dest,
                                         cancel=cancel)
                    ch['image'] = _asset(dest, key, cfg, ch['seed'], size)
                    stats['done'] += 1
                except Exception as exc:
                    _fail(exc, stats, cid)
                    ch['image'] = {'error': str(exc)[:300]}
                if progress:
                    progress(n / max(1, total), f'Лист персонажа {ch["name"]}')
            for s, f in frames:
                n += 1
                refs = []
                if ipa:
                    for cid in f['characters']:
                        img = sb['characters'].get(cid, {}).get('image')
                        if valid(img):
                            p = Path(settings.MEDIA_ROOT) / img['path']
                            if cid not in uploaded:
                                uploaded[cid] = cl.upload(p, f'clipforge_ref_{cid}_{_file_hash(p)}.png')
                            refs.append(uploaded[cid])
                key = image_key(cfg, f['prompt'], f['negative'], f['seed'], [r for r in refs])
                cur = f.get('assets', {}).get('image')
                if not force and (valid(cur, key) or (cur or {}).get('manual') and valid(cur)):
                    stats['cached'] += 1
                    continue
                dest = root(sid) / 'frames' / f'{f["id"]}_{key}.png'
                try:
                    size = _render_image(cl, cfg, f['prompt'], f['negative'], f['seed'], dest, refs, cancel=cancel)
                    f.setdefault('assets', {})['image'] = _asset(dest, key, cfg, f['seed'], size,
                                                                 references=len(refs))
                    stats['done'] += 1
                except Exception as exc:
                    _fail(exc, stats, f['id'])
                    f.setdefault('assets', {})['image'] = dict(cur or {}, error=str(exc)[:300])
                if progress:
                    progress(n / max(1, total), f'Кадр {n - len(chars)} из {len(frames)}')
        finally:
            cl.free()                                  # sequential loading: ComfyUI gives the VRAM back
    return stats


def _fail(exc, stats, what):
    from .procutil import JobCancelled
    if isinstance(exc, JobCancelled):
        raise exc
    log.warning('generation of %s failed: %s', what, str(exc)[:300])
    stats['errors'].append({'id': what, 'error': str(exc)[:300]})


# ------------------------------------------------------------------ voices
def voice_for(doc, speaker, cfg):
    sb = doc['storyboard']
    ch = sb['characters'].get(speaker) or {}
    if ch.get('tts_voice'):
        return ch['tts_voice']
    lang = doc.get('language', 'ru')
    return cfg.get('voice_en') if lang == 'en' else cfg.get('voice_ru')


def voice_key(cfg, text, voice, lang):
    return key_of('tts', cfg['tts_provider'], cfg.get('tts_model'), cfg.get('tts_speed'), voice, text, lang)


def gen_voices(doc, sid, cfg, scene_ids=None, lines=None, force=False, cancel=None, progress=None, post=None):
    """Synthesize every line (or the given scene ids / (scene, line) pairs), align words, fit scene timings."""
    from . import resources, tts
    from .procutil import JobCancelled
    sb = doc['storyboard']
    sb.setdefault('voice', {})
    todo = [(s, li, line) for s in doc['scenes'] for li, line in enumerate(s['lines'])
            if (scene_ids is None or s['id'] in scene_ids) and (lines is None or [s['id'], li] in lines)]
    stats = {'done': 0, 'cached': 0, 'errors': []}
    for n, (s, li, line) in enumerate(todo, 1):
        if cancel and cancel():
            raise JobCancelled()
        voice = voice_for(doc, line['speaker'], cfg)
        key = voice_key(cfg, line['text'], voice, doc['language'])
        slot = sb['voice'].setdefault(s['id'], {})
        cur = slot.get(str(li))
        if not force and (valid(cur, key) or (cur or {}).get('manual') and valid(cur)):
            stats['cached'] += 1
            continue
        dest = root(sid) / 'voice' / f'{s["id"]}_{li}_{key}.wav'
        try:
            with resources.cpu_lease('TTS', 2):
                dur = tts.synthesize(line['text'], voice, dest, cfg, post=post)
            words = tts.align(dest, line['text'], doc['language'], use_whisper=cfg.get('align') == 'whisper')
            slot[str(li)] = {'path': rel(dest), 'key': key, 'provider': cfg['tts_provider'], 'voice': voice,
                             'duration': round(dur, 3), 'words': words, 'manual': False, 'created': int(time.time())}
            stats['done'] += 1
        except JobCancelled:
            raise
        except Exception as exc:
            _fail(exc, stats, f'{s["id"]}:{li}')
            slot[str(li)] = dict(cur or {}, error=str(exc)[:300])
        if progress:
            progress(n / max(1, len(todo)), f'Реплика {n} из {len(todo)}')
    fit_timings(doc)
    return stats


GAP, LEAD, TAIL = 0.15, 0.2, 0.35


def fit_timings(doc):
    """Exact timings from the real audio: line starts inside their scene, scenes long enough for their voice,
    frames re-spread over the new scene length, script starts recomputed (nothing is squeezed below speech)."""
    from .storyboard import _retime
    sb = doc['storyboard']
    t = 0.0
    for s in doc['scenes']:
        slot = (sb.get('voice') or {}).get(s['id']) or {}
        pos = LEAD
        for li in range(len(s['lines'])):
            v = slot.get(str(li))
            if v and v.get('duration') and not v.get('error'):
                v['start'] = round(pos, 3)
                pos += v['duration'] + GAP
        need = pos - GAP + TAIL if pos > LEAD else 0.0
        if need > s['duration']:
            s['duration'] = round(need, 2)
        s['start'] = round(t, 2)
        t += s['duration']
        ssb = sb['scenes'].get(s['id'])
        if ssb and ssb.get('frames'):
            _retime(ssb['frames'], s['duration'])
    doc['total'] = round(t, 2)
    return doc


# ------------------------------------------------------------------ manual replacements
def replace_image(doc, sid, frame_id, upload):
    """A user's own picture for a frame (any format PIL reads) -> kept until a new version is asked for."""
    from PIL import Image
    from .storyboard import _find
    _, ssb, i = _find(doc, frame_id)
    f = ssb['frames'][i]
    dest = root(sid) / 'frames' / f'{f["id"]}_manual_{int(time.time())}.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(upload) as im:
        im = im.convert('RGB')
        im.save(dest, 'PNG')
        size = im.size
    f.setdefault('assets', {})['image'] = {'path': rel(dest), 'key': 'manual', 'provider': 'upload', 'manual': True,
                                           'width': size[0], 'height': size[1], 'created': int(time.time())}
    return doc


def replace_voice(doc, sid, scene_id, li, upload_path, cfg=None):
    from . import tts
    cfg = cfg or config()
    s = next((x for x in doc['scenes'] if x['id'] == scene_id), None)
    if s is None or not 0 <= int(li) < len(s['lines']):
        raise KeyError(scene_id)
    dest = root(sid) / 'voice' / f'{scene_id}_{li}_manual_{int(time.time())}.wav'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = tts.import_audio(upload_path, dest)
    words = tts.align(dest, s['lines'][int(li)]['text'], doc['language'], use_whisper=cfg.get('align') == 'whisper')
    doc['storyboard'].setdefault('voice', {}).setdefault(scene_id, {})[str(li)] = {
        'path': rel(dest), 'key': 'manual', 'provider': 'upload', 'voice': 'upload', 'duration': round(dur, 3),
        'words': words, 'manual': True, 'created': int(time.time())}
    return fit_timings(doc)


def summary(doc):
    sb = doc.get('storyboard') or {}
    frames = [f for s in (sb.get('scenes') or {}).values() for f in s.get('frames', [])]
    lines = [(s['id'], li) for s in doc.get('scenes', []) for li in range(len(s['lines']))]
    voice = sb.get('voice') or {}
    return {'frames': len(frames), 'images': sum(valid(f.get('assets', {}).get('image')) for f in frames),
            'image_errors': sum(bool((f.get('assets', {}).get('image') or {}).get('error')) for f in frames),
            'lines': len(lines), 'voiced': sum(valid((voice.get(sid) or {}).get(str(li))) for sid, li in lines),
            'characters': len(sb.get('characters') or {}),
            'sheets': sum(valid(c.get('image')) for c in (sb.get('characters') or {}).values())}
