"""AI Storyboard (3B): a script (3A) -> shot-by-shot storyboard ready for image / voice generation (3C).

Stored in Script.data['storyboard'] (version 1):
    {version, provider, note, generated_at,
     style:  {name, look, palette: [#hex], lighting, environment, camera_language, aspect, negative},
     characters: {char_id: {name, appearance, clothing, features, colors, voice, seed, ref_prompt}},
     scenes: {scene_id: {revision, location, frames: [frame...]}}}
    frame = {id, start, duration (scene-relative), shot, composition, action, camera, characters: [ids], line,
             transition, prompt, negative, seed, locked, revision,
             assets: {image: None, voice: None}}            <- 3C writes generated files here
* Character Bible / Style Bible text goes verbatim into every prompt -> the same look in every frame.
* Continuity: each scene knows the previous scene's location and last action; seeds are stable per frame and
  per character; frame ids are stable (they survive reorder, edits and regeneration of other frames).
* Locked frames are never touched by regeneration; a scene rewritten in the script (new revision) is marked
  stale and rebuilt (its locked frames kept).
Text comes from the same local LLM as the script (scripts._ask: VRAM lease, timeouts); without a model the
storyboard is built by rules from the script and says so.
"""
import copy
import hashlib
import logging
import re
import time

from . import scripts as sg
from .timeline.ids import stable_id

log = logging.getLogger('studio.storyboard')

VERSION = 1
SHOTS = ['extreme wide shot', 'wide shot', 'medium shot', 'medium close-up', 'close-up', 'extreme close-up',
         'over-the-shoulder shot', 'low angle shot', 'high angle shot', 'POV shot']
MOVES = ['static', 'slow push-in', 'pull-out', 'pan left', 'pan right', 'tilt up', 'tilt down', 'handheld',
         'dolly forward', 'orbit']
COMPOSITIONS = ['subject centered', 'rule of thirds, subject left', 'rule of thirds, subject right',
                'symmetrical', 'foreground framing', 'subject in lower third, space above for captions']
FRAME_SECONDS = 2.5
MAX_FRAMES = 6
NEG_BASE = ('blurry, low quality, deformed, extra limbs, extra fingers, bad anatomy, watermark, text, logo, '
            'cropped face, inconsistent character, different outfit')
STYLE_BY_FORMAT = {
    'cartoon': {'name': 'cartoon', 'look': '2D cartoon illustration, clean bold outlines, flat cel shading',
                'palette': ['#FF8A3D', '#3DA5FF', '#FFE14D', '#2E2A4F', '#F4F1EA'],
                'lighting': 'soft even daylight, gentle rim light', 'environment': 'simple stylised backgrounds'},
    'faceless': {'name': 'cinematic', 'look': 'cinematic photography, shallow depth of field, high detail',
                 'palette': ['#0E1A2B', '#1F6FEB', '#F2C14E', '#E8E8E8', '#5C677D'],
                 'lighting': 'dramatic contrast lighting', 'environment': 'real-world locations matching the topic'},
    'story': {'name': 'cinematic illustration', 'look': 'digital painting, cinematic composition, rich detail',
              'palette': ['#1B263B', '#E0A458', '#8D99AE', '#2B2D42', '#EDF2F4'],
              'lighting': 'moody volumetric light', 'environment': 'atmospheric settings that follow the story'},
    'education': {'name': 'clean infographic', 'look': 'clean modern 3D render, minimal, educational diagram style',
                  'palette': ['#00B4D8', '#0077B6', '#90E0EF', '#FFFFFF', '#023E8A'],
                  'lighting': 'bright studio lighting', 'environment': 'neutral studio background'},
}


def _s(v, n=400):
    return re.sub(r'\s+', ' ', str(v or '')).strip()[:n]


def _seed(*parts):
    return int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % 2_000_000_000


# ------------------------------------------------------------------ bibles
ASPECT_TEXT = {'9:16': 'vertical 9:16 framing', '16:9': 'horizontal 16:9 cinematic framing', '1:1': 'square 1:1 framing'}


def default_style(doc):
    base = dict(STYLE_BY_FORMAT.get(doc['format'], STYLE_BY_FORMAT['faceless']))
    if doc.get('style'):
        base['look'] = f'{doc["style"]}, {base["look"]}'
    aspect = doc.get('aspect') or '9:16'
    base.update(camera_language=ASPECT_TEXT.get(aspect, ASPECT_TEXT['9:16']) + ', subject readable on a phone',
                aspect=aspect, negative=NEG_BASE)
    return base


def default_character(c, doc):
    return {'name': c['name'], 'appearance': _s(c.get('description') or c['name'], 300),
            'clothing': 'same outfit in every scene', 'features': '', 'colors': '',
            'voice': c.get('voice', ''), 'seed': _seed(doc.get('generated_at'), c['id']),
            'ref_prompt': ''}


def char_text(ch):
    """The fixed description used in every prompt of this character."""
    bits = [ch['name'], ch['appearance'], ch.get('clothing'), ch.get('features'),
            ch.get('colors') and f'colors: {ch["colors"]}']
    return ', '.join(b for b in bits if b)


def style_text(st):
    pal = ', '.join(st.get('palette') or [])
    return ', '.join(b for b in (st.get('look'), pal and f'color palette {pal}', st.get('lighting'),
                                 st.get('camera_language')) if b)


# ------------------------------------------------------------------ prompts (the same text every time -> consistency)
def compose(frame, scene_sb, sb):
    """positive / negative prompt of a frame from its own fields + Style Bible + Character Bible."""
    chars = [sb['characters'][c] for c in frame['characters'] if c in sb['characters']]
    env = scene_sb.get('location') or sb['style'].get('environment', '')
    pos = ', '.join(b for b in (
        frame['shot'], frame['action'], f'setting: {env}' if env else '', frame['composition'],
        '; '.join(char_text(c) for c in chars), style_text(sb['style']),
        ASPECT_TEXT.get(sb['style'].get('aspect'), 'vertical 9:16').split(' framing')[0]) if b)
    neg = ', '.join(b for b in (sb['style'].get('negative') or NEG_BASE,) if b)
    return _s(pos, 1500), _s(neg, 600)


def ref_prompt(ch, sb):
    return _s(f'character sheet, {char_text(ch)}, front view, side view, neutral pose, plain background, '
              f'{style_text(sb["style"])}', 1200)


# ------------------------------------------------------------------ frames by rules (no model)
def _speakers(scene):
    return sorted({l['speaker'] for l in scene['lines'] if l['speaker'] != 'narrator'})


def rule_frames(scene, k_scene, prev=None):
    n = max(1, min(MAX_FRAMES, round(scene['duration'] / FRAME_SECONDS)))
    lines = scene['lines'] or [{'speaker': 'narrator', 'text': ''}]
    who = _speakers(scene)
    part = scene['part']
    plan_shots = {'hook': ['close-up', 'extreme close-up', 'medium shot'],
                  'development': ['wide shot', 'medium shot', 'over-the-shoulder shot', 'medium close-up'],
                  'climax': ['low angle shot', 'close-up', 'extreme close-up', 'POV shot'],
                  'ending': ['medium shot', 'wide shot', 'extreme wide shot']}[part]
    plan_moves = {'hook': ['slow push-in', 'static'], 'development': ['pan right', 'static', 'dolly forward'],
                  'climax': ['handheld', 'slow push-in', 'orbit'], 'ending': ['pull-out', 'static']}[part]
    frames = []
    t = 0.0
    for k in range(n):
        d = round(scene['duration'] / n, 2)
        line = lines[min(len(lines) - 1, k * len(lines) // n)]
        sp = [line['speaker']] if line['speaker'] != 'narrator' else who[:1]
        frames.append({'shot': plan_shots[k % len(plan_shots)], 'camera': plan_moves[k % len(plan_moves)],
                       'composition': COMPOSITIONS[(k_scene + k) % len(COMPOSITIONS)],
                       'action': _s(f'{scene["visual"]}' + (f', {line["text"][:120]}' if k else ''), 300),
                       'characters': sp, 'line': lines.index(line), 'start': round(t, 2), 'duration': d,
                       'transition': 'cut'})
        t += d
    return frames


# ------------------------------------------------------------------ LLM
SYSTEM = ('You are a storyboard artist for vertical short videos. You turn script scenes into shot lists that an image '
          'model can draw: shot size, composition, the action in the frame, camera movement, which characters are in '
          'it. Keep characters, outfits, places and style consistent between frames and scenes. Answer with ONE JSON '
          'object only.')


def prompt_bibles(doc):
    chars = '; '.join(f'{c["name"]}: {c["description"]}' for c in doc['characters']) or 'no recurring characters'
    return (f'Script "{doc["title"]}" ({sg.FORMATS[doc["format"]]}, style: {doc.get("style") or "free"}). '
            f'Characters: {chars}. Scenes: ' + ' | '.join(f'{s["title"]}: {s["visual"]}' for s in doc['scenes'])[:2500] +
            '\nDefine a Style Bible and a Character Bible (English, image-model friendly). JSON: {"style": {"look": str, '
            '"palette": [hex], "lighting": str, "environment": str, "negative": str}, "characters": [{"name": str, '
            '"appearance": str, "clothing": str, "features": str, "colors": str}]}')


def prompt_frames(doc, scene, sb, prev_summary):
    chars = '; '.join(f'{c["name"]}' for c in doc['characters'])
    n = max(1, min(MAX_FRAMES, round(scene['duration'] / FRAME_SECONDS)))
    lines = ' / '.join(f'{l["speaker"]}: {l["text"]}' for l in scene['lines'])
    return (f'Style: {style_text(sb["style"])}. Characters available: {chars or "none"}.\nPrevious scene ended with: '
            f'{prev_summary or "this is the first scene"}.\nScene ({scene["part"]}, {scene["duration"]:.1f} s): '
            f'{scene["visual"]}. Spoken: {lines}.\nSplit it into {n} frames (they share the {scene["duration"]:.1f} s). '
            f'Shot sizes from {SHOTS}; camera moves from {MOVES}. JSON: {{"location": str, "frames": [{{"shot": str, '
            f'"composition": str, "action": str (what we see, English), "camera": str, "characters": [names], '
            f'"duration": number}}]}}')


def _frames_from_answer(raw, scene, doc):
    names = {c['name'].lower(): c['id'] for c in doc['characters']}
    out = []
    for f in (raw or {}).get('frames') or []:
        if not isinstance(f, dict) or not _s(f.get('action')):
            continue
        chars = [names[n.lower()] for n in f.get('characters') or [] if isinstance(n, str) and n.lower() in names]
        try:
            d = float(f.get('duration') or 0)
        except (TypeError, ValueError):
            d = 0.0
        out.append({'shot': _s(f.get('shot'), 60) or 'medium shot', 'composition': _s(f.get('composition'), 120),
                    'action': _s(f.get('action'), 400), 'camera': _s(f.get('camera'), 60) or 'static',
                    'characters': chars, 'duration': d, 'transition': 'cut', 'line': 0})
    return out[:MAX_FRAMES]


def _retime(frames, total):
    if not frames:
        return frames
    want = [f['duration'] if f.get('duration', 0) > 0.3 else total / len(frames) for f in frames]
    k = total / sum(want)
    t = 0.0
    for f, w in zip(frames, want):
        f['duration'] = round(max(0.6, w * k), 2)
        f['start'] = round(t, 2)
        t += f['duration']
    if frames:                                     # rounding: the last frame closes the scene exactly
        frames[-1]['duration'] = round(max(0.6, total - frames[-1]['start']), 2)
    return frames


def _finish(frames, scene, k_scene, sb, scene_sb, doc, keep=(), batch=0):
    """ids, seeds, prompts, timing; locked frames from `keep` stay as they are, in their place.
    New frames get ids of this build (`batch`), so they can never take the id of a kept locked frame."""
    keep = {f['id']: f for f in keep if f.get('locked')}
    out = []
    for i, f in enumerate(frames):
        fid = f.get('id') or stable_id('fr', scene['id'], scene['revision'], batch, i)
        f = dict(f, id=fid, locked=bool(f.get('locked')), revision=int(f.get('revision') or 1),
                 seed=f.get('seed') or _seed(doc.get('generated_at'), fid),
                 assets=f.get('assets') or {'image': None, 'voice': None})
        f.setdefault('line', 0)
        out.append(f)
    for fid, f in keep.items():                     # a locked frame always survives
        if not any(x['id'] == fid for x in out):
            out.insert(min(len(out), int(f.get('_pos', len(out)))), f)
    _retime(out, scene['duration'])
    for f in out:
        if not f['locked']:
            f['prompt'], f['negative'] = compose(f, scene_sb, sb)
        f.pop('_pos', None)
    return out


# ------------------------------------------------------------------ build / regenerate
def build(doc, cancel=None, only=None):
    """Whole storyboard (or `only` the given scene ids). Locked frames and untouched scenes are kept."""
    doc = copy.deepcopy(doc)
    old = doc.get('storyboard') or {}
    sb = {'version': VERSION, 'style': old.get('style') or default_style(doc),
          'characters': dict(old.get('characters') or {}), 'scenes': dict(old.get('scenes') or {})}
    how = old.get('provider') or 'rules'
    notes = []
    if not old.get('style'):
        raw, how = sg._ask(SYSTEM, prompt_bibles(doc), cancel)
        if raw and isinstance(raw.get('style'), dict):
            st = raw['style']
            sb['style'].update({k: _s(st.get(k), 400) for k in ('look', 'lighting', 'environment', 'negative')
                                if _s(st.get(k))})
            pal = [p for p in st.get('palette') or [] if isinstance(p, str) and re.match(r'^#[0-9A-Fa-f]{6}$', p)]
            if pal:
                sb['style']['palette'] = pal[:6]
            by_name = {_s(c.get('name'), 60).lower(): c for c in raw.get('characters') or [] if isinstance(c, dict)}
        else:
            by_name = {}
            if not raw:
                notes.append(f'Библии составлены по правилам ({how}).')
        for c in doc['characters']:
            ch = default_character(c, doc)
            got = by_name.get(c['name'].lower())
            if got:
                ch.update({k: _s(got.get(k), 300) for k in ('appearance', 'clothing', 'features', 'colors')
                           if _s(got.get(k))})
            sb['characters'][c['id']] = ch
    for c in doc['characters']:                     # characters added to the script later
        sb['characters'].setdefault(c['id'], default_character(c, doc))
    for ch in sb['characters'].values():
        ch['ref_prompt'] = ref_prompt(ch, sb)
    prev = None
    used_llm = False
    batch = int(time.time() * 1000)
    for k, scene in enumerate(doc['scenes']):
        cur = sb['scenes'].get(scene['id'])
        stale = cur is None or cur.get('revision') != scene['revision']
        if (only is not None and scene['id'] not in only) or (only is None and cur and not stale and
                                                               old.get('style')):
            prev = cur
            continue
        if cancel and cancel():
            from .procutil import JobCancelled
            raise JobCancelled()
        prev_summary = prev['frames'][-1]['action'] if prev and prev.get('frames') else ''
        raw, how2 = sg._ask(SYSTEM, prompt_frames(doc, scene, sb, prev_summary), cancel)
        frames = _frames_from_answer(raw, scene, doc) if raw else []
        if frames:
            used_llm = True
            how = how2
        else:
            frames = rule_frames(scene, k, prev)
        location = _s((raw or {}).get('location'), 200) or (prev or {}).get('location') or \
            sb['style'].get('environment', '')
        keep = [dict(f, _pos=i) for i, f in enumerate((cur or {}).get('frames') or []) if f.get('locked')]
        scene_sb = {'revision': scene['revision'], 'location': location, 'stale': False}
        scene_sb['frames'] = _finish(frames, scene, k, sb, scene_sb, doc, keep, batch)
        sb['scenes'][scene['id']] = scene_sb
        prev = scene_sb
    for sid in list(sb['scenes']):                  # scenes removed from the script
        if not any(s['id'] == sid for s in doc['scenes']):
            sb['scenes'].pop(sid)
    sb['provider'] = how if used_llm or old.get('provider') not in (None, 'rules') else 'rules'
    sb['note'] = ' '.join(notes) or ('' if used_llm else 'Кадры составлены по правилам: локальная LLM не подключена '
                                     'или не ответила. Поправьте кадры вручную или подключите Ollama.')
    sb['generated_at'] = int(time.time())
    doc['storyboard'] = sb
    return doc


def mark_stale(doc):
    """Scenes changed in the script since their frames were made."""
    sb = doc.get('storyboard') or {}
    out = []
    for s in doc.get('scenes', []):
        cur = (sb.get('scenes') or {}).get(s['id'])
        if cur is None or cur.get('revision') != s['revision']:
            out.append(s['id'])
    return out


def _find(doc, frame_id):
    for s in doc['scenes']:
        for i, f in enumerate(doc['storyboard']['scenes'].get(s['id'], {}).get('frames', [])):
            if f['id'] == frame_id:
                return s, doc['storyboard']['scenes'][s['id']], i
    raise KeyError(frame_id)


def edit_frame(doc, frame_id, changes):
    doc = copy.deepcopy(doc)
    scene, ssb, i = _find(doc, frame_id)
    f = ssb['frames'][i]
    for k in ('shot', 'composition', 'action', 'camera', 'transition'):
        if k in changes:
            f[k] = _s(changes[k], 400)
    if isinstance(changes.get('characters'), list):
        f['characters'] = [c for c in changes['characters'] if c in doc['storyboard']['characters']]
    if 'duration' in changes:
        try:
            f['duration'] = max(0.6, float(changes['duration']))
        except (TypeError, ValueError):
            pass
        _retime(ssb['frames'], scene['duration'])
    if 'locked' in changes:
        f['locked'] = bool(changes['locked'])
    manual = {k: _s(changes[k], 1500) for k in ('prompt', 'negative') if k in changes}
    f['revision'] += 1
    if manual:
        f.update(manual)                            # a hand-written prompt is kept as written
        f['manual_prompt'] = True
    elif not f.get('manual_prompt'):
        f['prompt'], f['negative'] = compose(f, ssb, doc['storyboard'])
    return doc


def move_frame(doc, frame_id, to):
    doc = copy.deepcopy(doc)
    scene, ssb, i = _find(doc, frame_id)
    fr = ssb['frames']
    j = max(0, min(len(fr) - 1, int(to)))
    fr.insert(j, fr.pop(i))
    _retime(fr, scene['duration'])
    return doc


def regenerate_frame(doc, frame_id, instruction='', cancel=None):
    """A new version of one frame (same id and slot); the other frames are untouched."""
    doc = copy.deepcopy(doc)
    scene, ssb, i = _find(doc, frame_id)
    old = ssb['frames'][i]
    sb = doc['storyboard']
    ask = (f'Style: {style_text(sb["style"])}. Scene: {scene["visual"]}. Current frame: {old["shot"]}, '
           f'{old["action"]}, camera {old["camera"]}. Wish: {instruction or "a more striking frame"}. Keep the same '
           f'characters and place. JSON: {{"frames": [{{"shot": str, "composition": str, "action": str, '
           f'"camera": str, "characters": [names]}}]}}')
    raw, how = sg._ask(SYSTEM, ask, cancel)
    got = _frames_from_answer(raw, scene, doc) if raw else []
    if got:
        new = got[0]
    else:                                           # rules: next shot / move / composition
        how = 'rules'
        new = {'shot': SHOTS[(SHOTS.index(old['shot']) + 1) % len(SHOTS)] if old['shot'] in SHOTS else 'close-up',
               'camera': MOVES[(MOVES.index(old['camera']) + 1) % len(MOVES)] if old['camera'] in MOVES else 'static',
               'composition': COMPOSITIONS[(old['revision']) % len(COMPOSITIONS)], 'action': old['action'],
               'characters': old['characters']}
    f = dict(old, **{k: new[k] for k in ('shot', 'composition', 'action', 'camera') if new.get(k)},
             characters=new.get('characters') or old['characters'], revision=old['revision'] + 1, locked=False,
             manual_prompt=False, seed=_seed(old['seed'], old['revision'] + 1),
             assets={'image': None, 'voice': (old.get('assets') or {}).get('voice')})
    f['prompt'], f['negative'] = compose(f, ssb, sb)
    f['history'] = (old.get('history') or [])[-4:] + [{k: old[k] for k in ('shot', 'action', 'camera', 'revision')}]
    ssb['frames'][i] = f
    sb['provider_last'] = how
    return doc


def edit_bibles(doc, style=None, characters=None):
    """Change the Style / Character Bible; every frame without a hand-written prompt is re-composed."""
    doc = copy.deepcopy(doc)
    sb = doc['storyboard']
    if isinstance(style, dict):
        for k in ('look', 'lighting', 'environment', 'camera_language', 'negative', 'name'):
            if k in style:
                sb['style'][k] = _s(style[k], 400)
        if isinstance(style.get('palette'), list):
            sb['style']['palette'] = [p for p in style['palette'] if isinstance(p, str) and
                                      re.match(r'^#[0-9A-Fa-f]{6}$', p)][:6]
    for cid, ch in (characters or {}).items():
        if cid in sb['characters'] and isinstance(ch, dict):
            for k in ('appearance', 'clothing', 'features', 'colors', 'voice'):
                if k in ch:
                    sb['characters'][cid][k] = _s(ch[k], 300)
    for ch in sb['characters'].values():
        ch['ref_prompt'] = ref_prompt(ch, sb)
    for ssb in sb['scenes'].values():
        for f in ssb['frames']:
            if not f.get('manual_prompt') and not f.get('locked'):
                f['prompt'], f['negative'] = compose(f, ssb, sb)
    return doc


# ------------------------------------------------------------------ Timeline IR (inputs for 3C)
CAMERA_ANIM = {'slow push-in': ('zoom', 1.0, 1.12), 'dolly forward': ('zoom', 1.0, 1.15), 'pull-out': ('zoom', 1.15, 1.0),
               'pan left': ('pan_x', 0.08, -0.08), 'pan right': ('pan_x', -0.08, 0.08), 'tilt up': ('pan_y', 0.08, -0.08),
               'tilt down': ('pan_y', -0.08, 0.08)}


def apply_to_timeline(tl, doc):
    """Script timeline (scripts.to_timeline) + frames as image items whose assets are still to be generated
    ('gen:frames/<id>.png', Generation with prompt / negative / seed), character reference sheets and one voice
    asset per scene line. 3C fills exactly these files; nothing here needs them to exist."""
    from .timeline import model as m
    from .timeline.validate import ensure_valid
    sb = doc.get('storyboard')
    if not sb:
        return tl
    out = copy.deepcopy(tl)
    vtrack = next(t for t in out.tracks if t.type == 'video')
    assets, items, voice, exact = [], [], [], []
    for c in out.characters:
        ch = sb['characters'].get(c.id)
        if not ch:
            continue
        aid = stable_id('as', 'ref', c.id)
        img = ch.get('image') or {}
        made = bool(img.get('path') and not img.get('error'))
        assets.append(m.Asset(id=aid, type='image', uri=f'media:{img["path"]}' if made else f'gen:characters/{c.id}.png',
                              width=img.get('width') if made else 1024, height=img.get('height') if made else 1024,
                              generation=m.Generation(provider='image', prompt=ch['ref_prompt'],
                                                      negative=sb['style'].get('negative', ''), seed=ch['seed'],
                                                      params={'kind': 'character_sheet'})))
        c.refs = [aid]
        c.description = char_text(ch)[:400]
        c.seed = ch['seed']
    for s in doc['scenes']:
        ssb = sb['scenes'].get(s['id'])
        if not ssb:
            continue
        for f in ssb['frames']:
            aid = stable_id('as', 'frame', f['id'])
            img = (f.get('assets') or {}).get('image') or {}
            made = bool(img.get('path') and not img.get('error'))
            assets.append(m.Asset(id=aid, type='image', uri=f'media:{img["path"]}' if made else f'gen:frames/{f["id"]}.png',
                                  width=img.get('width') if made else 1080, height=img.get('height') if made else 1920,
                                  generation=m.Generation(provider='image', prompt=f['prompt'], negative=f['negative'],
                                                          seed=f['seed'], revision=f['revision'],
                                                          params={'shot': f['shot'], 'camera': f['camera'],
                                                                  'frame': f['id'], 'characters': f['characters']})))
            anim = []
            if f['camera'] in CAMERA_ANIM:
                prop, a, b = CAMERA_ANIM[f['camera']]
                if prop == 'zoom':
                    anim.append(m.Animation(property='zoom', keyframes=[m.Keyframe(t=0.0, value=a), m.Keyframe(
                        t=round(f['duration'], 3), value=b, easing='ease_in_out')]))
                else:
                    anim += [m.Animation(property='zoom', keyframes=[m.Keyframe(t=0.0, value=1.2)]),
                             m.Animation(property=prop, keyframes=[m.Keyframe(t=0.0, value=a), m.Keyframe(
                                 t=round(f['duration'], 3), value=b, easing='linear')])]
            items.append(m.ImageItem(id=stable_id('it', 'frame', f['id']), start=round(s['start'] + f['start'], 3),
                                     duration=round(f['duration'], 3), scene=s['id'], asset=aid, animations=anim,
                                     ext={'clipforge.frame': {'id': f['id'], 'locked': f['locked']}}))
        t = s['start'] + 0.2
        slot = (sb.get('voice') or {}).get(s['id']) or {}
        for li, line in enumerate(s['lines']):          # voice per line: generated file + exact timing, or the plan
            v = slot.get(str(li)) or {}
            made = bool(v.get('path') and v.get('duration') and not v.get('error'))
            dur = v['duration'] if made else max(0.5, len(line['text'].split()) / sg.WPS.get(doc['language'], 2.4))
            if made and v.get('start') is not None:
                t = s['start'] + v['start']
            ch = sb['characters'].get(line['speaker'])
            aid = stable_id('as', 'voice', s['id'], li, s['revision'])
            assets.append(m.Asset(id=aid, type='audio', uri=f'media:{v["path"]}' if made else
                                  f'gen:voice/{s["id"]}_{li}.wav', duration=dur if made else None,
                                  has_audio=True if made else None,
                                  generation=m.Generation(provider='tts', prompt=line['text'],
                                                          params={'voice': v.get('voice') or (ch or {}).get('tts_voice')
                                                                  or (ch or {}).get('voice') or 'narrator',
                                                                  'speaker': line['speaker'],
                                                                  'language': doc['language']})))
            d = round(min(dur, s['start'] + s['duration'] - t), 3)
            if d > 0.1:
                voice.append(m.AudioItem(id=stable_id('it', 'voice', s['id'], li), start=round(t, 3), duration=d,
                                         scene=s['id'], asset=aid, role='voice'))
                for w in (v.get('words') or []) if made else []:      # captions on the real audio
                    if w['t1'] <= dur + 0.01:
                        exact.append(m.Word(t0=round(t + w['t0'], 3), t1=round(min(t + w['t1'], t + d), 3),
                                            text=w['text']))
            t += dur + 0.15
    vtrack.items = items
    out.assets = assets
    cap = next((t for t in out.tracks if t.type == 'caption'), None)
    by_id = {a.id: a for a in assets}
    voiced = sum(by_id[x.asset].uri.startswith('media:') for x in voice)
    if cap is not None and exact and voiced == len(voice):
        exact.sort(key=lambda w: w.t0)
        cap.items[0].words = [w for w in exact if w.t1 > w.t0]
    if voice:
        out.tracks.append(m.Track(id=stable_id('tr', out.id, 'voice'), type='audio', name='Озвучка (3C)', items=voice))
    out.ext = dict(out.ext, **{'clipforge.storyboard': {'version': VERSION, 'style': sb['style'],
                                                        'provider': sb.get('provider')}})
    return ensure_valid(out)
