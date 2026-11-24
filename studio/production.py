"""Cartoon Studio (3D): one idea -> finished MP4.

idea -> script (3A) -> storyboard (3B) -> images + voice (3C) -> assembly (Timeline IR, 1A) -> render (1B) -> title,
description, hashtags. One job of the queue (1C) runs the stages in order; every stage reads its state from the script
document, so a finished stage is skipped, an interrupted one continues (3C saves every image / line at once) and an
edit (one frame, one line, one scene) only redoes what it touched: images and voices are cached by content, the
renderer caches scene clips, so a changed frame re-renders only its scene.

State: Script.data['production'] = {settings, stages: {key: {status, message, at, info}}, timeline_hash, output,
meta}. Status of a stage is derived from the document (done / pending / stale), stored only for running / error.

Music and sound effects are real audio: tracks from the user's music folder (any format ffmpeg reads) or music
synthesized here with ffmpeg (chord pads + pluck arpeggio by mood); SFX (whoosh, pop, impact, chime) synthesized with
ffmpeg or taken from the user's SFX folder when a file name matches the scene's sfx note. Nothing is a placeholder:
a stage that cannot produce its output fails with the reason.
"""
import hashlib
import json
import logging
import random
import re
import subprocess
import time
from pathlib import Path

from django.conf import settings as dj

from .timeline.ids import stable_id

log = logging.getLogger('studio.production')

STAGES = [('script', 'Сценарий'), ('storyboard', 'Раскадровка'), ('images', 'Изображения'), ('voice', 'Озвучка'),
          ('assemble', 'Монтаж'), ('render', 'Экспорт MP4'), ('meta', 'Название и описание')]
STAGE_KEYS = [k for k, _ in STAGES]
WEIGHTS = {'script': 6, 'storyboard': 6, 'images': 45, 'voice': 10, 'assemble': 3, 'render': 28, 'meta': 2}
CANVAS = {('9:16', '1080'): (1080, 1920), ('9:16', '720'): (720, 1280), ('16:9', '1080'): (1920, 1080),
          ('16:9', '720'): (1280, 720), ('1:1', '1080'): (1080, 1080), ('1:1', '720'): (720, 720)}
DEFAULTS = {'aspect': '9:16', 'resolution': '1080', 'fps': 30, 'quality': 'balanced', 'transition': 'smooth',
            'motion': 'smooth', 'music': 'auto', 'music_volume': 0.18, 'sfx': True, 'captions': True, 'voice': True,
            'music_dir': '', 'sfx_dir': ''}
CHOICES = {'aspect': ('9:16', '16:9', '1:1'), 'resolution': ('1080', '720'), 'fps': (24, 30),
           'quality': ('high', 'balanced', 'small'), 'transition': ('smooth', 'dynamic', 'cut'),
           'motion': ('smooth', 'dynamic', 'off'), 'music': ('auto', 'synth', 'library', 'off')}
AUDIO_EXT = ('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.opus')


class StageError(RuntimeError):
    pass


# ------------------------------------------------------------------ settings / state
def settings_of(doc):
    s = dict(DEFAULTS, **((doc.get('production') or {}).get('settings') or {}))
    return clean_settings(s)


def clean_settings(s):
    out = dict(DEFAULTS)
    for k, v in (s or {}).items():
        if k not in DEFAULTS:
            continue
        if k in CHOICES:
            v = type(CHOICES[k][0])(v) if v not in (None, '') else DEFAULTS[k]
            out[k] = v if v in CHOICES[k] else DEFAULTS[k]
        elif k == 'music_volume':
            out[k] = max(0.0, min(1.0, float(v)))
        elif isinstance(DEFAULTS[k], bool):
            out[k] = v in (True, 'true', '1', 1, 'on')
        else:
            out[k] = str(v)[:400]
    return out


def prod(doc):
    p = doc.setdefault('production', {})
    p.setdefault('settings', dict(DEFAULTS))
    p.setdefault('stages', {})
    return p


def canvas_size(st):
    return CANVAS[(st['aspect'], st['resolution'])]


def _h(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def media_cfg(doc):
    from . import genmedia
    return dict(genmedia.config(), aspect=settings_of(doc)['aspect'])


def stage_states(doc, sc=None):
    """{key: {'status': done|pending|stale|running|error|skipped, 'message', 'info'}} derived from the document."""
    from . import genmedia
    p = (doc or {}).get('production') or {}
    stored = p.get('stages') or {}
    st = settings_of(doc or {})
    out = {}
    has_script = bool((doc or {}).get('scenes'))
    sb = (doc or {}).get('storyboard')
    summ = genmedia.summary(doc) if has_script else {}
    voice_off = not st['voice'] or genmedia.config().get('tts_provider') == 'off'
    out['script'] = 'done' if has_script else 'pending'
    if sb and sb.get('scenes'):
        from .storyboard import mark_stale
        aligned = (sb.get('style') or {}).get('aspect', '9:16') == st['aspect']
        out['storyboard'] = 'stale' if mark_stale(doc) or not aligned else 'done'
    else:
        out['storyboard'] = 'pending'
    out['images'] = ('done' if sb and summ.get('frames') and summ['images'] == summ['frames'] else
                     'pending')
    out['voice'] = 'skipped' if voice_off else ('done' if sb and summ.get('voiced') == summ.get('lines') and
                                                summ.get('lines') else 'pending')
    tl_hash = assembly_key(doc) if out['images'] == 'done' and out['voice'] in ('done', 'skipped') else None
    out['assemble'] = ('done' if tl_hash and p.get('assembly_key') == tl_hash and _exists(p.get('timeline_path'))
                       else 'stale' if p.get('assembly_key') else 'pending')
    out['render'] = ('done' if out['assemble'] == 'done' and p.get('output', {}).get('assembly_key') == tl_hash
                     and _exists(p.get('output', {}).get('path')) else 'stale' if p.get('output') else 'pending')
    out['meta'] = 'done' if p.get('meta', {}).get('title') else 'pending'
    res = {}
    for k in STAGE_KEYS:
        s = stored.get(k) or {}
        status = out[k]
        if s.get('status') in ('running', 'error') and status != 'done':
            status = s['status']
        res[k] = {'status': status, 'message': s.get('message', ''), 'info': s.get('info') or {}, 'at': s.get('at')}
    return res


def _exists(rel):
    return bool(rel) and (Path(dj.MEDIA_ROOT) / rel).is_file()


def assembly_key(doc):
    """Everything the assembled timeline depends on (script timing + texts, frames, media files, settings)."""
    sb = doc.get('storyboard') or {}
    frames = [(f['id'], f['camera'], f['duration'], f['start'], ((f.get('assets') or {}).get('image') or {}).get('key'),
               ((f.get('assets') or {}).get('image') or {}).get('path'))
              for s in (sb.get('scenes') or {}).values() for f in s.get('frames', [])]
    scenes = [(s['id'], s['start'], s['duration'], s['lines'], s.get('on_screen_text'), s.get('sfx'), s.get('mood'),
               s.get('part')) for s in doc.get('scenes', [])]
    voice = sb.get('voice') or {}
    files = [_sig(f[5]) for f in frames] + [_sig(v.get('path')) for slot in voice.values() for v in slot.values()
                                            if isinstance(v, dict)]
    return _h(scenes, frames, voice, files, settings_of(doc), ASSEMBLY_VERSION)


def _sig(rel):
    """A regenerated or replaced file under the same name still changes the assembly."""
    try:
        st = (Path(dj.MEDIA_ROOT) / rel).stat()
        return [rel, st.st_size, st.st_mtime_ns]
    except (OSError, TypeError):
        return [rel, None]


ASSEMBLY_VERSION = 1


# ------------------------------------------------------------------ audio: music / sfx (ffmpeg)
def _ffmpeg(args, dest, timeout=300):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + '.part' + dest.suffix)
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args, str(tmp)], capture_output=True,
                       text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if p.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise StageError(f'ffmpeg: {p.stderr[-300:]}')
    tmp.replace(dest)
    return dest


NOTE = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392.0, 'A': 440.0, 'B': 493.88}
MOODS = {   # chord progression (root, third, fifth as note names; 'm' = minor third), seconds per chord, bpm of pluck
    'playful': (['C', 'G', 'A:m', 'F'], 2.0, 120),
    'calm': (['F', 'C', 'G', 'A:m'], 4.0, 70),
    'emotional': (['A:m', 'F', 'C', 'G'], 3.0, 80),
    'tense': (['A:m', 'A:m', 'F', 'E'], 2.0, 100),
    'epic': (['D:m', 'B', 'F', 'C'], 2.5, 96),
}
FORMAT_MOOD = {'cartoon': 'playful', 'education': 'calm', 'story': 'emotional', 'faceless': 'tense'}


def _chord(name):
    root, _, q = name.partition(':')
    names = 'CDEFGAB'
    f0 = NOTE[root]
    i = names.index(root)
    third = NOTE[names[(i + 2) % 7]] * (2 if (i + 2) >= 7 else 1)
    fifth = NOTE[names[(i + 4) % 7]] * (2 if (i + 4) >= 7 else 1)
    if q == 'm' and root in 'CFG':          # natural minor third where the scale gives a major one
        third *= 0.94387
    if not q and root in 'DEAB':            # major third where the scale gives a minor one
        third *= 1.05946
    if root == 'B':                         # B major -> use Bb-ish power chord (keeps it consonant)
        fifth = f0 * 1.5
    return f0 / 2, third / 2, fifth / 2      # one octave down: warmer pad


def music_mood(doc):
    moods = ' '.join(s.get('mood', '') + ' ' + s.get('music', '') for s in doc.get('scenes', [])).lower()
    for word, mood in (('epic', 'epic'), ('эпич', 'epic'), ('tense', 'tense'), ('dark', 'tense'), ('напряж', 'tense'),
                       ('sad', 'emotional'), ('emotional', 'emotional'), ('груст', 'emotional'), ('calm', 'calm'),
                       ('спокой', 'calm'), ('fun', 'playful'), ('весел', 'playful'), ('playful', 'playful')):
        if word in moods:
            return mood
    return FORMAT_MOOD.get(doc.get('format'), 'calm')


def synth_music(mood, seconds, dest):
    """Chord pad + bass + plucked arpeggio, soft reverb; seamless for `seconds`."""
    chords, L, bpm = MOODS[mood]
    beat = 60.0 / bpm / 2                   # eighth notes
    n = len(chords)
    pads, arps = [], []
    for j, name in enumerate(chords):
        a, b, c = _chord(name)
        sel = f'eq(mod(floor(t/{L}),{n}),{j})'
        pads.append(f'{sel}*(sin(2*PI*{a:.2f}*t)+0.7*sin(2*PI*{b:.2f}*t)+0.7*sin(2*PI*{c:.2f}*t)'
                    f'+0.6*sin(PI*{a:.2f}*t))')
        tone = f'if(eq(mod(floor(t/{beat:.4f}),4),0),{a * 2:.2f},if(eq(mod(floor(t/{beat:.4f}),4),1),{b * 2:.2f},' \
               f'if(eq(mod(floor(t/{beat:.4f}),4),2),{c * 2:.2f},{b * 2:.2f})))'
        arps.append(f'{sel}*sin(2*PI*{tone}*t)')
    env = f'min(1,mod(t,{L})*3)*min(1,({L}-mod(t,{L}))*3)'
    pluck = f'exp(-mod(t,{beat:.4f})*9)'
    expr = f"0.07*({'+'.join(pads)})*{env}+0.09*({'+'.join(arps)})*{pluck}"
    d = max(1.0, float(seconds))
    return _ffmpeg(['-f', 'lavfi', '-i', f"aevalsrc=exprs='{expr}|{expr}':s=48000:d={d + 0.5:.2f}",
                    '-af', f'lowpass=f=4200,aecho=0.8:0.6:60|110:0.25|0.15,afade=t=in:d=1.5,'
                    f'afade=t=out:st={max(0.0, d - 2):.2f}:d=2,atrim=duration={d:.2f}',
                    '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le'], dest)


SFX = {
    'whoosh': ['-f', 'lavfi', '-i', 'anoisesrc=d=0.7:c=pink:a=0.6', '-af',
               'highpass=f=400,lowpass=f=5000,afade=t=in:d=0.4:curve=exp,afade=t=out:st=0.4:d=0.3'],
    'pop': ['-f', 'lavfi', '-i', "aevalsrc=exprs='0.6*sin(2*PI*(500+900*exp(-t*35))*t)*exp(-t*22)':s=48000:d=0.3"],
    'impact': ['-f', 'lavfi', '-i',
               "aevalsrc=exprs='(0.9*sin(2*PI*52*t)+0.4*sin(2*PI*78*t)+0.2*(random(0)-0.5)*exp(-t*20))*exp(-t*4)'"
               ":s=48000:d=1.3", '-af', 'lowpass=f=900'],
    'chime': ['-f', 'lavfi', '-i',
              "aevalsrc=exprs='(0.35*sin(2*PI*880*t)+0.25*sin(2*PI*1318.5*t)+0.15*sin(2*PI*1760*t))*exp(-t*3)'"
              ":s=48000:d=1.6"],
}


def sfx_file(name, folder):
    dest = Path(folder) / f'{name}_v1.wav'
    if not dest.is_file():
        _ffmpeg(SFX[name] + ['-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le'], dest, timeout=60)
    return dest


def _library(folder):
    d = Path(folder) if folder else None
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob('*') if p.suffix.lower() in AUDIO_EXT and p.is_file())


def library_track(folder, mood, seed, seconds, dest_dir):
    """A track from the user's folder (mood word in the file name preferred), looped / cut to `seconds`."""
    files = _library(folder)
    if not files:
        return None
    hits = [p for p in files if mood in p.stem.lower()] or files
    src = random.Random(seed).choice(hits)
    sig = _h(str(src), src.stat().st_size, int(src.stat().st_mtime), round(seconds, 1))
    dest = Path(dest_dir) / f'music_lib_{sig}.wav'
    if not dest.is_file():
        d = max(1.0, seconds)
        _ffmpeg(['-stream_loop', '-1', '-i', str(src), '-t', f'{d:.2f}', '-af',
                 f'afade=t=in:d=1,afade=t=out:st={max(0.0, d - 2.5):.2f}:d=2.5', '-ac', '2', '-ar', '48000',
                 '-c:a', 'pcm_s16le'], dest)
    return dest, src.name


def library_sfx(folder, note):
    note = (note or '').lower()
    if not note:
        return None
    for p in _library(folder):
        words = [w for w in re.split(r'[\W_]+', p.stem.lower()) if len(w) > 2]
        if any(w in note for w in words):
            return p
    return None


# ------------------------------------------------------------------ assembly (Timeline IR)
def assemble(doc, sid, progress=None):
    """Script + storyboard + 3C media -> render-ready Timeline IR with canvas, camera moves, transitions, music,
    sfx and captions. -> (timeline, info)."""
    from . import genmedia, scripts
    from .timeline import model as m
    from .timeline.validate import ensure_valid
    st = settings_of(doc)
    tl = scripts.to_timeline(doc, sid)
    W, H = canvas_size(st)
    tl.canvas = m.Canvas(width=W, height=H, fps=float(st['fps']), background='#000000')
    scale = min(W, H) / 1080
    for s in tl.styles:
        s.size = max(28, int(round(s.size * scale * (0.8 if st['aspect'] == '16:9' else 1.0))))
    info = {'canvas': f'{W}x{H}', 'transitions': 0, 'sfx': [], 'music': None}
    fit_text(tl, W)
    root = genmedia.root(sid)
    audio_dir = root / 'audio'
    # missing media -> clear error (nothing is faked)
    voice_off = not st['voice'] or genmedia.config().get('tts_provider') == 'off'
    if voice_off:
        tl.tracks = [t for t in tl.tracks if not (t.type == 'audio' and any(
            i.type == 'audio' and i.role == 'voice' for i in t.items))]
    used = {i.asset for t in tl.tracks for i in t.items if getattr(i, 'asset', None)}
    missing = [a.uri for a in tl.assets if a.id in used and a.uri.startswith('gen:')]
    if missing:
        raise StageError(f'не сгенерировано {len(missing)} файл(ов): {", ".join(missing[:4])} — запустите этапы '
                         f'«Изображения» / «Озвучка» или замените элементы вручную')
    if not st['captions']:
        tl.tracks = [t for t in tl.tracks if t.type != 'caption']
    vtrack = next(t for t in tl.tracks if t.type == 'video')
    imgs = sorted((i for i in vtrack.items if i.type == 'image'), key=lambda i: i.start)
    # camera: storyboard moves, gentle Ken Burns on static frames, none when off
    amp = {'smooth': 1.0, 'dynamic': 1.8, 'off': 0.0}[st['motion']]
    for k, it in enumerate(imgs):
        if amp == 0:
            it.animations = []
            continue
        if not it.animations:
            z0, z1 = (1.0, 1.0 + 0.06 * amp) if k % 2 == 0 else (1.0 + 0.06 * amp, 1.0)
            it.animations = [m.Animation(property='zoom', keyframes=[m.Keyframe(t=0.0, value=round(z0, 3)), m.Keyframe(
                t=round(it.duration, 3), value=round(z1, 3), easing='ease_in_out')])]
        elif amp != 1.0:
            for a in it.animations:
                for kf in a.keyframes:
                    kf.value = round(1 + (kf.value - 1) * amp, 3) if a.property == 'zoom' else round(kf.value * amp, 3)
    # transitions: cross-fades (dynamic: cuts inside a scene, short fades between scenes)
    trans = []
    if st['transition'] != 'cut':
        for a, b in zip(imgs, imgs[1:]):
            cross = a.scene != b.scene
            d = {'smooth': 0.6 if cross else 0.35, 'dynamic': 0.25 if cross else 0.0}[st['transition']]
            d = min(d, a.duration * 0.4, b.duration * 0.4)
            if d < 0.1:
                continue
            a.duration = round(a.duration + d, 3)
            for an in a.animations:                     # keep the move's speed: its end lands on the new end
                if len(an.keyframes) > 1:
                    an.keyframes[-1].t = round(a.duration, 3)
            trans.append(m.TransitionItem(id=stable_id('it', 'tr', a.id, b.id), start=b.start, duration=round(d, 3),
                                          kind='fade', from_item=a.id, to_item=b.id))
        last = imgs[-1] if imgs else None
        if last is not None and last.start + last.duration > tl.duration:
            last.duration = round(tl.duration - last.start, 3)
        trans = [t for t in trans if t.start + t.duration <= tl.duration + 1e-6]
        vtrack.items += trans
    info['transitions'] = len(trans)
    # sound effects
    sfx_items, sfx_assets = [], {}

    def add_sfx(name, at, vol, path=None):
        if at < 0 or at >= tl.duration - 0.1:
            return
        path = path or sfx_file(name, Path(dj.MEDIA_ROOT) / 'generated' / 'sfx')
        aid = stable_id('as', 'sfx', str(path))
        if aid not in sfx_assets:
            from .timeline.media import probe
            dur = probe(path).get('duration') or 1.0
            sfx_assets[aid] = m.Asset(id=aid, type='audio', uri=f'media:{genmedia.rel(path)}', duration=dur,
                                      has_audio=True)
        dur = min(sfx_assets[aid].duration, tl.duration - at)
        sfx_items.append(m.AudioItem(id=stable_id('it', 'sfx', name, round(at, 2)), start=round(at, 3),
                                     duration=round(dur, 3), asset=aid, role='sfx', volume=vol))
        info['sfx'].append(name)
    if st['sfx']:
        for t in trans:
            if by_scene_change(t, vtrack):
                add_sfx('whoosh', t.start - 0.25, 0.45)
        for s in doc['scenes']:
            lib = library_sfx(st['sfx_dir'], s.get('sfx'))
            if lib is not None:
                add_sfx('lib', s['start'] + 0.05, 0.7, path=_lib_wav(lib, audio_dir))
        climax = next((s for s in doc['scenes'] if s.get('part') == 'climax'), None)
        if climax:
            add_sfx('impact', climax['start'], 0.6)
        for t in tl.tracks:
            if t.type == 'overlay':
                for it in t.items:
                    if it.type == 'text':
                        add_sfx('pop', it.start + 0.05, 0.5)
        end = next((s for s in reversed(doc['scenes']) if s.get('part') == 'ending'), None)
        if end and len(doc['scenes']) > 1:
            add_sfx('chime', end['start'] + 0.1, 0.35)
    # music
    music_items = []
    if st['music'] != 'off':
        mood = music_mood(doc)
        found = None
        if st['music'] in ('auto', 'library'):
            found = library_track(st['music_dir'], mood, sid or 0, tl.duration, audio_dir)
            if found is None and st['music'] == 'library':
                raise StageError(f'в папке музыки «{st["music_dir"] or "не задана"}» нет аудиофайлов '
                                 f'({", ".join(AUDIO_EXT)})')
        if found:
            path, name = found
            info['music'] = {'source': 'library', 'file': name, 'mood': mood}
        else:
            path = audio_dir / f'music_{mood}_{_h(mood, round(tl.duration, 1))}.wav'
            if not path.is_file():
                if progress:
                    progress(0.5, 'Синтез фоновой музыки')
                synth_music(mood, tl.duration, path)
            info['music'] = {'source': 'synth', 'mood': mood}
        aid = stable_id('as', 'music', str(path))
        sfx_assets[aid] = m.Asset(id=aid, type='audio', uri=f'media:{genmedia.rel(path)}', duration=tl.duration,
                                  has_audio=True)
        music_items.append(m.AudioItem(id=stable_id('it', 'music', sid or 0), start=0.0, duration=tl.duration,
                                       asset=aid, role='music', volume=st['music_volume'], duck=not voice_off,
                                       fade_in=0.8, fade_out=1.5))
    tl.assets += list(sfx_assets.values())
    if sfx_items:
        tl.tracks.append(m.Track(id=stable_id('tr', tl.id, 'sfx'), type='audio', name='Звуковые эффекты',
                                 items=sfx_items))
    if music_items:
        tl.tracks.append(m.Track(id=stable_id('tr', tl.id, 'music'), type='audio', name='Музыка', items=music_items))
    tl.meta = dict(tl.meta, production=st)
    return ensure_valid(tl), info


def wrap(text, cpl):
    lines, cur = [], ''
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > cpl:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    return lines + ([cur] if cur else [])


def fit_text(tl, W):
    """On-screen text: wrapped to short lines and sized so the widest line fits the frame (any format)."""
    styles = {s.id: s for s in tl.styles}
    for t in tl.tracks:
        for it in t.items:
            if it.type != 'text' or it.style not in styles:
                continue
            st = styles[it.style]
            lines = wrap(it.text.replace('\n', ' '), 16 if W < tl.canvas.height else 26)[:3]
            it.text = '\n'.join(lines)
            widest = max(len(x) for x in lines) or 1
            st.size = max(24, min(st.size, int(0.88 * W / (0.68 * widest))))


def by_scene_change(tr, vtrack):
    by = {i.id: i for i in vtrack.items}
    a, b = by.get(tr.from_item), by.get(tr.to_item)
    return bool(a and b and a.scene != b.scene)


def _lib_wav(src, folder):
    dest = Path(folder) / f'sfx_{_h(str(src), src.stat().st_size)}.wav'
    if not dest.is_file():
        _ffmpeg(['-i', str(src), '-t', '4', '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le'], dest, timeout=60)
    return dest


# ------------------------------------------------------------------ title / description / hashtags
SYSTEM_META = ('You write YouTube Shorts / TikTok metadata. Answer JSON only: {"title": "...", "description": "...", '
               '"hashtags": ["#tag", ...]}. Title <= 90 characters, catchy, no clickbait lies. Description 2-3 '
               'sentences. 5-8 hashtags. Write in the language of the script.')
TAGS = {'cartoon': ['мультфильм', 'анимация'], 'education': ['обучение', 'факты'], 'story': ['история'],
        'faceless': ['shorts', 'факты']}
TAGS_EN = {'cartoon': ['cartoon', 'animation'], 'education': ['learn', 'facts'], 'story': ['story'],
           'faceless': ['shorts', 'facts']}
STOP = set('это как что почему зачем который которые когда если или для про под над без the and for with that this '
           'from what why how your you are was were have has into about'.split())


def local_meta(doc):
    en = doc.get('language') == 'en'
    title = (doc.get('title') or '').strip()[:90] or 'Video'
    lines = [l['text'] for s in doc.get('scenes', [])[:2] for l in s['lines']]
    desc = ' '.join(x for x in [doc.get('logline', '')[:200], ' '.join(lines)[:260]] if x).strip()
    words = re.findall(r'[\wА-Яа-яЁё]{5,}', f'{doc.get("title", "")} {doc.get("logline", "")}'.lower())
    seen, kws = set(), []
    for w in words:
        if w not in STOP and w not in seen:
            seen.add(w)
            kws.append(w)
    base = (TAGS_EN if en else TAGS).get(doc.get('format'), [])
    tags = ['#' + re.sub(r'\W', '', t) for t in base + kws[:4] + ['shorts']]
    tags = list(dict.fromkeys(t for t in tags if len(t) > 2))[:8]
    return {'title': title, 'description': desc[:500], 'hashtags': tags, 'provider': 'local'}


def gen_meta(doc, cancel=None):
    from . import scripts
    user = json.dumps({'title': doc.get('title'), 'logline': doc.get('logline'), 'format': doc.get('format'),
                       'language': doc.get('language'),
                       'lines': [l['text'] for s in doc.get('scenes', []) for l in s['lines']][:12]},
                      ensure_ascii=False)
    raw, how = scripts._ask(SYSTEM_META, user, cancel)
    base = local_meta(doc)
    if isinstance(raw, dict) and raw.get('title'):
        tags = raw.get('hashtags') or []
        if isinstance(tags, str):
            tags = tags.split()
        tags = ['#' + re.sub(r'[^\w]', '', str(t).lstrip('#')) for t in tags]
        return {'title': str(raw['title']).strip()[:100], 'description': str(raw.get('description') or
                                                                             base['description']).strip()[:1500],
                'hashtags': [t for t in dict.fromkeys(tags) if len(t) > 1][:10] or base['hashtags'],
                'provider': how if isinstance(how, str) else 'llm'}
    return dict(base, note=str(how)[:120])


# ------------------------------------------------------------------ the pipeline
def output_path(sid, doc):
    from . import genmedia
    st = settings_of(doc)
    return genmedia.root(sid) / 'export' / f'video_{st["aspect"].replace(":", "x")}_{st["resolution"]}p.mp4'


def run(sc, rep, only=None, force=(), until=None, saver=None):
    """Run the pipeline for Script `sc` (model instance). only: stage keys to run (manual mode), until: stop after
    this stage, force: stages to redo although done. Saves the document after every step."""
    from .models import Script
    from .procutil import JobCancelled
    doc = sc.data or {}
    keys = [k for k in STAGE_KEYS if (only is None or k in only)]
    if until:
        keys = keys[:keys.index(until) + 1] if until in keys else keys
    done_w = [0.0]
    total_w = sum(WEIGHTS[k] for k in keys) or 1
    cur = {'k': None}

    def save(**fields):
        Script.objects.filter(pk=sc.pk).update(data=doc, **fields)

    def mark(k, status, message='', **info):
        p = prod(doc)
        p['stages'][k] = {'status': status, 'message': message[:500], 'at': int(time.time()), 'info': info}
        save()

    def prog(frac, msg=None):
        if rep is not None:
            rep.overall_fn = lambda: int(min(99, 100 * (done_w[0] + WEIGHTS[cur['k']] * frac) / total_w))
            rep.frac(frac, msg)
    for k in keys:
        states = stage_states(doc)
        if states[k]['status'] in ('done', 'skipped') and k not in force:
            done_w[0] += WEIGHTS[k]
            continue
        cur['k'] = k
        label = dict(STAGES)[k]
        if rep is not None:
            rep.overall_fn = lambda: int(min(99, 100 * done_w[0] / total_w))
            rep.stage(f'p_{k}', f'{label}...', frac=0.0)
        mark(k, 'running', f'{label}...')
        try:
            info = _stage(k, sc, doc, rep, prog, force)
        except JobCancelled:
            mark(k, 'pending', 'Отменено — продолжится с этого места')
            raise
        except Exception as exc:
            log.warning('production %s stage %s failed: %s', sc.pk, k, exc)
            mark(k, 'error', str(exc)[:500])
            save(status='error', error=f'{label}: {str(exc)[:900]}')
            raise
        mark(k, 'done', (info or {}).pop('message', '') if isinstance(info, dict) else '', **(info or {}))
        done_w[0] += WEIGHTS[k]
        if rep is not None:
            rep.checkpoint(f'p_{k}')
    save(status='ready', error='')
    return doc


def _stage(k, sc, doc, rep, prog, force):
    from . import genmedia, scripts, storyboard
    cancel = rep.cancelled if rep is not None else None
    if k == 'script':
        new = scripts.generate(scripts.settings_of(sc), cancel=cancel)
        keep = doc.get('production')
        doc.clear()
        doc.update(new)
        if keep:
            doc['production'] = keep
        doc['aspect'] = settings_of(doc)['aspect']
        from .models import Script
        Script.objects.filter(pk=sc.pk).update(title=sc.title or doc['title'][:250])
        return {'message': f'{len(doc["scenes"])} сцен, {doc.get("provider")}', 'provider': doc.get('provider')}
    if k == 'storyboard':
        aspect = doc['aspect'] = settings_of(doc)['aspect']
        stale = storyboard.mark_stale(doc) if doc.get('storyboard') else None
        new = storyboard.build(doc, cancel=cancel, only=set(stale) if stale and 'storyboard' not in force else None)
        if new['storyboard']['style'].get('aspect', '9:16') != aspect:          # format changed: re-frame prompts
            new = storyboard.edit_bibles(new, style={'camera_language': storyboard.ASPECT_TEXT[aspect] +
                                                     ', subject readable on a phone'})
            new['storyboard']['style']['aspect'] = aspect
            for ssb in new['storyboard']['scenes'].values():
                for f in ssb['frames']:
                    if not f.get('manual_prompt') and not f.get('locked'):
                        f['prompt'], f['negative'] = storyboard.compose(f, ssb, new['storyboard'])
        keep = doc.get('production')
        doc.clear()
        doc.update(new)
        if keep:
            doc['production'] = keep
        n = sum(len(s['frames']) for s in doc['storyboard']['scenes'].values())
        return {'message': f'{n} кадров', 'frames': n}
    if k == 'images':
        cfg = media_cfg(doc)
        r = genmedia.gen_images(doc, sc.pk, cfg, force='images' in force, cancel=cancel,
                                progress=lambda f, m: (_save(sc, doc), prog(f, f'Изображения: {m}')))
        if r['errors']:
            raise StageError(f'{len(r["errors"])} изображ. не удалось: ' + '; '.join(
                e['error'] for e in r['errors'][:2]))
        return {'message': f'новых {r["done"]}, из кэша {r["cached"]}' + (f'. {r["ip_note"]}' if r.get('ip_note')
                                                                        else ''), 'done': r['done'],
                'cached': r['cached'], 'ip_adapter': r['ip_adapter']}
    if k == 'voice':
        cfg = media_cfg(doc)
        r = genmedia.gen_voices(doc, sc.pk, cfg, force='voice' in force, cancel=cancel,
                                progress=lambda f, m: (_save(sc, doc), prog(f, f'Озвучка: {m}')))
        if r['errors']:
            raise StageError(f'{len(r["errors"])} реплик не озвучено: ' + '; '.join(
                e['error'] for e in r['errors'][:2]))
        return {'message': f'новых {r["done"]}, из кэша {r["cached"]}', 'done': r['done'], 'cached': r['cached']}
    if k == 'assemble':
        tl, info = assemble(doc, sc.pk, progress=prog)
        from .timeline import serialize
        path = genmedia.root(sc.pk) / 'timeline.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialize.to_dict(tl), ensure_ascii=False), encoding='utf-8')
        p = prod(doc)
        p['assembly_key'] = assembly_key(doc)
        p['timeline_path'] = genmedia.rel(path)
        mus = info['music']
        return dict(info, message=f'{info["canvas"]}, переходов {info["transitions"]}, эффектов {len(info["sfx"])}, '
                                  f'музыка: {"нет" if not mus else mus["source"] + " / " + mus["mood"]}',
                    sfx=sorted(set(info['sfx'])))
    if k == 'render':
        from .timeline import renderer, serialize
        p = prod(doc)
        tl = serialize.from_dict(json.loads((Path(dj.MEDIA_ROOT) / p['timeline_path']).read_text(encoding='utf-8')))
        dest = output_path(sc.pk, doc)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + '.part.mp4')
        r = renderer.render(tl, tmp, renderer.RenderSettings(quality=settings_of(doc)['quality']),
                            on_progress=lambda f: prog(f, f'Экспорт MP4: {int(f * 100)}%'), cancel=cancel)
        tmp.replace(dest)
        p['output'] = {'path': genmedia.rel(dest), 'assembly_key': p.get('assembly_key'), 'seconds': r['seconds'],
                       'encoder': r['encoder'], 'size': r['size'], 'rendered_scenes': r['rendered_scenes'],
                       'scenes': len(r['scenes']), 'at': int(time.time()), 'bytes': dest.stat().st_size}
        return {'message': f'{r["size"]}, {r["encoder"]}, перерендер сцен {r["rendered_scenes"]} из '
                           f'{len(r["scenes"])}, {r["seconds"]} с', 'rendered_scenes': r['rendered_scenes']}
    if k == 'meta':
        meta = gen_meta(doc, cancel)
        prod(doc)['meta'] = meta
        out = prod(doc).get('output', {}).get('path')
        if out:
            (Path(dj.MEDIA_ROOT) / out).with_suffix('.txt').write_text(
                f'{meta["title"]}\n\n{meta["description"]}\n\n{" ".join(meta["hashtags"])}\n', encoding='utf-8')
        return {'message': meta['title'][:80], 'provider': meta.get('provider')}
    raise KeyError(k)


def _save(sc, doc):
    from .models import Script
    Script.objects.filter(pk=sc.pk).update(data=doc)


def overview(doc, sc=None):
    """For the UI: settings, stage states, output, meta, media summary."""
    from . import genmedia
    p = (doc or {}).get('production') or {}
    out = p.get('output') or {}
    return {'settings': settings_of(doc or {}), 'stages': stage_states(doc or {}),
            'labels': dict(STAGES), 'order': STAGE_KEYS, 'output': out,
            'output_url': (dj.MEDIA_URL + out['path']) if out.get('path') and _exists(out['path']) else None,
            'meta': p.get('meta') or {}, 'media': genmedia.summary(doc) if (doc or {}).get('scenes') else {},
            'title': (doc or {}).get('title', ''), 'format': (doc or {}).get('format'),
            'scenes': len((doc or {}).get('scenes') or [])}
