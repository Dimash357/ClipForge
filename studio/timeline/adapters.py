"""Bridges between the existing Stream Clipper and Timeline IR (gradual integration).

clip_timeline() describes a clip exactly as the current renderer (editor.render + layouts.build_graph) produced
it: the scene plan, layouts with their parameters, the transition, the source audio with its loudness target and
the captions. The renderer itself is untouched; the timeline is stored next to the render as a snapshot that the
coming IR renderer, Viral Director and per-scene regeneration build on.
"""
from .. import captions as cap
from .. import exports
from .. import textutil as tx
from . import model as m
from .ids import stable_id
from .serialize import to_dict
from .validate import ensure_valid

TRANSITION_SECONDS = 0.25          # layouts.build_graph(..., tdur=0.25)
LAYOUTS = set(m.LayoutType.__args__)


def _round_path(path):
    return [[round(float(t), 3), round(float(x), 1), round(float(y), 1)] for t, x, y in (path or [])]


def _cam_params(cam):
    if not cam:
        return None
    keep = {k: cam[k] for k in ('x', 'y', 'w', 'h', 'corner', 'confidence', 'source', 'face_h', 'eye_rel') if k in cam}
    for k in ('x', 'y', 'w', 'h'):
        keep[k] = round(float(keep[k]), 4)
    if cam.get('face_rel') is not None:
        keep['face_rel'] = [round(float(v), 3) for v in cam['face_rel']]
    return keep


def _layout(seg, plan, opts):
    typ = seg.get('type') if seg.get('type') in LAYOUTS else 'fit'
    params = {'bg': opts.get('bg_fill', 'blur')}
    if typ in ('crop', 'smart', 'center', 'talking', 'action'):
        if seg.get('path'):
            params['path'] = _round_path(seg['path'])
    if typ in ('game_cam', 'reaction_cam'):
        params['cam'] = _cam_params(plan.get('cam'))
        params['fit'] = opts.get('cam_fit', 'balanced')
        params['fill'] = opts.get('cam_fill', 'mirror')
        params['hide_cam'] = bool(opts.get('hide_cam', True))
        if typ == 'game_cam':
            params['cam_position'] = opts.get('cam_position', 'bottom')
        if seg.get('face_rel') is not None:
            params['face_rel'] = [round(float(v), 3) for v in seg['face_rel']]
    if typ == 'gameplay':
        params['zoom'] = float(opts.get('gameplay_zoom', 1.0))
    return m.Layout(type=typ, params=params)


def _style(opts):
    name = opts.get('subtitle_style', 'stream')
    font_choice = opts.get('caption_font', 'style')
    if name == 'classic':
        st = {'font': 'Arial', 'size': 64, 'bold': 1, 'upper': False, 'primary': '#FFFFFF', 'active': '#FFFF00',
              'outline_c': '#101010', 'outline': 5, 'max_words': 5}
    else:
        st = dict(cap.STYLES.get(name) or cap.STYLES[cap.DEFAULT_STYLE])
    font, bold = st['font'], bool(st['bold'])
    if font_choice in cap.FONTS:
        font, b = cap.FONTS[font_choice]
        bold = bool(b)
    words = int(opts.get('caption_words') or 0) or int(st['max_words'])
    return m.Style(id=stable_id('st', name, font, words), font=font, size=int(st['size']), bold=bold,
                   uppercase=bool(st['upper']), color=st['primary'], highlight_color=st['active'],
                   outline_color=st['outline_c'], outline=float(st['outline']), max_words=max(1, min(12, words)),
                   ext={'clipforge.style': {'name': name, 'censor': bool(opts.get('caption_censor')),
                                            'font_choice': font_choice, 'animate': bool(opts.get('caption_animate', True)),
                                            'highlight': opts.get('caption_highlight')}})


def _words(transcript, start, end, censor):
    words = [w for w in tx.flat_words([s for s in transcript if s['end'] > start and s['start'] < end])
             if w['end'] > start and w['start'] < end and w['text'].strip()]
    keys = cap._key_words(words) if words else set()
    out = []
    for w in words:
        t0, t1 = max(0.0, w['start'] - start), min(end - start, w['end'] - start)
        if t1 <= t0:
            continue
        text = cap.censor_word(w['text'].strip()) if censor else w['text'].strip()
        out.append(m.Word(t0=round(t0, 3), t1=round(t1, 3), text=text, emphasis=cap._norm(w['text']) in keys))
    out.sort(key=lambda w: w.t0)
    return out


def clip_timeline(*, project_id, clip_id, source_name, fingerprint, clip_start, clip_end, plan, opts, vinfo,
                  render_info, transcript, title=''):
    """Timeline IR of a clip rendered by the current Stream Clipper renderer (validated)."""
    L = round(float(clip_end) - float(clip_start), 3)
    W, H = exports.dims(opts)
    fps = render_info.get('fps')
    fps = float(fps) if isinstance(fps, (int, float)) else round(min(120.0, max(1.0, float(vinfo.get('fps') or 30))), 3)
    tl_id = stable_id('tl', 'clip', project_id, clip_id)
    src = m.Asset(id=stable_id('as', 'source', fingerprint or source_name), type='video', uri=f'media:{source_name}',
                  duration=round(float(vinfo['duration']), 3) if vinfo.get('duration') else None,
                  width=int(vinfo['width']), height=int(vinfo['height']), has_audio=bool(vinfo.get('has_audio', True)),
                  checksum=fingerprint or '')
    segs = list(plan.get('segments') or [{'start': 0.0, 'end': L, 'type': 'center'}])
    fade = render_info.get('transition') == 'fade' and len(segs) > 1
    scenes, video_items, transitions = [], [], []
    for i, sg in enumerate(segs):
        s0, s1 = round(float(sg['start']), 3), round(min(L, float(sg['end'])), 3)
        if s1 <= s0:
            continue
        sid = stable_id('sc', 'clip', clip_id, i)
        scenes.append(m.Scene(id=sid, start=s0, duration=round(s1 - s0, 3), name=str(sg.get('type', '')),
                              status='generated', generation=m.Generation(
                                  provider='stream_clipper', params={k: sg.get(k) for k in
                                                                     ('detected', 'confidence', 'fallback', 'reason')
                                                                     if sg.get(k) is not None})))
        extra = TRANSITION_SECONDS if fade and i < len(segs) - 1 else 0.0
        video_items.append(m.VideoItem(id=stable_id('it', 'clip', clip_id, 'video', i), start=s0,
                                       duration=round(s1 - s0 + extra, 3), scene=sid, asset=src.id,
                                       src_in=round(float(clip_start) + s0, 3), volume=0.0,   # sound: audio track
                                       layout=_layout(sg, plan, opts)))
    if fade:
        for a, b in zip(video_items, video_items[1:]):
            transitions.append(m.TransitionItem(id=stable_id('it', 'clip', clip_id, 'fade', a.id), start=b.start,
                                                duration=TRANSITION_SECONDS, kind='fade', from_item=a.id,
                                                to_item=b.id))
    tracks = [m.Track(id=stable_id('tr', tl_id, 'video'), type='video', name='Видео',
                      items=video_items + transitions)]
    if vinfo.get('has_audio', True):
        loud = render_info.get('loudness') or {}
        tracks.append(m.Track(id=stable_id('tr', tl_id, 'audio'), type='audio', name='Звук', items=[
            m.AudioItem(id=stable_id('it', 'clip', clip_id, 'audio'), start=0.0, duration=L, asset=src.id,
                        role='source', src_in=round(float(clip_start), 3),
                        ext={'clipforge.loudness': loud} if loud else {})]))
    styles = []
    if opts.get('captions', True):
        style = _style(opts)
        styles.append(style)
        words = _words(transcript or [], float(clip_start), float(clip_end), bool(opts.get('caption_censor')))
        if words:
            tracks.append(m.Track(id=stable_id('tr', tl_id, 'captions'), type='caption', name='Субтитры', items=[
                m.CaptionItem(id=stable_id('it', 'clip', clip_id, 'captions'), start=0.0, duration=L, words=words,
                              style=style.id)]))
    tl = m.Timeline(
        id=tl_id, kind='clip', duration=L,
        canvas=m.Canvas(width=W, height=H, fps=fps),
        assets=[src], scenes=scenes, tracks=tracks, styles=styles,
        meta={'title': title, 'source': 'stream_clipper', 'project_id': project_id, 'clip_id': clip_id,
              'clip_range': [round(float(clip_start), 3), round(float(clip_end), 3)],
              'mode': render_info.get('mode'), 'auto': render_info.get('auto'),
              'notes': list(render_info.get('notes') or [])},
        ext={'clipforge.render': {k: render_info.get(k) for k in ('encoder', 'size', 'graph_chars', 'transition')
                                  if render_info.get(k) is not None}})
    return ensure_valid(tl)


def clip_timeline_dict(**kw):
    return to_dict(clip_timeline(**kw))
