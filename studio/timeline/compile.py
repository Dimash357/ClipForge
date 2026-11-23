"""Timeline IR -> ffmpeg filtergraphs (pure functions; renderer.py runs them).

Two kinds of ffmpeg commands are compiled:
* scene pass: the picture of one scene (its video + overlay items, layouts, camera moves, in-scene transitions)
  on the scene's background -> an intermediate clip that is cached by the scene fingerprint;
* final pass: scene clips placed on the timeline (cross-scene transitions as alpha fades), scene-less overlays,
  text + captions (one ASS file), every audio item mixed (loudness, volume, fades, ducking) -> the output file.
Everything is placed by timeline time, so picture, sound and captions share one clock.

Anything the renderer cannot draw yet is reported by check_supported() as an explicit error - never skipped.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .. import layouts
from . import model as m
from .serialize import TimelineError

TRANSITIONS = {'cut', 'fade', 'dissolve'}            # fade and dissolve are the same cross-fade here
ANIMS = {'video': {'zoom', 'pan_x', 'pan_y'}, 'image': {'zoom', 'pan_x', 'pan_y'},
         'text': {'x', 'y', 'scale', 'rotation', 'opacity'}, 'audio': {'volume'}, 'caption': set(), 'transition': set()}
TEXT_EASING = {'linear', 'ease_in', 'ease_out', 'hold'}
LAYOUT_PARAMS = {'bg', 'path', 'cam', 'fit', 'fill', 'hide_cam', 'cam_position', 'face_rel', 'zoom'}
SEG_TYPE = {'fit': 'fullframe', 'fullframe': 'fullframe', 'screen': 'screen', 'gameplay': 'gameplay', 'crop': 'center',
            'smart': 'smart', 'center': 'center', 'talking': 'talking', 'action': 'action', 'game_cam': 'game_cam',
            'reaction_cam': 'reaction_cam'}
DUCK = 'sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400'


class UnsupportedFeature(TimelineError):
    """The timeline is valid, but uses something this renderer cannot draw yet (see .issues)."""


def n(x):
    """Number for a filter argument: plain decimal, no exponent."""
    s = f'{float(x):.6f}'.rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def color(hex_rgb):
    return '0x' + hex_rgb.lstrip('#')[:6]


# ------------------------------------------------------------------ what this renderer supports
def check_supported(tl):
    """-> issues (same shape as validate()) for valid-but-unsupported features."""
    out = []

    def add(path, msg):
        out.append({'path': path, 'message': msg, 'level': 'error'})

    for si, st in enumerate(tl.styles):
        if any(c in st.font for c in ',\r\n{}'):
            add(f'styles[{si}].font', 'font name must not contain commas, braces or line breaks')
    scenes = {s.id for s in tl.scenes}
    for ti, t in enumerate(tl.tracks):
        for ii, it in enumerate(t.items):
            p = f'tracks[{ti}].items[{ii}]'
            allowed = ANIMS[it.type]
            props = {}
            for ai, a in enumerate(it.animations):
                if a.property not in allowed:
                    add(f'{p}.animations[{ai}].property', f'{a.property!r} animation on a {it.type} item is not '
                        f'supported yet (supported: {", ".join(sorted(allowed)) or "none"})')
                if a.property in props:
                    add(f'{p}.animations[{ai}]', f'second {a.property!r} animation on one item')
                props[a.property] = a
                if a.property == 'zoom' and any(k.value < 1.0 for k in a.keyframes):
                    add(f'{p}.animations[{ai}]', 'zoom below 1.0 (zooming out past the frame) is not supported')
                if it.type == 'text' and any(k.easing not in TEXT_EASING for k in a.keyframes):
                    add(f'{p}.animations[{ai}]', f'text animations support easing {sorted(TEXT_EASING)}')
            if it.type == 'text':
                xy = [props[k] for k in ('x', 'y') if k in props]
                if any(len(a.keyframes) > 2 for a in xy):
                    add(p, 'text x/y animation supports at most 2 keyframes (one straight move)')
                if any(k.easing not in ('linear', 'hold') for a in xy for k in a.keyframes[1:]):
                    add(p, 'text x/y animation supports linear easing only')
                if len(xy) == 2 and [k.t for k in xy[0].keyframes] != [k.t for k in xy[1].keyframes]:
                    add(p, 'text x and y animations must use the same keyframe times')
            if it.type == 'transition' and it.kind not in TRANSITIONS:
                add(f'{p}.kind', f'transition {it.kind!r} is not supported yet (supported: {sorted(TRANSITIONS)})')
            if it.type == 'video':
                if not 0.25 <= it.speed <= 4.0:
                    add(f'{p}.speed', 'speed must be within 0.25..4')
                elif it.speed != 1.0 and it.volume > 0 and not 0.5 <= it.speed <= 2.0:
                    add(f'{p}.speed', 'a video with its own sound supports speed 0.5..2 (or set volume 0)')
                if it.layout:
                    extra = set(it.layout.params) - LAYOUT_PARAMS
                    if extra:
                        add(f'{p}.layout.params', f'unknown layout parameter(s) {sorted(extra)}')
            if it.type in ('video', 'image') and t.type == 'video' and scenes and it.scene not in scenes:
                add(f'{p}.scene', 'items of a video track must belong to a scene when the timeline has scenes')
            if it.type == 'caption' and it.animations:
                add(f'{p}.animations', 'captions cannot be animated yet')
    return out


# ------------------------------------------------------------------ animation expressions
def _ease(name, p):
    return {'linear': p, 'ease_in': f'pow({p},2)', 'ease_out': f'(1-pow(1-{p},2))',
            'ease_in_out': f'if(lt({p},0.5),2*pow({p},2),1-2*pow(1-{p},2))', 'hold': '0'}[name]


def anim_expr(keyframes, var, default=0.0):
    """Piecewise keyframe curve as an ffmpeg expression of `var` (seconds from the item start).
    The easing of a keyframe shapes the segment that *arrives* at it; before the first / after the last keyframe
    the value holds."""
    if not keyframes:
        return n(default)
    k = keyframes
    expr = n(k[-1].value)
    for a, b in reversed(list(zip(k, k[1:]))):
        dt = b.t - a.t
        if dt <= 0:
            continue
        p = f'(({var})-{n(a.t)})/{n(dt)}'
        expr = f'if(lt({var},{n(b.t)}),{n(a.value)}+({n(b.value - a.value)})*{_ease(b.easing, p)},{expr})'
    return f'if(lt({var},{n(k[0].t)}),{n(k[0].value)},{expr})'


# ------------------------------------------------------------------ scene / layer planning
@dataclass
class Input:
    args: List[str]                 # input options + '-i' + path


@dataclass
class Graph:
    inputs: List[Input] = field(default_factory=list)
    parts: List[str] = field(default_factory=list)
    margins: list = field(default_factory=list)          # (t0, t1, caption margin px) on the timeline

    def add_input(self, *args):
        self.inputs.append(Input(list(args)))
        return len(self.inputs) - 1

    def cmd_inputs(self):
        return [a for i in self.inputs for a in i.args]

    def text(self):
        return ';'.join(self.parts)


@dataclass
class SceneWindow:
    id: Optional[str]               # None = implicit whole-timeline scene
    start: float
    duration: float                 # including the tail into the next scene (outgoing transition)
    own: float                      # the scene's own length


def scene_windows(tl):
    if not tl.scenes:
        return [SceneWindow(None, 0.0, tl.duration, tl.duration)]
    out = []
    for sc in sorted(tl.scenes, key=lambda s: s.start):
        end = sc.start + sc.duration
        tail = 0.0
        for t, it in tl.items(scene=sc.id):
            if t.type in ('video', 'overlay') and it.type in ('video', 'image'):
                tail = max(tail, it.start + it.duration - end)
        out.append(SceneWindow(sc.id, sc.start, round(min(tl.duration - sc.start, sc.duration + tail), 3), sc.duration))
    return out


def visual_tracks(tl):
    return [t for t in tl.tracks if t.type in ('video', 'overlay') and not t.hidden]


def transitions_by_target(tl):
    """to_item id -> (TransitionItem, from item, to item)."""
    out = {}
    for t in tl.tracks:
        byid = {i.id: i for i in t.items}
        for it in t.items:
            if it.type == 'transition' and it.kind != 'cut':
                out[it.to_item] = (it, byid.get(it.from_item), byid.get(it.to_item))
    return out


def _camera(it, W, H, fps, alpha):
    props = {a.property: a.keyframes for a in it.animations if a.property in ('zoom', 'pan_x', 'pan_y')}
    if not props:
        return ''
    T = f'on/{n(fps)}'
    z = anim_expr(props.get('zoom'), T, 1.0)
    px = anim_expr(props.get('pan_x'), T, 0.0)
    py = anim_expr(props.get('pan_y'), T, 0.0)
    fmt = 'yuva420p' if alpha else 'yuv420p'
    # work at 2x so slow moves do not step pixel by pixel
    return (f',format={fmt},scale={2 * W}:{2 * H}:flags=bicubic,'
            f"zoompan=z='{z}':x='max(0,min(iw-iw/zoom,(iw-iw/zoom)/2+({px})*iw))':"
            f"y='max(0,min(ih-ih/zoom,(ih-ih/zoom)/2+({py})*ih))':d=1:s={W}x{H}:fps={n(fps)}")


def _layout_ctx(lay, W, H, src_w, src_h, tol):
    p = lay.params if lay else {}
    return {'W': W, 'H': H, 'src_w': src_w, 'src_h': src_h, 'bg': p.get('bg', 'blur'),
            'zoom': float(p.get('zoom', 1.0)), 'cam': p.get('cam'), 'cam_position': p.get('cam_position', 'bottom'),
            'hide_cam': bool(p.get('hide_cam', True)), 'cam_fit': p.get('fit', 'balanced'),
            'cam_fill': p.get('fill', 'mirror'), 'path_tol': tol}


def _seg(lay):
    seg = {'type': SEG_TYPE[lay.type] if lay else 'center'}
    if lay and lay.params.get('path'):
        seg['path'] = [tuple(float(v) for v in pt) for pt in lay.params['path']]
    if lay and lay.params.get('face_rel') is not None:
        seg['face_rel'] = tuple(float(v) for v in lay.params['face_rel'])
    return seg


def add_layer(g, k, it, track, tl, media, offset, fade_in=None, tol=3.0):
    """Input k (already added) -> one canvas-sized layer placed at `offset` seconds of the output. -> label."""
    W, H, fps = tl.canvas.width, tl.canvas.height, tl.canvas.fps
    tag = f'L{len(g.parts)}'
    alpha = track.type == 'overlay'
    D = n(it.duration)
    if it.type == 'video':
        info = media[it.asset]
        speed = f'/{n(it.speed)}' if it.speed != 1.0 else ''
        g.parts.append(f'[{k}:v]setpts=(PTS-STARTPTS){speed},fps={n(fps)},trim=duration={D},'
                       f'setpts=PTS-STARTPTS[{tag}a]')
        chain, margin = layouts.seg_layout(_seg(it.layout), _layout_ctx(it.layout, W, H, info['width'], info['height'], tol),
                                           f'{tag}a', f'{tag}b')
        g.parts.append(chain)
        if track.type == 'video':
            g.margins.append((it.start, it.start + it.duration, margin))
        src = f'[{tag}b]setsar=1'
    else:                       # image
        fit = {'cover': f'scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}',
               'contain': f'scale={W}:{H}:force_original_aspect_ratio=decrease,'
                          f'pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={color(tl.canvas.background)}@{"0" if alpha else "1"}',
               'fill': f'scale={W}:{H}'}[it.fit]
        src = f'[{k}:v]format=rgba,{fit},setsar=1,fps={n(fps)},trim=duration={D},setpts=PTS-STARTPTS'
    src += _camera(it, W, H, fps, alpha)
    if fade_in:
        st, d = fade_in
        src += f',format=yuva420p,fade=t=in:st={n(st)}:d={n(d)}:alpha=1'
    elif alpha:
        src += ',format=yuva420p'
    g.parts.append(f'{src},setpts=PTS+{n(offset)}/TB[{tag}]')
    return tag


def item_input(it, path):
    """ffmpeg input options for a visual item (only the needed range is decoded)."""
    from .media import ffmpeg_input
    if it.type == 'image':
        return ['-loop', '1', '-framerate', '25', '-t', n(it.duration + 0.5), '-i', ffmpeg_input(path)]
    need = it.duration * it.speed
    return ['-ss', n(it.src_in), '-t', n(need + 0.5), '-i', ffmpeg_input(path)]


def scene_graph(tl, win, media, paths, tol=3.0):
    """Graph of one scene's picture, local time 0..win.duration. -> Graph (output label [vout])."""
    W, H, fps = tl.canvas.width, tl.canvas.height, tl.canvas.fps
    g = Graph()
    g.parts.append(f'color=c={color(tl.canvas.background)}:s={W}x{H}:r={n(fps)}:d={n(win.duration)},'
                   f'format=yuv420p[bg]')
    cur = 'bg'
    trans = transitions_by_target(tl)
    for track in visual_tracks(tl):
        for it in sorted((i for i in track.items if i.type in ('video', 'image')), key=lambda i: i.start):
            if win.id is not None and it.scene != win.id:
                continue
            if track.type == 'overlay' and win.id is not None and it.scene is None:
                continue
            fade = None
            tr = trans.get(it.id)
            if tr and tr[1] is not None and (win.id is None or tr[1].scene == it.scene):
                fade = (tr[0].start - it.start, tr[0].duration)          # in-scene cross-fade
            k = g.add_input(*item_input(it, paths[it.asset]))
            lab = add_layer(g, k, it, track, tl, media, it.start - win.start, fade, tol)
            g.parts.append(f'[{cur}][{lab}]overlay=eof_action=pass:format=auto[o{lab}]')
            cur = f'o{lab}'
    g.parts.append(f'[{cur}]format=yuv420p[vout]')
    return g


def scene_margins(tl, win, media):
    """Caption margins of the scene's video-track layouts (without building the whole graph)."""
    out = []
    W, H = tl.canvas.width, tl.canvas.height
    for track in visual_tracks(tl):
        if track.type != 'video':
            continue
        for it in track.items:
            if it.type == 'video' and (win.id is None or it.scene == win.id):
                info = media[it.asset]
                _, mv = layouts.seg_layout(_seg(it.layout), _layout_ctx(it.layout, W, H, info['width'], info['height'],
                                                                         3.0), 'a', 'b')
                out.append((it.start, it.start + it.duration, mv))
    return out


def final_video(tl, g, windows, clip_inputs, media, paths, ass_name=None):
    """Scene clips + scene-less overlays + text/captions -> [vout] in graph g."""
    W, H, fps = tl.canvas.width, tl.canvas.height, tl.canvas.fps
    g.parts.append(f'color=c={color(tl.canvas.background)}:s={W}x{H}:r={n(fps)}:d={n(tl.duration)},'
                   f'format=yuv420p[fbg]')
    cur = 'fbg'
    trans = transitions_by_target(tl)
    first_items = {}
    for track in visual_tracks(tl):
        for it in track.items:
            if it.type in ('video', 'image') and it.scene:
                first_items.setdefault(it.scene, []).append(it)
    for win in windows:
        k = clip_inputs[win.id]
        chain = f'[{k}:v]setpts=PTS-STARTPTS'
        for it in first_items.get(win.id, []):
            tr = trans.get(it.id)
            if tr and tr[1] is not None and tr[1].scene != it.scene:              # cross-scene cross-fade
                chain += f',format=yuva420p,fade=t=in:st={n(tr[0].start - win.start)}:d={n(tr[0].duration)}:alpha=1'
                break
        g.parts.append(f'{chain},setpts=PTS+{n(win.start)}/TB[sc{k}]')
        g.parts.append(f'[{cur}][sc{k}]overlay=eof_action=pass:format=auto[fo{k}]')
        cur = f'fo{k}'
    if tl.scenes:                                              # overlays that belong to no scene: on top of all
        for track in visual_tracks(tl):
            if track.type != 'overlay':
                continue
            for it in track.items:
                if it.type in ('video', 'image') and it.scene is None:
                    tr = trans.get(it.id)
                    fade = (tr[0].start - it.start, tr[0].duration) if tr and tr[1] is not None else None
                    kk = g.add_input(*item_input(it, paths[it.asset]))
                    lab = add_layer(g, kk, it, track, tl, media, it.start, fade)
                    g.parts.append(f'[{cur}][{lab}]overlay=eof_action=pass:format=auto[go{lab}]')
                    cur = f'go{lab}'
    tail = [f'ass={ass_name}:fontsdir=fonts'] if ass_name else []
    tail.append('format=yuv420p')
    g.parts.append(f'[{cur}]' + ','.join(tail) + '[vout]')


# ------------------------------------------------------------------ audio
@dataclass
class AudioSource:
    item_id: str
    path: object
    src_in: float
    start: float
    duration: float
    speed: float = 1.0
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    role: str = 'source'
    duck: bool = False
    volume_anim: list = None
    loudnorm: Optional[str] = None


def audio_sources(tl, paths, media):
    out = []
    for t in tl.tracks:
        if t.type == 'audio' and not t.muted:
            for it in t.items:
                vol = [a.keyframes for a in it.animations if a.property == 'volume']
                out.append(AudioSource(it.id, paths[it.asset], it.src_in, it.start, it.duration, 1.0, it.volume,
                                       it.fade_in, it.fade_out, it.role, it.duck, vol[0] if vol else None))
        elif t.type in ('video', 'overlay') and not t.muted and not t.hidden:
            for it in t.items:                         # a video's own sound (volume > 0)
                if it.type == 'video' and it.volume > 0 and media[it.asset]['has_audio']:
                    out.append(AudioSource(it.id, paths[it.asset], it.src_in, it.start, it.duration, it.speed,
                                           it.volume, role='source'))
    return out


def audio_graph(g, sources, total):
    """Every audio source on the timeline clock -> [aout] (48 kHz stereo, exactly `total` seconds)."""
    from .media import ffmpeg_input
    if not sources:
        g.parts.append(f'anullsrc=r=48000:cl=stereo,atrim=duration={n(total)}[aout]')
        return
    lead, bed, rest = [], [], []
    for i, s in enumerate(sources):
        k = g.add_input('-ss', n(s.src_in), '-t', n(s.duration * s.speed + 0.5), '-i', ffmpeg_input(s.path))
        ch = [f'[{k}:a]asetpts=PTS-STARTPTS']
        if s.speed != 1.0:
            ch.append(f'atempo={n(s.speed)}')
        ch += ['aresample=48000', 'aformat=sample_fmts=fltp:channel_layouts=stereo', f'atrim=duration={n(s.duration)}']
        if s.loudnorm:
            ch += [s.loudnorm, 'aresample=48000', 'aformat=sample_fmts=fltp:channel_layouts=stereo']
        if s.volume_anim:
            ch.append(f"volume='{anim_expr(s.volume_anim, 't', 1.0)}*{n(s.volume)}':eval=frame")
        elif s.volume != 1.0:
            ch.append(f'volume={n(s.volume)}')
        if s.fade_in > 0:
            ch.append(f'afade=t=in:st=0:d={n(s.fade_in)}')
        if s.fade_out > 0:
            ch.append(f'afade=t=out:st={n(max(0.0, s.duration - s.fade_out))}:d={n(s.fade_out)}')
        if s.start > 0:            # leading silence; then count time in samples so later trims keep it
            ch += [f'adelay={int(round(s.start * 1000))}:all=1', 'asetpts=N/SR/TB']
        g.parts.append(','.join(ch) + f'[a{i}]')
        (bed if s.duck else lead if s.role in ('voice', 'source') else rest).append(f'a{i}')

    def mix(labels, out):
        if len(labels) == 1:
            g.parts.append(f'[{labels[0]}]anull[{out}]')
        else:
            g.parts.append(''.join(f'[{x}]' for x in labels) +
                           f'amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0[{out}]')
        return out

    final = []
    if lead and bed:
        mix(lead, 'lead')
        g.parts.append('[lead]asplit=2[leadm][leadsc]')
        mix(bed, 'bed')
        g.parts.append(f'[bed][leadsc]{DUCK}[ducked]')
        final += ['leadm', 'ducked']
    else:
        final += lead + bed
    final += rest
    mix(final, 'amixed')
    g.parts.append(f'[amixed]apad=whole_dur={n(total)},atrim=duration={n(total)},asetpts=N/SR/TB[aout]')
