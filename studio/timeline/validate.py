"""Semantic rules of Timeline IR. validate() collects every problem with its path; ensure_valid() raises.

Structure (types, enums, unknown keys) is checked by serialize.from_dict; this module checks what a structure
cannot: ids, references, time ranges, overlaps, keyframes, asset bounds.
"""
import re

from . import ids
from . import model as m
from .serialize import TimelineError

EPS = 0.002          # seconds of tolerance for float rounding
HEX_COLOR = re.compile(r'^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$')
# which asset types an item may point at
ASSET_FOR = {'video': {'video'}, 'image': {'image'}, 'audio': {'audio', 'video'}}


class _Issues(list):
    def add(self, path, message, level='error'):
        self.append({'path': path, 'message': message, 'level': level})


def validate(tl):
    """-> list of {'path', 'message', 'level'} ('error' | 'warning'); empty = valid."""
    out = _Issues()
    if tl.schema != m.SCHEMA_ID:
        out.add('schema', f'expected {m.SCHEMA_ID!r}')
    if tl.version != m.VERSION:
        out.add('version', f'expected {m.VERSION} (migrate first)')
    if not (tl.duration > 0):
        out.add('duration', 'must be > 0')
    _canvas(tl.canvas, out)
    seen = {}

    def claim(eid, path):
        if not ids.is_valid(eid):
            out.add(path, f'bad id {eid!r}: expected <prefix>_<6-40 chars of 0-9a-z>')
        elif eid in seen:
            out.add(path, f'duplicate id {eid!r} (also at {seen[eid]})')
        else:
            seen[eid] = path

    claim(tl.id, 'id')
    assets = {}
    for i, a in enumerate(tl.assets):
        p = f'assets[{i}]'
        claim(a.id, f'{p}.id')
        assets[a.id] = a
        if not a.uri:
            out.add(f'{p}.uri', 'must not be empty')
        for k in ('duration', 'width', 'height'):
            v = getattr(a, k)
            if v is not None and v <= 0:
                out.add(f'{p}.{k}', 'must be > 0')
    styles = {}
    for i, s in enumerate(tl.styles):
        p = f'styles[{i}]'
        claim(s.id, f'{p}.id')
        styles[s.id] = s
        for k in ('color', 'highlight_color', 'outline_color'):
            if not HEX_COLOR.match(getattr(s, k)):
                out.add(f'{p}.{k}', 'expected #RRGGBB or #RRGGBBAA')
        if s.size <= 0:
            out.add(f'{p}.size', 'must be > 0')
        if not 1 <= s.max_words <= 12:
            out.add(f'{p}.max_words', 'must be 1..12')
    chars = {}
    for i, c in enumerate(tl.characters):
        p = f'characters[{i}]'
        claim(c.id, f'{p}.id')
        chars[c.id] = c
        for j, r in enumerate(c.refs):
            if r not in assets:
                out.add(f'{p}.refs[{j}]', f'unknown asset {r!r}')
            elif assets[r].type != 'image':
                out.add(f'{p}.refs[{j}]', 'a reference must be an image asset')
    scenes = {}
    for i, s in enumerate(tl.scenes):
        p = f'scenes[{i}]'
        claim(s.id, f'{p}.id')
        scenes[s.id] = s
        _range(s.start, s.duration, tl.duration, p, out)
        for j, c in enumerate(s.characters):
            if c not in chars:
                out.add(f'{p}.characters[{j}]', f'unknown character {c!r}')
    ordered = sorted(tl.scenes, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start < a.start + a.duration - EPS:
            out.add('scenes', f'scenes {a.id} and {b.id} overlap')

    for ti, t in enumerate(tl.tracks):
        tp = f'tracks[{ti}]'
        claim(t.id, f'{tp}.id')
        by_id = {}
        # an item that a transition starts from plays on for the transition's length (overlap with the next one)
        tail = {it.from_item: it.duration for it in t.items if it.type == 'transition'}
        for ii, it in enumerate(t.items):
            p = f'{tp}.items[{ii}]'
            claim(it.id, f'{p}.id')
            by_id[it.id] = it
            if it.type not in m.TRACK_ITEMS[t.type]:
                out.add(f'{p}.type', f'a {t.type!r} track cannot hold {it.type!r} items')
            _range(it.start, it.duration, tl.duration, p, out)
            if it.scene is not None:
                sc = scenes.get(it.scene)
                if sc is None:
                    out.add(f'{p}.scene', f'unknown scene {it.scene!r}')
                elif it.start < sc.start - EPS or \
                        it.start + it.duration > sc.start + sc.duration + tail.get(it.id, 0.0) + EPS:
                    out.add(f'{p}.scene', f'item lies outside its scene {sc.id}')
            _animations(it, p, out)
            _item(it, p, assets, styles, out)
        _transitions_and_overlaps(t, by_id, tp, out)
    return list(out)


def ensure_valid(tl):
    issues = [i for i in validate(tl) if i['level'] == 'error']
    if issues:
        head = '; '.join(f'{i["path"]}: {i["message"]}' for i in issues[:5])
        raise TimelineError(f'invalid timeline ({len(issues)} problem(s)): {head}', issues)
    return tl


# ------------------------------------------------------------------ helpers
def _canvas(c, out):
    if c.width <= 0 or c.height <= 0 or c.width % 2 or c.height % 2:
        out.add('canvas', 'width and height must be positive even numbers (yuv420)')
    if not 1 <= c.fps <= 120:
        out.add('canvas.fps', 'must be 1..120')
    if not HEX_COLOR.match(c.background):
        out.add('canvas.background', 'expected #RRGGBB')


def _range(start, duration, total, p, out):
    if start < 0:
        out.add(f'{p}.start', 'must be >= 0')
    if not duration > 0:
        out.add(f'{p}.duration', 'must be > 0')
    elif start + duration > total + EPS:
        out.add(f'{p}.duration', f'ends at {start + duration:.3f}s, after the timeline ({total:.3f}s)')


def _animations(it, p, out):
    for ai, an in enumerate(it.animations):
        ap = f'{p}.animations[{ai}]'
        if not an.keyframes:
            out.add(ap, 'needs at least one keyframe')
        prev = None
        for ki, k in enumerate(an.keyframes):
            if k.t < -EPS or k.t > it.duration + EPS:
                out.add(f'{ap}.keyframes[{ki}].t', f'must be within the item (0..{it.duration:.3f}s)')
            if prev is not None and k.t < prev - EPS:
                out.add(f'{ap}.keyframes[{ki}].t', 'keyframes must be in time order')
            prev = k.t
            if an.property == 'opacity' and not 0 <= k.value <= 1:
                out.add(f'{ap}.keyframes[{ki}].value', 'opacity must be 0..1')
            if an.property in ('scale', 'zoom') and k.value <= 0:
                out.add(f'{ap}.keyframes[{ki}].value', f'{an.property} must be > 0')


def _item(it, p, assets, styles, out):
    if hasattr(it, 'asset'):
        a = assets.get(it.asset)
        if a is None:
            out.add(f'{p}.asset', f'unknown asset {it.asset!r}')
        else:
            if a.type not in ASSET_FOR[it.type]:
                out.add(f'{p}.asset', f'a {it.type} item cannot use a {a.type} asset')
            src_in = getattr(it, 'src_in', 0.0)
            speed = getattr(it, 'speed', 1.0)
            if src_in < 0:
                out.add(f'{p}.src_in', 'must be >= 0')
            if a.duration is not None and it.type in ('video', 'audio') and \
                    src_in + it.duration * speed > a.duration + EPS:
                out.add(f'{p}.duration', f'reads past the end of asset {a.id} ({a.duration:.3f}s)')
            if it.type == 'audio' and a.type == 'video' and a.has_audio is False:
                out.add(f'{p}.asset', f'asset {a.id} has no audio track')
    if getattr(it, 'style', None) is not None and it.style not in styles:
        out.add(f'{p}.style', f'unknown style {it.style!r}')
    if it.type == 'video':
        if it.speed <= 0:
            out.add(f'{p}.speed', 'must be > 0')
        if not 0 <= it.volume <= 4:
            out.add(f'{p}.volume', 'must be 0..4')
        if it.layout is not None:
            _layout(it.layout, f'{p}.layout', out)
    if it.type == 'audio':
        if not 0 <= it.volume <= 4:
            out.add(f'{p}.volume', 'must be 0..4')
        if it.fade_in < 0 or it.fade_out < 0 or it.fade_in + it.fade_out > it.duration + EPS:
            out.add(f'{p}.fade_in', 'fades must be >= 0 and fit inside the item')
    if it.type == 'text':
        if not it.text.strip():
            out.add(f'{p}.text', 'must not be empty')
        if not (0 <= it.x <= 1 and 0 <= it.y <= 1):
            out.add(f'{p}.x', 'x and y are normalised to the canvas (0..1)')
    if it.type == 'caption':
        end = it.start + it.duration
        prev = None
        for wi, w in enumerate(it.words):
            wp = f'{p}.words[{wi}]'
            if not w.text.strip():
                out.add(f'{wp}.text', 'must not be empty')
            if w.t1 <= w.t0:
                out.add(wp, 't1 must be after t0')
            if w.t0 < it.start - EPS or w.t1 > end + EPS:
                out.add(wp, 'word lies outside its caption item (words use timeline time)')
            if prev is not None and w.t0 < prev - EPS:
                out.add(wp, 'words must be in time order')
            prev = w.t0


def _layout(lay, p, out):
    path = lay.params.get('path')
    if path is not None:
        if not isinstance(path, list) or not all(isinstance(pt, (list, tuple)) and len(pt) == 3 and
                                                 all(isinstance(v, (int, float)) for v in pt) for pt in path):
            out.add(f'{p}.params.path', 'expected [[t, x, y], ...]')
        elif any(b[0] < a[0] for a, b in zip(path, path[1:])):
            out.add(f'{p}.params.path', 'path points must be in time order')
    cam = lay.params.get('cam')
    if cam is not None:
        try:
            ok = all(0 <= float(cam[k]) <= 1 for k in ('x', 'y', 'w', 'h'))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            out.add(f'{p}.params.cam', 'expected normalised {x, y, w, h} in 0..1')
    if lay.type in ('game_cam', 'reaction_cam') and cam is None:
        out.add(f'{p}.params.cam', f'{lay.type} needs the webcam rectangle')


def _transitions_and_overlaps(t, by_id, tp, out):
    trans = [it for it in t.items if it.type == 'transition']
    allowed = set()          # pairs of items joined by a transition may overlap by its duration
    for tr in trans:
        a, b = by_id.get(tr.from_item), by_id.get(tr.to_item)
        p = f'{tp}.items[{t.items.index(tr)}]'
        if a is None or b is None:
            out.add(p, 'from_item / to_item must be items of the same track')
            continue
        if a.start >= b.start:
            out.add(p, 'from_item must start before to_item')
        a_end = a.start + a.duration
        overlap = a_end - b.start
        if abs(tr.start - b.start) > 0.05:
            out.add(p, 'a transition starts where its second item starts')
        elif tr.kind == 'cut':
            if abs(overlap) > 0.05:
                out.add(p, 'a cut joins items that touch (no gap, no overlap)')
        elif abs(overlap - tr.duration) > 0.05:
            out.add(p, f'the items overlap by {overlap:.3f}s but the transition lasts {tr.duration:.3f}s')
        allowed.add((a.id, b.id))
    if t.type not in ('video',):
        return                      # overlays, audio and captions may stack
    media = sorted((it for it in t.items if it.type != 'transition'), key=lambda it: it.start)
    for a, b in zip(media, media[1:]):
        if b.start < a.start + a.duration - EPS and (a.id, b.id) not in allowed:
            out.add(f'{tp}', f'items {a.id} and {b.id} overlap on a video track without a transition')
