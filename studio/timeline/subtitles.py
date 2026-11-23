"""Text items + caption items of a timeline -> one ASS file (burned in by the final pass).

* Captions whose style came from the Stream Clipper (Style.ext['clipforge.style'].name is a captions.STYLES key)
  are written by captions.build_ass itself, so they look exactly like the classic renderer's captions
  (same grouping, pop animation, key-word highlight, per-layout margins).
* Any other style is drawn from the Style fields: font, size (px at a 1080-wide canvas, scaled to the canvas),
  colours, outline, uppercase, max_words, position / margin_v; emphasised words use highlight_color.
* Text items: position x/y (0..1 of the canvas), align, and animations x/y (one straight move), scale, rotation
  (degrees, clockwise), opacity. Times are on the timeline; keyframe times are relative to the item.
"""
import tempfile
from pathlib import Path

from .. import captions as cap
from . import model as m
from .compile import n

ACCEL = {'linear': 1.0, 'ease_in': 2.0, 'ease_out': 0.5}
POS_ALIGN = {'bottom': 2, 'custom': 2, 'middle': 5, 'top': 8}
TEXT_ALIGN = {'left': 4, 'center': 5, 'right': 6}


def _esc(text):
    return '\\N'.join(cap._esc(line) for line in str(text).split('\n'))


def _alpha(opacity):
    return f'&H{max(0, min(255, round((1 - float(opacity)) * 255))):02X}&'


def _style_line(name, st, W, align, margin_v):
    k = W / 1080.0
    return (f'Style: {name},{st.font},{round(st.size * k)},{cap.ass_color(st.color)},{cap.ass_color(st.highlight_color)},'
            f'{cap.ass_color(st.outline_color)},{cap.ass_color("#000000", 0x70)},{-1 if st.bold else 0},0,0,0,100,100,0,0,'
            f'1,{n(st.outline * k)},{n(2 * k)},{align},{round(60 * k)},{round(60 * k)},{int(margin_v)},1\n')


def _margin_fn(margins, H):
    default = round(H * 0.22)

    def at(t):
        for a, b, mv in margins:
            if a <= t < b:
                return mv
        return margins[-1][2] if margins else default
    return at


def _cf_captions(it, st, tl, margin_for, prefix):
    """Stream Clipper captions via captions.build_ass -> (style lines, dialogue lines, report)."""
    ext = st.ext.get('clipforge.style') or {}
    words = [{'start': w.t0, 'end': w.t1, 'text': w.text} for w in it.words]
    seg = [{'start': words[0]['start'], 'end': words[-1]['end'], 'text': ' '.join(w['text'] for w in words),
            'words': words}]
    font_choice = ext.get('font_choice')
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'c.ass'
        rep = cap.build_ass(seg, 0.0, it.start + it.duration, f, tl.canvas.width, tl.canvas.height, ext['name'],
                            font=font_choice if font_choice in cap.FONTS else st.font,
                            max_words=st.max_words, margin_for=margin_for, animate=ext.get('animate', True),
                            highlight=ext.get('highlight'))
        text = f.read_text(encoding='utf-8')
    styles, events = [], []
    for line in text.splitlines(True):
        if line.startswith('Style: '):
            name = line[7:].split(',', 1)[0]
            styles.append(f'Style: {prefix}{name},' + line[7:].split(',', 1)[1])
        elif line.startswith('Dialogue: '):
            parts = line.split(',', 4)
            parts[3] = prefix + parts[3]
            events.append(','.join(parts))
    return styles, events, rep


def _generic_captions(it, st, tl, margin_for, name):
    W, H = tl.canvas.width, tl.canvas.height
    align = POS_ALIGN[st.position]
    mv = st.margin_v if st.margin_v is not None else (
        0 if st.position == 'middle' else round(H * 0.08) if st.position == 'top' else int(margin_for(it.start)))
    words = [{'start': w.t0, 'end': w.t1, 'text': w.text, 'em': w.emphasis} for w in it.words]
    groups = cap.group_words(words, st.max_words, 32) if st.max_words > 1 else [[w] for w in words]
    events = []
    end_item = it.start + it.duration
    for gi, grp in enumerate(groups):
        a = grp[0]['start']
        nxt = groups[gi + 1][0]['start'] if gi + 1 < len(groups) else end_item
        b = min(max(grp[-1]['end'] + 0.15, a + 0.3), nxt, end_item)
        if b <= a:
            continue
        pieces = []
        for w in grp:
            t = _esc(w['text'].upper() if st.uppercase else w['text'])
            pieces.append(r'{\1c%s}%s{\1c}' % (cap.ass_color(st.highlight_color), t) if w['em'] else t)
        events.append(f'Dialogue: 0,{cap.ass_time(a)},{cap.ass_time(b)},{name},,0,0,0,,{" ".join(pieces)}\n')
    return [_style_line(name, st, W, align, mv)], events, {'mode': 'generic', 'events': len(events)}


def _text_event(it, st, tl, name):
    W, H = tl.canvas.width, tl.canvas.height
    an = {a.property: a.keyframes for a in it.animations}
    tags = [rf'\an{TEXT_ALIGN[it.align]}']
    xs, ys = an.get('x'), an.get('y')
    if xs or ys:
        kx = xs or [m.Keyframe(t=k.t, value=it.x) for k in ys]
        ky = ys or [m.Keyframe(t=k.t, value=it.y) for k in xs]
        x0, y0 = kx[0].value * W, ky[0].value * H
        if len(kx) == 1:
            tags.append(rf'\pos({n(x0)},{n(y0)})')
        else:
            t0, t1 = round(kx[0].t * 1000), round(kx[1].t * 1000)
            if kx[1].easing == 'hold':
                t0 = t1
            tags.append(rf'\move({n(x0)},{n(y0)},{n(kx[1].value * W)},{n(ky[1].value * H)},{t0},{max(t0, t1)})')
    else:
        tags.append(rf'\pos({n(it.x * W)},{n(it.y * H)})')

    def curve(prop, fmt):
        ks = an.get(prop)
        if not ks:
            return
        tags.append(fmt(ks[0].value))
        for a, b in zip(ks, ks[1:]):
            ta, tb = round(a.t * 1000), round(b.t * 1000)
            if b.easing == 'hold':
                ta = tb
            tags.append(rf'\t({ta},{max(ta, tb)},{n(ACCEL.get(b.easing, 1.0))},{fmt(b.value)})')
    curve('scale', lambda v: rf'\fscx{n(v * 100)}\fscy{n(v * 100)}')
    curve('rotation', lambda v: rf'\frz{n(-v)}')
    curve('opacity', lambda v: rf'\alpha{_alpha(v)}')
    text = it.text.upper() if st.uppercase else it.text
    a, b = it.start, it.start + it.duration
    return f'Dialogue: 1,{cap.ass_time(a)},{cap.ass_time(b)},{name},,0,0,0,,{{{"".join(tags)}}}{_esc(text)}\n'


def write_ass(tl, dest, margins):
    """-> report dict, or None when the timeline has no text and no captions (no ASS pass needed)."""
    W, H = tl.canvas.width, tl.canvas.height
    styles_by_id = {s.id: s for s in tl.styles}
    default = m.Style(id='st_default0', size=72, max_words=3)
    margin_for = _margin_fn(sorted(margins), H)
    style_lines, events, reports = [], [], []
    for ti, track in enumerate(tl.tracks):
        if track.hidden or track.type not in ('caption', 'overlay'):
            continue
        for ii, it in enumerate(track.items):
            st = styles_by_id.get(it.style, default) if it.type in ('caption', 'text') else None
            if it.type == 'caption' and it.words:
                prefix = f'C{ti}x{ii}'
                name = (st.ext.get('clipforge.style') or {}).get('name')
                fn = _cf_captions if name in cap.STYLES else _generic_captions
                s, e, rep = fn(it, st, tl, margin_for, prefix if fn is _cf_captions else prefix + 'S')
                style_lines += s
                events += e
                reports.append(dict(rep, item=it.id))
            elif it.type == 'text':
                name = f'T{ti}x{ii}'
                style_lines.append(_style_line(name, st, W, 5, 0))
                events.append(_text_event(it, st, tl, name))
    if not events:
        return None
    head = (f'[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\n'
            'ScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, '
            'SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, '
            'Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n')
    body = '\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    Path(dest).write_text(head + ''.join(style_lines) + body + ''.join(events), encoding='utf-8')
    return {'items': reports, 'events': len(events)}
