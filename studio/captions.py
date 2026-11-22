"""Caption Styler: ASS subtitle generation with 5 styles, safe margins and per-layout vertical position.

`classic` is the legacy generator (pipeline.captions_ass) kept as a backup.
"""
import re

from . import textutil as tx

STYLES = {
    'clean': {'label': 'Clean Minimal', 'font': 'Arial', 'size': 58, 'bold': 0, 'italic': 0, 'primary': '#FFFFFF',
              'active': '#FFFFFF', 'outline_c': '#000000', 'outline': 3, 'shadow': 1, 'upper': False, 'box': False,
              'max_words': 5, 'highlight': False, 'anim': 'fade', 'wk': 0.52},
    'bold': {'label': 'Bold Shorts', 'font': 'Arial', 'size': 84, 'bold': 1, 'italic': 0, 'primary': '#FFFFFF',
             'active': '#FFE14D', 'outline_c': '#000000', 'outline': 7, 'shadow': 2, 'upper': True, 'box': False,
             'max_words': 3, 'highlight': True, 'anim': 'pop', 'wk': 0.62},
    'gaming': {'label': 'Gaming', 'font': 'Impact', 'size': 90, 'bold': 0, 'italic': 0, 'primary': '#FFFFFF',
               'active': '#39FF14', 'outline_c': '#101010', 'outline': 8, 'shadow': 3, 'upper': True, 'box': False,
               'max_words': 3, 'highlight': True, 'anim': 'pop', 'wk': 0.50},
    'karaoke': {'label': 'Karaoke Highlight', 'font': 'Arial', 'size': 72, 'bold': 1, 'italic': 0, 'primary': '#FFFFFF',
                'active': '#FFC400', 'outline_c': '#000000', 'outline': 5, 'shadow': 1, 'upper': False, 'box': False,
                'max_words': 4, 'highlight': True, 'anim': 'fade', 'wk': 0.58},
    'cinematic': {'label': 'Cinematic', 'font': 'Georgia', 'size': 54, 'bold': 0, 'italic': 1, 'primary': '#F2EBDD',
                  'active': '#F2EBDD', 'outline_c': '#000000', 'outline': 2, 'shadow': 3, 'upper': False, 'box': False,
                  'max_words': 5, 'highlight': False, 'anim': 'fade', 'wk': 0.50},
    # one word at a time, big bold white with a black outline; key/emotional words turn yellow
    'stream': {'label': 'Stream (по одному слову, ключевые — жёлтым)', 'font': 'Arial', 'size': 104, 'bold': 1,
               'italic': 0, 'primary': '#FFFFFF', 'active': '#FFE14D', 'outline_c': '#000000', 'outline': 9,
               'shadow': 2, 'upper': True, 'box': False, 'max_words': 1, 'highlight': False, 'key_highlight': True,
               'anim': 'pop', 'wk': 0.64},
}
STYLE_CHOICES = [('classic', 'Classic (как в v4, резервный)')] + [(k, v['label']) for k, v in STYLES.items()]
DEFAULT_STYLE = 'clean'


def ass_color(hex_rgb, alpha=0):
    h = hex_rgb.lstrip('#')
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f'&H{alpha:02X}{b}{g}{r}'.upper()


def _esc(text):
    return text.replace('\\', '').replace('{', '(').replace('}', ')').replace('\n', ' ')


def ass_time(sec):
    cs = max(0, round(sec * 100))
    return f'{cs // 360000}:{(cs // 6000) % 60:02}:{(cs // 100) % 60:02}.{cs % 100:02}'


def _clean_word(t):
    t = t.strip()
    return re.sub(r'\s+', ' ', t)


def _clean_group_text(words):
    """Group-level tidy: no dangling comma at the end, no orphan punctuation at the start."""
    texts = [_clean_word(w['text']) for w in words]
    texts = [t for t in texts if t]
    if not texts:
        return texts
    texts[0] = re.sub(r'^[,;:.\-–—]+\s*', '', texts[0]) or texts[0]
    texts[-1] = re.sub(r'[,;:\-–—]+$', '', texts[-1])
    return [t for t in texts if t]


def est_width(text, size, wk, upper):
    return len(text) * size * wk * (1.12 if upper else 1.0)


def group_words(words, max_words, max_chars, max_gap=0.4):
    groups, cur = [], []
    for w in words:
        t = _clean_word(w['text'])
        if not t:
            continue
        if cur and (w['start'] - cur[-1]['end'] > max_gap or len(cur) >= max_words or
                    sum(len(x['text']) + 1 for x in cur) + len(t) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r'[.!?…]["»”)]?$', t):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    # merge dangling single words into a neighbour when it reads naturally
    merged = []
    for g in groups:
        if merged and len(g) == 1 and len(merged[-1]) + 1 <= max_words and g[0]['start'] - merged[-1][-1]['end'] < 0.15 \
                and not re.search(r'[.!?…]["»”)]?$', merged[-1][-1]['text']):
            merged[-1].append(g[0])
        else:
            merged.append(g)
    return merged


def _segment_fallback_words(segments, start, end):
    """No reliable word timing: spread each segment's words evenly (no per-word highlight)."""
    out = []
    for seg in segments:
        if seg['end'] <= start or seg['start'] >= end:
            continue
        parts = seg['text'].split()
        if not parts:
            continue
        step = (seg['end'] - seg['start']) / len(parts)
        for k, t in enumerate(parts):
            out.append({'start': seg['start'] + k * step, 'end': seg['start'] + (k + 1) * step, 'text': t})
    return out


def _balanced_break(texts):
    best, best_i = 1e9, 1
    for i in range(1, len(texts)):
        a = len(' '.join(texts[:i]))
        b = len(' '.join(texts[i:]))
        if abs(a - b) < best:
            best, best_i = abs(a - b), i
    return best_i


def _norm(t):
    return t.strip('.,!?…:;"«»()-–—').lower().replace('ё', 'е')


def _key_words(words):
    """Words to paint yellow: reaction cues (swearing, fails, exclamations...), emotional/hook words, numbers,
    shouting in caps and the clip's main topic words."""
    from . import highlights, moments
    text = ' '.join(w['text'] for w in words)
    keys = {_norm(k) for k in moments.key_words(text)}
    for w in words:
        t = w['text']
        low = t.lower()
        if (highlights.EMO.search(low) or highlights.HOOK.search(low) or
                (highlights.SURPRISE.search(low) and len(_norm(t)) > 3) or t.rstrip('"»”)').endswith(('!', '?!'))):
            keys.add(_norm(t))
    keys |= {_norm(k) for k in tx.keywords(text, n=3)}
    return {k for k in keys if k}


def build_ass(segments, start, end, dest, W=1080, H=1920, style_id=DEFAULT_STYLE, max_words=None,
              margin_for=None, animate=True, highlight=None, font=None):
    """Write an ASS file for [start,end) (times shifted to 0). Returns a report dict.

    margin_for(t_rel) -> MarginV in px at the *output* resolution (None -> default safe value).
    """
    st = dict(STYLES.get(style_id) or STYLES[DEFAULT_STYLE])
    if font:
        st['font'] = font
    scale = W / 1080.0
    size = round(st['size'] * scale)
    mwords = max(1, min(8, int(max_words or st['max_words'])))
    do_hl = st['highlight'] if highlight is None else bool(highlight)
    side = round(110 * scale)
    avail = W - 2 * side
    default_margin = round(H * 0.22)

    words = tx.flat_words([s for s in segments if s['end'] > start and s['start'] < end])
    words = [w for w in words if w['end'] > start and w['start'] < end]
    reliable = tx.words_reliable(words)
    if not reliable:
        words = _segment_fallback_words(segments, start, end)
        do_hl = False
    max_chars = max(8, int(avail / (size * st['wk'] * (1.12 if st['upper'] else 1.0))) * 2 - 2)  # up to two lines
    groups = group_words(words, mwords, max_chars) if mwords > 1 else [[w] for w in words if _clean_word(w['text'])]
    keys = _key_words(words) if st.get('key_highlight') and highlight is not False else set()

    margins = {}
    events = []
    shrunk = 0
    for gi, g in enumerate(groups):
        texts = _clean_group_text(g)
        if not texts:
            continue
        shown = [t.upper() if st['upper'] else t for t in texts]
        a = max(start, g[0]['start']) - start
        b_word = min(end, g[-1]['end']) - start
        nxt = groups[gi + 1][0]['start'] - start if gi + 1 < len(groups) else (end - start)
        b = min(max(b_word + 0.22, a + 0.4), nxt - 0.01 if nxt - 0.01 > a + 0.15 else b_word, end - start)
        if b <= a:
            continue
        mv = int(margin_for((a + b) / 2) if margin_for else default_margin)
        margins.setdefault(mv, f'S{len(margins)}')
        sname = margins[mv]
        # layout: one line if it fits, else two balanced lines, else shrink
        full = ' '.join(shown)
        fs = size
        two_lines = False
        if est_width(full, size, st['wk'], False) > avail:
            if len(shown) >= 2:
                k = _balanced_break(shown)
                widest = max(est_width(' '.join(shown[:k]), size, st['wk'], False),
                             est_width(' '.join(shown[k:]), size, st['wk'], False))
                two_lines = True
                if widest > avail:
                    fs = max(round(size * 0.7), int(size * avail / widest))
                    shrunk += 1
            else:
                fs = max(round(size * 0.7), int(size * avail / est_width(full, size, st['wk'], False)))
                shrunk += 1
        brk = _balanced_break(shown) if two_lines else None
        prefix = '' if fs == size else r'{\fs%d}' % fs

        def compose(active):
            out = []
            for idx, t in enumerate(shown):
                piece = _esc(t)
                if active is not None and idx == active:
                    piece = r'{\1c%s}%s{\1c}' % (ass_color(st['active']), piece)
                out.append(piece)
                if brk is not None and idx == brk - 1:
                    out.append(r'\N')
                elif idx < len(shown) - 1:
                    out.append(' ')
            return ''.join(out).replace(' \\N', '\\N').replace('\\N ', '\\N')

        anim = ''
        if animate and st['anim'] == 'pop':
            anim = r'{\fscx88\fscy88\t(0,110,\fscx100\fscy100)}'
        elif animate and st['anim'] == 'fade':
            anim = r'{\fad(90,60)}'
        if do_hl and len(g) == len(texts):
            starts = [max(start, w['start']) - start for w in g]
            for k in range(len(g)):
                ea = a if k == 0 else starts[k]
                eb = (starts[k + 1] if k + 1 < len(g) else b)
                eb = min(eb, b)
                if eb - ea < 0.02:
                    continue
                events.append((ea, eb, sname, (anim if k == 0 else '') + prefix + compose(k)))
        elif keys and len(shown) == 1 and _norm(texts[0]) in keys:
            events.append((a, b, sname, anim + prefix + compose(0)))
        else:
            events.append((a, b, sname, anim + prefix + compose(None)))

    lines = [f'[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n',
             '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, '
             'Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
             'Alignment, MarginL, MarginR, MarginV, Encoding\n']
    if not margins:
        margins[default_margin] = 'S0'
    for mv, name in margins.items():
        lines.append(f'Style: {name},{st["font"]},{size},{ass_color(st["primary"])},{ass_color(st["active"])},'
                     f'{ass_color(st["outline_c"])},{ass_color("#000000", 0x70)},{st["bold"] and -1 or 0},'
                     f'{st["italic"] and -1 or 0},0,0,100,100,0,0,1,{round(st["outline"] * scale)},'
                     f'{round(st["shadow"] * scale)},2,{side},{side},{mv},1\n')
    lines.append('\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
    for a, b, sname, text in events:
        lines.append(f'Dialogue: 0,{ass_time(a)},{ass_time(b)},{sname},,0,0,0,,{text}\n')
    dest.write_text(''.join(lines), encoding='utf-8')
    return {'mode': 'word' if ((do_hl or mwords == 1) and reliable) else 'segment', 'events': len(events), 'shrunk': shrunk,
            'key_words': len(keys),
            'groups': len(groups), 'words_reliable': reliable, 'style': style_id, 'max_words': mwords,
            'est_overflow': 0}
