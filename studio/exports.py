"""Export presets and validated per-project options (every value is whitelisted)."""
from .captions import STYLES, DEFAULT_STYLE

PRESETS = {
    '1080p': {'label': '1080×1920 (Full HD)', 'W': 1080, 'H': 1920},
    '720p': {'label': '720×1280 (быстрее, меньше файл)', 'W': 720, 'H': 1280},
}
# (x264 preset, x264 crf, nvenc cq)
QUALITY = {
    'high': {'label': 'Высокое', 'x264': ('fast', 19), 'nvenc': 19},
    'balanced': {'label': 'Баланс', 'x264': ('veryfast', 22), 'nvenc': 23},
    'small': {'label': 'Компактное', 'x264': ('veryfast', 26), 'nvenc': 28},
}
EDIT_MODES = [
    ('director', 'Auto Scene Director (сам выбирает по сценам)'),
    ('smart', 'Smart Reframe (лицо / центр)'),
    ('gameplay_facecam', 'Gameplay + Facecam (разделённый экран)'),
    ('fullframe', 'Без обрезки (весь кадр на размытом фоне: кино, интервью, вшитые титры)'),
    ('gameplay', 'Gameplay (сохранять обзор игры)'),
    ('center', 'Обычный центральный кроп'),
]
MODE_KEYS = [m for m, _ in EDIT_MODES]

DEFAULTS = {
    'preset': '1080p', 'quality': 'balanced', 'fps': 'source', 'encoder': 'auto',
    'captions': True, 'subtitle_style': DEFAULT_STYLE, 'caption_words': 0, 'caption_animate': True,
    'caption_highlight': None,
    'bg_fill': 'blur', 'director_fallback': 'gameplay', 'transition': 'none', 'smart_tracking': True,
    'hide_cam': True, 'cam_position': 'bottom', 'highlight_engine': 'smart', 'review_first': False,
    'gameplay_zoom': 1.0, 'text_guard': True,
}
CHOICES = {
    'preset': set(PRESETS), 'quality': set(QUALITY), 'fps': {'source', '24', '30', '60'},
    'encoder': {'auto', 'cpu'}, 'subtitle_style': set(STYLES) | {'classic'},
    'bg_fill': {'blur', 'solid', 'focus'}, 'director_fallback': {'gameplay', 'smart'},
    'transition': {'none', 'fade'}, 'cam_position': {'bottom', 'top'}, 'highlight_engine': {'smart', 'classic'},
}
BOOLS = ('captions', 'caption_animate', 'smart_tracking', 'hide_cam', 'review_first', 'text_guard')


def _b(v):
    if isinstance(v, str):
        return v.lower() in ('1', 'true', 'on', 'yes')
    return bool(v)


def resolve(*layers):
    """Merge option dicts (later wins), keeping only valid values."""
    out = dict(DEFAULTS)
    for layer in layers:
        if not layer:
            continue
        for key, val in layer.items():
            if key not in DEFAULTS or val is None and key != 'caption_highlight':
                continue
            if key in CHOICES:
                if str(val) in CHOICES[key]:
                    out[key] = str(val)
            elif key in BOOLS:
                out[key] = _b(val)
            elif key == 'caption_words':
                try:
                    out[key] = max(0, min(6, int(val)))
                except (TypeError, ValueError):
                    pass
            elif key == 'gameplay_zoom':
                try:
                    out[key] = max(1.0, min(1.5, float(val)))
                except (TypeError, ValueError):
                    pass
            elif key == 'caption_highlight':
                out[key] = None if val in (None, '', 'auto') else _b(val)
    if out['bg_fill'] == 'focus':
        out['gameplay_zoom'] = max(out['gameplay_zoom'], 1.3)
    return out


def dims(opts):
    p = PRESETS[opts['preset']]
    return p['W'], p['H']


def encoder_args(encoder, opts):
    q = QUALITY[opts['quality']]
    if encoder == 'h264_nvenc':
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', str(q['nvenc']), '-b:v', '0',
                '-profile:v', 'high', '-pix_fmt', 'yuv420p']
    preset, crf = q['x264']
    return ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p']
