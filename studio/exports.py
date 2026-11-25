"""Export presets and validated per-project options (every value is whitelisted)."""
import re

from .captions import STYLES

STREAM_STYLE = 'stream'

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
    ('auto', 'AI Auto (сам выбирает режим для каждого клипа)'),
    ('director', 'Auto Scene Director (сам выбирает по сценам)'),
    ('reaction', 'Reaction Priority (стримы: вебка сверху 40%, игра снизу 60%)'),
    ('smart', 'Smart Reframe (лицо / центр)'),
    ('gameplay_facecam', 'Gameplay Priority (Gameplay + Facecam: игра сверху, вебка снизу)'),
    ('fullframe', 'Без обрезки (весь кадр на размытом фоне: кино, интервью, вшитые титры)'),
    ('gameplay', 'Gameplay (сохранять обзор игры)'),
    ('center', 'Обычный центральный кроп'),
]
MODE_KEYS = [m for m, _ in EDIT_MODES]

DEFAULTS = {
    'preset': '1080p', 'quality': 'balanced', 'fps': '30', 'encoder': 'auto',
    'captions': True, 'subtitle_style': STREAM_STYLE, 'caption_words': 0, 'caption_animate': True,
    'caption_highlight': None,
    'bg_fill': 'blur', 'director_fallback': 'gameplay', 'transition': 'none', 'smart_tracking': True,
    'hide_cam': True, 'cam_position': 'bottom', 'highlight_engine': 'smart', 'review_first': False,
    'gameplay_zoom': 1.0, 'text_guard': True,
    # v5.1: Gameplay + Facecam geometry / duplicate-webcam removal, multimodal highlight scoring
    'cam_fit': 'balanced', 'cam_fill': 'mirror', 'highlight_audio': True, 'highlight_motion': True,
    # v5.2: stream editor
    'loudness': 'stream', 'auto_length': False,
    # v5.3: the streamer's name (optional, typed by the user) for Shorts titles
    'streamer': '',
    # v5.4: sound classifier, caption font and swearing filter
    'highlight_sound_ai': True, 'caption_font': 'style', 'caption_censor': False,
    # v6 (2A): Hook Engine - off | suggest (analyse, show variants) | auto (also apply the recommended one)
    'hook_engine': 'suggest',
    # v6 (2B): Smart Editing - off | natural | dynamic | aggressive (cuts pauses / repeats / dull parts)
    'smart_edit': 'off',
    # Smart Editing switches: silence / dead-air removal and Smart Auto Zoom (both inside the chosen mode)
    'smart_silence': True, 'smart_zoom': True,
    # v6 (2C): Viral Director - off | rank (score, explain, mark duplicates) | auto (also pick the best clips + combo)
    'viral_director': 'rank',
}
CHOICES = {
    'preset': set(PRESETS), 'quality': set(QUALITY), 'fps': {'source', '24', '30', '60'},
    'encoder': {'auto', 'cpu'}, 'subtitle_style': set(STYLES) | {'classic'},
    'bg_fill': {'blur', 'solid', 'focus'}, 'director_fallback': {'gameplay', 'smart'},
    'transition': {'none', 'fade'}, 'cam_position': {'bottom', 'top'}, 'highlight_engine': {'smart', 'classic'},
    'cam_fit': {'full', 'balanced', 'fill'}, 'cam_fill': {'mirror', 'blur'}, 'loudness': {'stream', 'off'},
    'caption_font': {'style', 'Montserrat', 'Arial Black', 'Arial', 'Impact'},
    'hook_engine': {'off', 'suggest', 'auto'},
    'smart_edit': {'off', 'natural', 'dynamic', 'aggressive'},
    'viral_director': {'off', 'rank', 'auto'},
}
LOUDNESS = {'stream': {'I': -14.0, 'TP': -1.5, 'LRA': 11.0}}
BOOLS = ('captions', 'caption_animate', 'smart_tracking', 'hide_cam', 'review_first', 'text_guard', 'highlight_audio',
         'highlight_motion', 'auto_length', 'highlight_sound_ai', 'caption_censor', 'smart_silence', 'smart_zoom')


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
            elif key == 'streamer':
                out[key] = re.sub(r'[^\w .\-]', '', str(val), flags=re.U).strip()[:40]
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
