import shutil
import tempfile
from pathlib import Path

from . import synth

HAVE_FFMPEG = bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))
_cache = {}


def sample_video(kind='mix'):
    """One shared synthetic 16 s video (4 s per scene): game+cam -> talking -> gameplay -> screen."""
    if kind not in _cache:
        d = Path(tempfile.mkdtemp(prefix='cf_test_'))
        tl = [(4, 'game_cam'), (4, 'talking'), (4, 'gameplay'), (4, 'screen')]
        _cache[kind] = synth.make_video(d / 'mix.mp4', timeline=tl)
    return _cache[kind]


def fake_transcript(duration=16.0):
    """Sentences with word timings, ~3 words/s, RU, with questions and a conclusion."""
    sents = ['Почему это работает именно так?', 'Я расскажу вам секрет который никто не знает.',
             'Смотрите внимательно, это невероятно просто!', 'Но потом вдруг всё меняется и оказывается наоборот.',
             'Поэтому запомните главное правило этого видео.', 'Вот так мы пришли к победе в матче.',
             'Теперь второй пример, он ещё лучше первого.', 'И в итоге всё получилось отлично.']
    segs, t = [], 0.2
    for s in sents:
        words, st = [], t
        for w in s.split():
            words.append({'start': round(t, 2), 'end': round(t + 0.3, 2), 'text': w})
            t += 0.34
        segs.append({'start': st, 'end': round(t, 2), 'text': s, 'words': words})
        t += 0.45
        if t > duration - 0.5:
            break
    return segs


def film_video(subs=True):
    key = 'film' if subs else 'film_plain'
    if key not in _cache:
        d = Path(tempfile.mkdtemp(prefix='cf_film_'))
        _cache[key] = synth.make_video(d / 'film.mp4', timeline=[(6, key)])
    return _cache[key]
