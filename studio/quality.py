"""Quality Validator: automatic checks on a rendered clip. Everything is measured, nothing is guessed."""
import json
import re
import subprocess

from . import textutil as tx
from .procutil import run_capture


def probe_streams(path):
    d = json.loads(run_capture(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type,duration,width,height',
                                '-of', 'json', str(path)], timeout=30))
    out = {'format': float(d['format'].get('duration') or 0), 'video': None, 'audio': None, 'w': 0, 'h': 0}
    for s in d.get('streams', []):
        dur = float(s['duration']) if s.get('duration') not in (None, 'N/A') else None
        if s['codec_type'] == 'video':
            out['video'], out['w'], out['h'] = dur or out['format'], s.get('width', 0), s.get('height', 0)
        elif s['codec_type'] == 'audio':
            out['audio'] = dur or out['format']
    return out


def parse_detect(stderr, total):
    black, silence, s_start = [], [], None
    for line in stderr.splitlines():
        m = re.search(r'black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)', line)
        if m:
            black.append((float(m.group(1)), float(m.group(2))))
        m = re.search(r'silence_start:\s*(-?[\d.]+)', line)
        if m:
            s_start = max(0.0, float(m.group(1)))
        m = re.search(r'silence_end:\s*([\d.]+)', line)
        if m and s_start is not None:
            silence.append((s_start, float(m.group(1))))
            s_start = None
    if s_start is not None:
        silence.append((s_start, total))
    return black, silence


def run_detectors(path, has_audio=True, timeout=300):
    cmd = ['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-vf', 'blackdetect=d=0.4:pix_th=0.10']
    if has_audio:
        cmd += ['-af', 'silencedetect=noise=-38dB:d=2.0']
    cmd += ['-f', 'null', '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                       timeout=timeout)
    return p.stderr


def _check(cid, status, label, detail=''):
    return {'id': cid, 'status': status, 'label': label, 'detail': detail}


def validate_clip(path, expected, source_has_audio=True, caption_report=None, text='', siblings=(), clip_min=15,
                  clip_max=60, span=None):
    """siblings: [(text, (start, end))] of the other clips in the project (duplicate detection)."""
    checks = []
    st = probe_streams(path)
    dur = st['video'] or st['format']
    diff = abs(dur - expected)
    checks.append(_check('duration', 'ok' if diff <= 0.6 else ('warn' if diff <= 1.5 else 'fail'), 'Длительность соответствует плану',
                         f'{dur:.1f} с (план {expected:.1f} с)'))
    if source_has_audio:
        if st['audio'] is None:
            checks.append(_check('av_sync', 'fail', 'Аудио-дорожка', 'в готовом клипе нет звука'))
        else:
            d = abs(st['audio'] - dur)
            checks.append(_check('av_sync', 'ok' if d <= 0.35 else ('warn' if d <= 1.0 else 'fail'), 'Синхронизация звука и видео',
                                 f'расхождение длительностей {d:.2f} с'))
    try:
        err = run_detectors(path, has_audio=st['audio'] is not None)
        black, silence = parse_detect(err, dur)
    except Exception as exc:
        black, silence = [], []
        checks.append(_check('detectors', 'warn', 'Проверка чёрных кадров и тишины', f'не выполнена: {str(exc)[:80]}'))
    else:
        tb = sum(b - a for a, b in black)
        longest = max((b - a for a, b in black), default=0)
        status = 'ok'
        if tb / max(dur, 1) > 0.25:
            status = 'fail'
        elif tb / max(dur, 1) > 0.10 or longest >= 1.0:
            status = 'warn'
        checks.append(_check('black', status, 'Чёрные кадры', 'не найдены' if not black else f'всего {tb:.1f} с, самый длинный {longest:.1f} с'))
        ts = sum(b - a for a, b in silence)
        long_sil = max((b - a for a, b in silence), default=0)
        status = 'warn' if (long_sil >= 2.5 or ts / max(dur, 1) > 0.40) else 'ok'
        checks.append(_check('silence', status, 'Длинная тишина', 'не найдена' if not silence else
                             f'самая длинная пауза {long_sil:.1f} с, всего {ts:.1f} с'))
    if dur < 10:
        checks.append(_check('length', 'warn', 'Длина клипа', f'{dur:.0f} с — слишком коротко'))
    elif dur > max(clip_max + 5, 90):
        checks.append(_check('length', 'warn', 'Длина клипа', f'{dur:.0f} с — длиннее заданного максимума'))
    else:
        checks.append(_check('length', 'ok', 'Длина клипа', f'{dur:.0f} с'))
    if caption_report and caption_report.get('enabled', True):
        if caption_report.get('est_overflow', 0):
            checks.append(_check('captions', 'fail', 'Субтитры помещаются в кадр', f'{caption_report["est_overflow"]} подписей выходят за край'))
        else:
            sh = caption_report.get('shrunk', 0)
            checks.append(_check('captions', 'ok', 'Субтитры помещаются в кадр',
                                 'все подписи в безопасной зоне' if not sh else f'{sh} длинных подписей автоматически уменьшены/перенесены'))
        if caption_report.get('words_reliable') is False:
            checks.append(_check('captions_timing', 'warn', 'Таймкоды слов', 'ненадёжны — подсветка слов отключена'))
    mine = tx.keyword_set(text)
    for other_text, (a, b) in siblings:
        overlap = 0.0
        if span:
            overlap = max(0.0, min(span[1], b) - max(span[0], a)) / max(0.1, min(span[1] - span[0], b - a))
        sim = tx.jaccard(mine, tx.keyword_set(other_text))
        if overlap > 0.5 or sim > 0.6:
            checks.append(_check('duplicate', 'warn', 'Повтор содержания', 'похож на другой клип этого проекта'))
            break
    else:
        checks.append(_check('duplicate', 'ok', 'Повтор содержания', 'уникальный клип'))
    score = 100
    for c in checks:
        score -= {'fail': 30, 'warn': 10}.get(c['status'], 0)
    return {'score': max(0, score), 'ok': not any(c['status'] == 'fail' for c in checks), 'checks': checks,
            'size': f'{st["w"]}x{st["h"]}'}


def make_poster(video, dest, at=1.0):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{at:.2f}', '-i', str(video), '-frames:v', '1',
                    '-vf', 'scale=360:-2', '-q:v', '5', str(dest)], timeout=60, check=False)
    return dest.exists() and dest.stat().st_size > 0
