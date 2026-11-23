"""Opt-in retrieval for content the operator is authorized to download.
No authentication, cookies, DRM bypass, or arbitrary URL fetching.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from django.core.files import File
from django.db import close_old_connections

from .models import Project
from .youtube import parse_youtube_url

from django.conf import settings

MAX_BYTES = getattr(settings, 'MAX_SOURCE_BYTES', 8 * 1024 ** 3)
DOWNLOAD_TIMEOUT = int(float(os.getenv('CLIPFORGE_DOWNLOAD_MINUTES', '120')) * 60)   # long streams are big
# never accept a format without video: every fallback requires a video codec
FORMAT = 'bv*[height<=1080]+ba/b[height<=1080][vcodec!=none]/bv*+ba/b[vcodec!=none]'
# YouTube sometimes answers 403 Forbidden for the direct (https/DASH) streams of a video - more often for long
# ones. Its HLS (m3u8) streams are fetched in small fragments and usually still work; a lower resolution is the
# last resort. (name shown to the user, format selector)
STRATEGIES = [
    ('обычная загрузка', FORMAT),
    ('поток HLS', 'b[protocol^=m3u8][height<=1080][vcodec!=none]/bv*[protocol^=m3u8][height<=1080]+ba[protocol^=m3u8]'
                  '/b[protocol^=m3u8][vcodec!=none]'),
    ('720p', 'bv*[height<=720]+ba/b[height<=720][vcodec!=none]'),
]
FORBIDDEN_RX = re.compile(r'HTTP Error 403|403: Forbidden|Forbidden', re.I)


class Forbidden(RuntimeError):
    """yt-dlp got 403 Forbidden for the chosen stream - another strategy may still work."""
STALL_TIMEOUT = 300  # No output for five minutes (merging a long stream is silent)
POLL_SECONDS = 1
PROGRESS_RE = re.compile(r"CLIPFORGE_PROGRESS:\s*([\d.]+)%")


def _update(pk, progress, message):
    Project.objects.filter(pk=pk).update(status='fetching', progress=progress, message=message)


def _stop_tree(proc):
    """Kill yt-dlp and its ffmpeg children before temp folder cleanup."""
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15, check=False)
    else:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def js_runtime_args():
    """yt-dlp needs a JavaScript runtime for the full YouTube format list (otherwise video formats may be missing).
    deno is used automatically; Node.js has to be enabled explicitly."""
    import shutil
    if shutil.which('deno'):
        return []
    if shutil.which('node') and _ytdlp_version() >= (2025, 11):
        return ['--js-runtimes', 'node']
    return []


def _ytdlp_version():
    try:
        from yt_dlp.version import __version__ as v
        return tuple(int(x) for x in v.split('.')[:2])
    except Exception:
        return (0, 0)


def _keep_log(pk, log_path):
    """Copy yt-dlp's log to logs/ytdlp-<project>.log so a failed or odd download can be diagnosed later."""
    try:
        logs = Path(settings.BASE_DIR) / 'logs'
        logs.mkdir(exist_ok=True)
        (logs / f'ytdlp-{pk}.log').write_bytes(log_path.read_bytes()[-200000:])
    except Exception:
        pass


def _download(pk, video_id, destination, rep=None):
    """Try the strategies in order; only a 403 Forbidden moves on to the next one."""
    tried = []
    combined = destination / 'yt_dlp_all.log'
    try:
        for n, (name, fmt) in enumerate(STRATEGIES, 1):
            if n > 1:
                _clean_partial(destination)
                msg = f'YouTube отказал в загрузке (403) — пробуем ещё раз: {name} (попытка {n} из {len(STRATEGIES)})'
                if rep is not None:
                    rep.frac(0.0, msg)
                else:
                    _update(pk, 2, msg)
            try:
                _download_once(pk, video_id, destination, rep, fmt, combined, f'попытка {n}: {name}')
                return name
            except Forbidden:
                tried.append(name)
        raise RuntimeError(
            'YouTube запрещает загрузку этого видео (HTTP 403) всеми способами: ' + ', '.join(tried) + '.\n'
            'Что можно сделать:\n'
            '1) обновить yt-dlp — запустите setup.bat (он обновляет yt-dlp) и нажмите «Продолжить»;\n'
            '2) попробовать позже — такие отказы часто временные;\n'
            '3) скачать видео самостоятельно (если у вас есть права) и положить файл в папку source_library\\ '
            f'с именем {video_id}.mp4 или загрузить его файлом (до {MAX_BYTES / 1024 ** 3:.0f} ГБ).\n'
            f'Журнал: logs\\ytdlp-{pk}.log')
    finally:
        _keep_log(pk, combined if combined.exists() else destination / 'yt_dlp.log')


def _clean_partial(destination):
    """Remove half-downloaded pieces of a failed attempt (temporary download folder only)."""
    for f in destination.iterdir():
        if f.is_file() and not f.name.startswith('yt_dlp'):
            try:
                f.unlink()
            except OSError:
                pass


def _download_once(pk, video_id, destination, rep, fmt, combined, title):
    # sys.executable avoids depending on yt-dlp.exe being present in PATH.
    cmd = [
        sys.executable, '-m', 'yt_dlp', '--no-playlist',
        '--newline', '--progress-template', 'download:CLIPFORGE_PROGRESS:%(progress._percent_str)s',
        '--no-write-info-json', '--no-write-thumbnail',
        '--max-filesize', str(MAX_BYTES), '--socket-timeout', '20', '--retries', '3',
        '--fragment-retries', '10', '--merge-output-format', 'mp4',
        '-f', fmt, *js_runtime_args(),
        '-o', str(destination / '%(id)s.%(ext)s'),
        'https://www.youtube.com/watch?v=' + video_id,
    ]
    log_path = destination / 'yt_dlp.log'
    started = time.monotonic()
    last_output = started
    last_position = 0
    last_progress = 2
    proc = None
    try:
        # File-backed output avoids pipe deadlocks on large downloads.
        with log_path.open('w+b') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0))
            while True:
                time.sleep(POLL_SECONDS)
                if rep is not None:
                    rep.check()  # raises JobCancelled -> finally kills yt-dlp and children
                now = time.monotonic()
                log.flush()
                with log_path.open('rb') as reader:
                    reader.seek(last_position)
                    chunk = reader.read()
                    last_position = reader.tell()
                if chunk:
                    last_output = now
                    for match in PROGRESS_RE.finditer(chunk.decode('utf-8', errors='replace')):
                        percent = min(100.0, float(match.group(1)))
                        # 2..35% is the download stage, not total job completion.
                        if rep is not None:
                            rep.frac(percent / 100.0, f'Получаем видео: {percent:.1f}% загрузки')
                        else:
                            next_progress = max(last_progress, min(35, 2 + int(percent * 0.33)))
                            if next_progress > last_progress:
                                last_progress = next_progress
                                _update(pk, last_progress, f'Получаем видео: {percent:.1f}% загрузки')
                if proc.poll() is not None:
                    break
                if now - started > DOWNLOAD_TIMEOUT:
                    raise TimeoutError(f'Превышен лимит загрузки: {DOWNLOAD_TIMEOUT // 60} минут (CLIPFORGE_DOWNLOAD_MINUTES).')
                if now - last_output > STALL_TIMEOUT:
                    raise TimeoutError('yt-dlp не выводит новых сообщений более 5 минут.')
            if proc.returncode != 0:
                tail = log_path.read_bytes()[-4000:].decode('utf-8', errors='replace')
                if FORBIDDEN_RX.search(tail):
                    raise Forbidden(tail[-1500:])
                raise RuntimeError('yt-dlp завершился с ошибкой: ' + tail[-1500:])
    finally:
        if proc is not None:
            _stop_tree(proc)
        try:
            with combined.open('ab') as out:
                out.write(f'\n===== {title} =====\n'.encode('utf-8'))
                out.write(log_path.read_bytes()[-200000:])
        except OSError:
            pass


def fetch_source(pk, rep=None):
    """Download (or reuse an already downloaded copy of) the video and attach it to the project."""
    from pathlib import Path as _P
    p = Project.objects.get(pk=pk)
    video_id = parse_youtube_url(p.youtube_url)
    if not video_id:
        raise ValueError('Некорректная YouTube-ссылка.')
    # Same video already on disk in another project -> no second download.
    from .procutil import describe_streams, has_video
    for other in Project.objects.filter(youtube_url=p.youtube_url).exclude(pk=pk).exclude(source=''):
        try:
            if other.source and _P(other.source.path).is_file() and has_video(other.source.path):
                Project.objects.filter(pk=pk).update(source=other.source.name)
                if rep is not None:
                    rep.frac(1.0, 'Видео уже скачано в другом проекте — используем его')
                return
        except Exception:
            continue
    _update(pk, 2, 'Подключаемся к источнику...')
    with tempfile.TemporaryDirectory(prefix='clipforge_youtube_') as tmp:
        destination = Path(tmp)
        _download(pk, video_id, destination, rep)
        files = [f for f in destination.iterdir()
                 if f.is_file() and f.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')]
        if len(files) != 1:
            raise RuntimeError('После загрузки не найден ровно один готовый видеофайл.')
        src = files[0]
        if src.stat().st_size > MAX_BYTES:
            raise RuntimeError(f'Видео превышает лимит {MAX_BYTES / 1024 ** 3:.0f} ГБ '
                               '(можно поднять: set CLIPFORGE_MAX_SOURCE_GB=16).')
        if not has_video(src):
            hint = ('' if js_runtime_args() or __import__('shutil').which('deno') else
                    ' Скорее всего, yt-dlp не получил видеоформаты: для YouTube ему нужен JavaScript-движок — '
                    'установите deno (winget install DenoLand.Deno) или Node.js и нажмите «Продолжить».')
            raise RuntimeError(f'YouTube отдал файл без видеодорожки ({describe_streams(src)}).{hint} '
                               f'Журнал загрузки: logs\\ytdlp-{pk}.log')
        with src.open('rb') as stream:
            p.source.save(video_id + src.suffix, File(stream), save=True)


def retrieve_and_process(pk):
    """v4-compatible entry point (the job runner now does fetch + process itself)."""
    from .pipeline import process_project
    process_project(pk)
