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

MAX_BYTES = 1024 ** 3
DOWNLOAD_TIMEOUT = 1800  # 30 minutes overall
STALL_TIMEOUT = 180  # No output for three minutes
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


def _download(pk, video_id, destination, rep=None):
    # sys.executable avoids depending on yt-dlp.exe being present in PATH.
    cmd = [
        sys.executable, '-m', 'yt_dlp', '--no-playlist',
        '--newline', '--progress-template', 'download:CLIPFORGE_PROGRESS:%(progress._percent_str)s',
        '--no-write-info-json', '--no-write-thumbnail',
        '--max-filesize', '1G', '--socket-timeout', '20', '--retries', '2',
        '--fragment-retries', '2', '--merge-output-format', 'mp4',
        '-f', 'bv*[height<=1080]+ba/b[height<=1080]/best',
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
                    raise TimeoutError('Превышен лимит загрузки: 30 минут.')
                if now - last_output > STALL_TIMEOUT:
                    raise TimeoutError('yt-dlp не выводит новых сообщений более 3 минут.')
            if proc.returncode != 0:
                tail = log_path.read_bytes()[-4000:].decode('utf-8', errors='replace')
                raise RuntimeError('yt-dlp завершился с ошибкой: ' + tail[-1500:])
    finally:
        if proc is not None:
            _stop_tree(proc)


def fetch_source(pk, rep=None):
    """Download (or reuse an already downloaded copy of) the video and attach it to the project."""
    from pathlib import Path as _P
    p = Project.objects.get(pk=pk)
    video_id = parse_youtube_url(p.youtube_url)
    if not video_id:
        raise ValueError('Некорректная YouTube-ссылка.')
    # Same video already on disk in another project -> no second download.
    for other in Project.objects.filter(youtube_url=p.youtube_url).exclude(pk=pk).exclude(source=''):
        try:
            if other.source and _P(other.source.path).is_file():
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
            raise RuntimeError('Видео превышает лимит 1 ГБ.')
        with src.open('rb') as stream:
            p.source.save(video_id + src.suffix, File(stream), save=True)


def retrieve_and_process(pk):
    """v4-compatible entry point (the job runner now does fetch + process itself)."""
    from .pipeline import process_project
    process_project(pk)
