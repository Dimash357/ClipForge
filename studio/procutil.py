"""Subprocess helpers: process-tree kill, ffmpeg with real progress and cancellation."""
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time

log = logging.getLogger('studio.proc')


class JobCancelled(Exception):
    """Raised inside a worker when the user asked to stop the job."""


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, check=False)
        else:
            proc.kill()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _flags():
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0


def run_ffmpeg(args, duration=None, on_progress=None, cancel_check=None, timeout=1800, cwd=None):
    """Run ffmpeg with `-progress pipe:1`; report real fraction (0..1) of `duration`.

    args: complete command incl. the 'ffmpeg' executable. Raises RuntimeError with the stderr
    tail on failure, JobCancelled if cancel_check() turns true, TimeoutError on timeout.
    """
    cmd = list(args)
    cmd[1:1] = ['-nostats', '-progress', 'pipe:1']
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, cwd=cwd, creationflags=_flags(),
                                text=True, encoding='utf-8', errors='replace')
        q = queue.Queue()

        def pump():
            try:
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        started = time.monotonic()
        done = False
        try:
            while not done:
                if cancel_check and cancel_check():
                    raise JobCancelled()
                if time.monotonic() - started > timeout:
                    raise TimeoutError('ffmpeg превысил лимит времени')
                try:
                    line = q.get(timeout=0.4)
                except queue.Empty:
                    continue
                if line is None:
                    done = True
                    continue
                if on_progress and duration and line.startswith('out_time_us='):
                    try:
                        sec = int(line.split('=')[1]) / 1e6
                        on_progress(max(0.0, min(1.0, sec / duration)))
                    except ValueError:
                        pass
            proc.wait(timeout=30)
        except BaseException:
            kill_tree(proc)
            raise
        if proc.returncode:
            err.seek(0)
            tail = err.read()[-1700:].decode('utf-8', errors='replace')
            raise RuntimeError(tail or f'ffmpeg завершился с кодом {proc.returncode}')


def run_capture(args, timeout=60, cwd=None):
    """Short helper: run, capture text, raise on non-zero."""
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                       errors='replace', timeout=timeout, cwd=cwd)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1700:])
    return p.stdout


def media_streams(path, timeout=60):
    """[(codec_type, codec_name)] of a media file via ffprobe; [] if it cannot be read."""
    import json
    try:
        out = run_capture(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name', '-of', 'json',
                           str(path)], timeout=timeout)
        return [(st.get('codec_type'), st.get('codec_name')) for st in json.loads(out).get('streams', [])]
    except Exception:
        return []


def has_video(path):
    return any(t == 'video' and c not in ('mjpeg', 'png') for t, c in media_streams(path))


def describe_streams(path):
    st = media_streams(path)
    if not st:
        return 'ffprobe не смог прочитать файл'
    return ', '.join(f'{t}:{c}' for t, c in st)
