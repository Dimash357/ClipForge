"""In-process job runner for ClipForge.

* one shared queue, `settings.MAX_CONCURRENT_JOBS` worker threads (default 1) -> heavy work never overlaps
* workers are started lazily in the serving process only (no double workers on Django autoreload)
* a project can have only one queued/running job
* cooperative cancellation via threading.Event (+ DB flag for the UI)
* projects left in an active state by a dead process are marked `interrupted`, never stuck forever
"""
import logging
import queue
import threading
import time

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .procutil import JobCancelled

log = logging.getLogger('studio.jobs')

ACTIVE_STATUSES = ('queued', 'fetching', 'processing')

# (key, label, weight) -- weights only feed the legacy `progress` number; the UI shows stage-level truth.
STAGES = [
    ('fetch', 'Получение видео', 12),
    ('audio', 'Анализ аудио', 3),
    ('transcribe', 'Транскрибация', 30),
    ('highlights', 'Поиск интересных моментов', 5),
    ('scenes', 'Анализ сцен', 10),
    ('edit', 'Монтаж', 3),
    ('render', 'Рендеринг', 32),
    ('validate', 'Проверка результата', 5),
]
STAGE_KEYS = [s[0] for s in STAGES]

_q = queue.Queue()
_lock = threading.Lock()
_active = {}      # pk -> {'kind':..., 'running': bool}
_cancel = {}      # pk -> threading.Event
_workers = []
_recovered = False


# ---------------------------------------------------------------- DB helpers
def set_state(pk, **fields):
    from .models import Project
    Project.objects.filter(pk=pk).update(updated_at=timezone.now(), **fields)


class Reporter:
    """Progress/cancellation handle passed into pipeline functions."""

    def __init__(self, pk, skipped=()):
        self.pk = pk
        self.skipped = set(skipped)
        self.current = None
        self._last_write = 0.0
        self._stage_frac = None
        self._first_frac = True

    # --- cancellation
    def cancelled(self):
        ev = _cancel.get(self.pk)
        return bool(ev and ev.is_set())

    def check(self):
        if self.cancelled():
            raise JobCancelled()

    # --- progress
    def skip(self, *keys):
        self.skipped.update(keys)

    def _overall(self):
        keys = [k for k in STAGE_KEYS if k not in self.skipped]
        total = sum(w for k, _, w in STAGES if k in keys) or 1
        done = 0.0
        for k, _, w in STAGES:
            if k not in keys:
                continue
            if k == self.current:
                done += w * (self._stage_frac or 0.0)
                break
            done += w
        return int(max(0, min(99, round(100 * done / total))))

    def stage(self, key, message, status='processing', frac=None):
        self.check()
        self.current = key
        self._stage_frac = frac
        self._first_frac = True
        self._write(key, message, status, force=True)

    def frac(self, value, message=None):
        """Report a *real* fraction of the current stage (0..1)."""
        self._stage_frac = max(0.0, min(1.0, float(value)))
        now = time.monotonic()
        if self._first_frac or now - self._last_write >= 0.6 or self._stage_frac >= 1.0:
            self._first_frac = False
            self._write(self.current, message, None)

    def _write(self, key, message, status, force=False):
        self._last_write = time.monotonic()
        fields = {'stage': key or '', 'stage_progress': self._stage_frac, 'progress': self._overall()}
        if message is not None:
            fields['message'] = message
        if status:
            fields['status'] = status
        set_state(self.pk, **fields)


# ---------------------------------------------------------------- queue API
def is_active(pk):
    return pk in _active


def enqueue(kind, pk, **payload):
    """Queue a job. Returns False if this project already has a queued/running job."""
    with _lock:
        if pk in _active:
            return False
        _active[pk] = {'kind': kind, 'running': False}
        _cancel[pk] = threading.Event()
    set_state(pk, status='queued', cancel_requested=False, error='', message='В очереди...', stage='queued',
              stage_progress=None)
    _q.put((kind, pk, payload))
    _ensure_workers()
    return True


def request_cancel(pk):
    """Ask a queued/running job to stop. Returns True if there was something to cancel."""
    with _lock:
        info = _active.get(pk)
        ev = _cancel.get(pk)
        if not info:
            return False
        if ev:
            ev.set()
        queued_only = not info['running']
    set_state(pk, cancel_requested=True, message='Останавливаем обработку...')
    if queued_only:
        # not started yet: the worker will skip it; reflect immediately in the UI
        set_state(pk, status='cancelled', message='Отменено до начала обработки', stage='')
    return True


def reap(pk):
    """If the DB says 'active' but no live job exists in this process, mark it interrupted."""
    from .models import Project
    p = Project.objects.filter(pk=pk).only('status').first()
    if p and p.status in ACTIVE_STATUSES and pk not in _active:
        mark_interrupted(pk)
        return True
    return False


def mark_interrupted(pk):
    from .models import Clip
    set_state(pk, status='interrupted', stage='', stage_progress=None,
              message='Обработка была прервана (перезапуск сервера или сбой). Нажмите «Продолжить» — '
                      'скачанное видео и транскрипт будут использованы повторно.')
    for c in Clip.objects.filter(project_id=pk, status__in=('queued', 'rendering')):
        c.status = 'ready' if c.video else 'candidate'
        c.save(update_fields=['status', 'updated_at'])


def recover_once():
    """At process start: nothing can be running yet, so any 'active' project is orphaned."""
    global _recovered
    if _recovered:
        return 0
    _recovered = True
    try:
        from .models import Project
        pks = list(Project.objects.filter(status__in=ACTIVE_STATUSES).values_list('pk', flat=True))
        for pk in pks:
            mark_interrupted(pk)
        if pks:
            log.warning('Recovered %d orphaned project(s): %s', len(pks), pks)
        return len(pks)
    except Exception as exc:  # table may not exist yet (before migrate)
        log.info('recover_once skipped: %s', exc)
        return 0


def _ensure_workers():
    with _lock:
        _workers[:] = [t for t in _workers if t.is_alive()]
        while len(_workers) < settings.MAX_CONCURRENT_JOBS:
            t = threading.Thread(target=_worker_loop, name=f'clipforge-worker-{len(_workers)}', daemon=True)
            _workers.append(t)
            t.start()


def _worker_loop():
    while True:
        kind, pk, payload = _q.get()
        try:
            ev = _cancel.get(pk)
            if ev and ev.is_set():
                set_state(pk, status='cancelled', message='Отменено', stage='')
                continue
            with _lock:
                if pk in _active:
                    _active[pk]['running'] = True
            close_old_connections()
            _run_job(kind, pk, payload)
        except Exception:  # never let the worker die
            log.exception('worker crashed on job %s/%s', kind, pk)
        finally:
            with _lock:
                _active.pop(pk, None)
                _cancel.pop(pk, None)
            close_old_connections()


def _run_job(kind, pk, payload):
    from . import pipeline
    rep = Reporter(pk)
    try:
        pipeline.run_job(kind, pk, payload, rep)
    except JobCancelled:
        log.info('job %s/%s cancelled', kind, pk)
        pipeline.after_cancel(pk)
        set_state(pk, status='cancelled', stage='', stage_progress=None, message='Обработка остановлена. '
                  'Уже готовые клипы сохранены; можно продолжить.')
    except Exception as exc:
        log.exception('job %s/%s failed', kind, pk)
        set_state(pk, status='error', stage_progress=None, error=str(exc)[-2000:],
                  message='Ошибка обработки')
