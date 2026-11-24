"""Persistent job queue for ClipForge (Django + SQLite, no external services).

* Jobs live in the `Job` table: they survive closing the app, a Windows restart or a crash.
* Worker threads (started lazily, only in the serving process) claim jobs with one atomic SQL UPDATE, so a job can
  never run twice - not even with two ClipForge processes - and at most `MAX_CONCURRENT_JOBS` (default 1) *heavy*
  jobs run at once (across processes; the full GPU scheduler comes later).
* A project has at most one live job (queued / running / paused): partial unique index in the database.
* Running jobs write a heartbeat. A job whose heartbeat stops (crash, power loss) is marked `interrupted` and - up to
  `max_attempts` starts - queued again automatically; manual retries reset the counter, so there is no retry loop.
* Checkpoints (`Reporter.checkpoint`) record finished steps in the job; a resumed job skips them.
* Pause is honoured only at safe points (stage boundaries and checkpoints); cancel stops as soon as possible
  (ffmpeg is killed).
* New kinds of work register a handler: `register('cartoon', fn, label='Мультфильм')`; fn(job, reporter).
"""
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

from django.conf import settings
from django.db import IntegrityError, OperationalError, close_old_connections, connection, transaction
from django.utils import timezone

from .procutil import JobCancelled

log = logging.getLogger('studio.jobs')

ACTIVE_STATUSES = ('queued', 'fetching', 'processing')       # Project.status values of a project being worked on

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

HEARTBEAT_SECONDS = 5
IDLE_POLL_SECONDS = 1.0

_lock = threading.Lock()
_wake = threading.Event()
_active = {}      # key (project pk or 'job<id>') -> {'kind', 'running', 'job'}: jobs running in THIS process
_cancel = {}      # key -> threading.Event (cancel)
_pause = {}       # key -> threading.Event (pause at the next safe point)
_workers = []
_recovered = False
_last_sweep = [0.0]
HOST = socket.gethostname()[:20].replace(':', '-')
PROCESS = f'{HOST}:{os.getpid()}:{uuid.uuid4().hex[:6]}'


def _pid_alive(pid):
    """Is a process with this pid running on this computer? (never sends a signal on Windows)"""
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259    # STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner_dead(worker):
    """True when the job's owner was a process on this computer that no longer runs (crash, closed window)."""
    try:
        host, pid, run = worker.split(':')[:3]
        pid = int(pid)
    except (ValueError, AttributeError):
        return False
    if host != HOST:
        return False                                     # another computer: only the heartbeat can tell
    if pid == os.getpid():
        return not worker.startswith(PROCESS + ':')      # same pid, other run token -> an earlier, dead process
    return not _pid_alive(pid)


class JobPaused(JobCancelled):
    """Raised at a safe point when a pause was requested. A JobCancelled subclass, so every existing
    `except JobCancelled: raise` path also lets a pause through (and cleans up the same way)."""


def stale_seconds():
    return int(getattr(settings, 'JOB_STALE_SECONDS', 30))


def max_heavy():
    """Heavy jobs at once: CLIPFORGE_MAX_JOBS if set, else the resource mode (balanced/low 1, performance 2)."""
    if os.getenv('CLIPFORGE_MAX_JOBS') or getattr(settings, 'MAX_CONCURRENT_JOBS_FIXED', False):
        return max(1, int(getattr(settings, 'MAX_CONCURRENT_JOBS', 1)))
    from . import resources
    return max(1, int(resources.manager().profile()['heavy_jobs']))


# ---------------------------------------------------------------- handlers (extensible)
@dataclass
class Handler:
    kind: str
    fn: object                    # fn(job, reporter)
    label: str
    resource: str = 'heavy'       # 'heavy' counts against MAX_CONCURRENT_JOBS; 'light' does not
    needs_project: bool = True


_handlers = {}
_builtin = []


def register(kind, fn, *, label, resource='heavy', needs_project=True):
    _handlers[kind] = Handler(kind, fn, label, resource, needs_project)


def handler(kind):
    if not _builtin:
        _builtin.append(True)
        _register_builtin()
    if kind not in _handlers:
        raise ValueError(f'unknown job kind {kind!r}')
    return _handlers[kind]


def _pipeline_job(job, rep):
    from . import pipeline
    pipeline.run_job(job.kind, job.project_id, job.payload or {}, rep)


def _timeline_job(job, rep):
    """Render a Timeline IR document (payload: timeline, optional base_dir) into MEDIA_ROOT/renders/timeline/."""
    from pathlib import Path
    from .timeline import from_dict, renderer
    rep.stage('render', 'Рендер таймлайна...', frac=0.0)
    out = Path(settings.MEDIA_ROOT) / 'renders' / 'timeline' / f'job_{job.pk}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    tl = from_dict(job.payload['timeline'])
    r = renderer.render(tl, out, renderer.RenderSettings(base_dir=job.payload.get('base_dir')),
                        on_progress=rep.frac, cancel=rep.cancelled)
    rep.checkpoint('output', path=str(out.relative_to(settings.MEDIA_ROOT)), seconds=r['seconds'])


def _script_job(job, rep):
    """Generate a whole script or regenerate one scene (payload: script_id, action, scene_id, instruction)."""
    from . import scripts
    from .models import Script
    sc = Script.objects.get(pk=job.payload['script_id'])
    try:
        if job.payload.get('action') == 'storyboard':
            from . import storyboard
            rep.stage('scenes', 'Раскадровка: библии стиля и персонажей, кадры по сценам...')
            only = job.payload.get('scenes')
            doc = storyboard.build(sc.data, cancel=rep.cancelled, only=set(only) if only else None)
        elif job.payload.get('action') == 'frame':
            from . import storyboard
            rep.stage('scenes', 'Перерисовываем кадр раскадровки...')
            doc = storyboard.regenerate_frame(sc.data, job.payload['frame_id'], job.payload.get('instruction', ''),
                                              cancel=rep.cancelled)
        elif job.payload.get('action') == 'scene':
            rep.stage('edit', 'Переписываем сцену...')
            doc = scripts.regenerate_scene(sc.data, job.payload['scene_id'], job.payload.get('instruction', ''),
                                           cancel=rep.cancelled)
        else:
            rep.stage('edit', 'Пишем сценарий (локальная модель может думать минуту-две)...')
            doc = scripts.generate(scripts.settings_of(sc), cancel=rep.cancelled)
        Script.objects.filter(pk=sc.pk).update(data=doc, status='ready', error='',
                                               title=sc.title or doc['title'][:250])
        rep.checkpoint('script', provider=doc.get('provider_last') or doc.get('provider'))
    except JobCancelled:
        Script.objects.filter(pk=sc.pk).update(status='ready' if sc.data else 'draft')
        raise
    except Exception as exc:
        Script.objects.filter(pk=sc.pk).update(status='error', error=str(exc)[:1000])
        raise


def _media_job(job, rep):
    """3C: images (ComfyUI) and / or voices (TTS) for a script's storyboard. Every finished item is saved at once,
    so a cancelled or interrupted job continues where it stopped. payload: script_id, what, ids, force."""
    from . import genmedia
    from .models import Script
    sc = Script.objects.get(pk=job.payload['script_id'])
    doc = sc.data
    what, ids, force = job.payload.get('what', 'all'), job.payload.get('ids'), bool(job.payload.get('force'))
    cfg = genmedia.config()
    report = {}

    def saver(stage):
        def save(frac, msg):
            Script.objects.filter(pk=sc.pk).update(data=doc)
            rep.frac(frac, f'{stage}: {msg}')
        return save
    try:
        if what in ('all', 'images', 'frames', 'characters'):
            rep.stage('render', 'Изображения: ComfyUI...', frac=0.0)
            report['images'] = genmedia.gen_images(
                doc, sc.pk, cfg, frame_ids=set(ids) if what == 'frames' and ids else None,
                characters=set(ids) if what == 'characters' and ids else None, force=force, cancel=rep.cancelled,
                progress=saver('Изображения'))
            rep.checkpoint('images', **{k: v for k, v in report['images'].items() if k != 'errors'})
        if what in ('all', 'voice', 'lines'):
            rep.stage('transcribe', 'Озвучка и точные тайминги...', frac=0.0)
            report['voice'] = genmedia.gen_voices(
                doc, sc.pk, cfg, lines=[list(x) for x in ids] if what == 'lines' and ids else None,
                scene_ids=None, force=force, cancel=rep.cancelled, progress=saver('Озвучка'))
            rep.checkpoint('voice', **{k: v for k, v in report['voice'].items() if k != 'errors'})
        errs = [e for r in report.values() for e in r.get('errors', [])]
        doc['storyboard']['media_report'] = dict(report, at=int(time.time()))
        Script.objects.filter(pk=sc.pk).update(data=doc, status='ready', error=(
            f'{len(errs)} элемент(ов) не удалось: ' + '; '.join(e['error'] for e in errs[:3])) if errs else '')
    except JobCancelled:
        Script.objects.filter(pk=sc.pk).update(data=doc, status='ready')
        raise
    except Exception as exc:
        Script.objects.filter(pk=sc.pk).update(data=doc, status='error', error=str(exc)[:1000])
        raise


def _production_job(job, rep):
    """3D Cartoon Studio: idea -> MP4. payload: script_id, only (stage keys), until, force (stage keys)."""
    from . import production
    from .models import Script
    sc = Script.objects.get(pk=job.payload['script_id'])
    Script.objects.filter(pk=sc.pk).update(status='generating', error='')
    try:
        production.run(sc, rep, only=job.payload.get('only'), until=job.payload.get('until'),
                       force=set(job.payload.get('force') or ()))
    except JobCancelled:
        Script.objects.filter(pk=sc.pk).update(status='ready' if (sc.data or {}).get('scenes') else 'draft')
        raise


def _register_builtin():
    for kind, label in (('full', 'Stream Clipper: обработка'), ('resume', 'Stream Clipper: продолжение'),
                        ('reprocess', 'Stream Clipper: заново'), ('render', 'Stream Clipper: рендер клипов')):
        register(kind, _pipeline_job, label=label)
    register('timeline_render', _timeline_job, label='Рендер Timeline IR', needs_project=False)
    register('script', _script_job, label='Сценарий (AI Script Generator)', resource='light', needs_project=False)
    register('media', _media_job, label='Изображения и озвучка (3C)', resource='heavy', needs_project=False)
    register('production', _production_job, label='Cartoon Studio: видео из идеи (3D)', resource='heavy',
             needs_project=False)


# ---------------------------------------------------------------- DB helpers (SQLite: retry on "locked")
def _retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except OperationalError as exc:
            if 'locked' not in str(exc).lower() or i == tries - 1:
                raise
            time.sleep(0.1 * 2 ** i)


def set_state(pk, **fields):
    from .models import Project
    _retry(lambda: Project.objects.filter(pk=pk).update(updated_at=timezone.now(), **fields))


def job_update(job_id, only_worker=None, **fields):
    from .models import Job
    q = Job.objects.filter(pk=job_id)
    if only_worker is not None:
        q = q.filter(worker=only_worker)
    return _retry(lambda: q.update(updated_at=timezone.now(), **fields))


def _key(job):
    return job.project_id if job.project_id else f'job{job.pk}'


# ---------------------------------------------------------------- progress / cancel / pause / checkpoints
class Reporter:
    """Progress, cancellation, pause and checkpoint handle passed into pipeline functions."""

    def __init__(self, pk, skipped=(), job=None):
        self.pk = pk
        self.job_id = job.pk if job is not None else None
        self.key = _key(job) if job is not None else pk
        self.checkpoints = dict(job.checkpoints or {}) if job is not None else {}
        self.skipped = set(skipped)
        self.current = None
        self._last_write = 0.0
        self._stage_frac = None
        self._first_frac = True
        self._message = ''
        self._stage_started = None
        self._ops = {}                          # parallel operations: key -> {'label', 'frac', 'started', ...}
        self._ops_lock = threading.Lock()
        self.last_progress = time.monotonic()
        self.timings = dict((job.activity or {}).get('timings') or {}) if job is not None else {}

    # --- cancellation / pause
    def cancelled(self):
        ev = _cancel.get(self.key)
        return bool(ev and ev.is_set())

    def check(self):
        if self.cancelled():
            raise JobCancelled()

    def safe_point(self):
        """A place where stopping loses nothing: honour a pause request here."""
        self.check()
        ev = _pause.get(self.key)
        if ev and ev.is_set():
            raise JobPaused()

    # --- checkpoints
    def done(self, name):
        return name in self.checkpoints

    def get(self, name, default=None):
        return self.checkpoints.get(name, default)

    def checkpoint(self, step, **data):
        """Record a finished step (persisted in the job), then allow a pause."""
        self.checkpoints[step] = dict(data, at=timezone.now().isoformat(timespec='seconds'))
        if self.job_id:
            job_update(self.job_id, checkpoints=self.checkpoints)
        self.safe_point()

    def checkpoint_add(self, step, value):
        """Append one id to a list checkpoint (e.g. every rendered clip)."""
        ids = list((self.checkpoints.get(step) or {}).get('ids', []))
        if value not in ids:
            ids.append(value)
        self.checkpoint(step, ids=ids)

    # --- progress
    def skip(self, *keys):
        self.skipped.update(keys)

    def _overall(self):
        if getattr(self, 'overall_fn', None) is not None:      # jobs with their own stage list (3D production)
            return self.overall_fn()
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

    # --- live operations (shown with elapsed / remaining time)
    def op(self, key, label=None, frac=None, done=False, **info):
        """Report a side operation (e.g. the screen scan running next to Whisper). Thread-safe."""
        now = time.time()
        with self._ops_lock:
            if done:
                o = self._ops.pop(key, None)
                if o:
                    self.timings[key] = round(now - o['started'], 1)
            else:
                o = self._ops.setdefault(key, {'label': label or key, 'frac': None, 'started': now})
                if label:
                    o['label'] = label
                if frac is not None:
                    o['frac'] = max(0.0, min(1.0, float(frac)))
                o['updated'] = now
                o.update(info)
        self.last_progress = time.monotonic()

    @staticmethod
    def _eta(o, now):
        el = now - o['started']
        f = o.get('frac')
        eta = el * (1 - f) / f if f and f >= 0.02 and el >= 3 and f < 1 else None
        return dict(o, elapsed=round(el, 1), eta=None if eta is None else round(eta, 1))

    def activity(self):
        now = time.time()
        ops = []
        if self.current:
            ops.append(self._eta({'key': 'stage', 'label': self._message or self.current, 'frac': self._stage_frac,
                                  'started': self._stage_started or now, 'stage': self.current}, now))
        with self._ops_lock:
            ops += [self._eta(dict(o, key=k), now) for k, o in self._ops.items()]
        idle = time.monotonic() - self.last_progress
        return {'ops': ops, 'timings': self.timings, 'at': now,
                'stalled_for': round(idle) if idle > float(getattr(settings, 'JOB_STALL_WARN_SECONDS', 600)) else None}

    def stage(self, key, message, status='processing', frac=None):
        self.safe_point()
        if self.current and self._stage_started and key != self.current:
            took = round(time.time() - self._stage_started, 1)
            self.timings[self.current] = round(self.timings.get(self.current, 0) + took, 1)
            log.info('stage %s took %.1f s', self.current, took)
        if key != self.current or self._stage_started is None:
            self._stage_started = time.time()
        self.current = key
        self._stage_frac = frac
        self._first_frac = True
        self._write(key, message, status, force=True)

    def note(self, message):
        """Update only the message (stage progress stays as it is, e.g. unknown)."""
        self._write(self.current, message, None)

    def frac(self, value, message=None):
        """Report a *real* fraction of the current stage (0..1). A new message is always shown at once."""
        self._stage_frac = max(0.0, min(1.0, float(value)))
        now = time.monotonic()
        self.last_progress = now
        changed = message is not None and message.split(':')[0] != (self._message or '').split(':')[0]
        if self._first_frac or changed or now - self._last_write >= 0.6 or self._stage_frac >= 1.0:
            self._first_frac = False
            self._write(self.current, message, None)

    def _write(self, key, message, status, force=False):
        self._last_write = time.monotonic()
        self.last_progress = self._last_write
        fields = {'stage': key or '', 'stage_progress': self._stage_frac, 'progress': self._overall()}
        if message is not None:
            fields['message'] = message
            self._message = message
        if self.pk:
            set_state(self.pk, **fields, **({'status': status} if status else {}))
        if self.job_id:
            job_update(self.job_id, heartbeat_at=timezone.now(), activity=self.activity(), **fields)


# ---------------------------------------------------------------- queue API
def _live():
    from .models import Job
    return Job.objects.filter(status__in=Job.LIVE)


def live_job(pk):
    return _live().filter(project_id=pk).first()


def is_active(pk):
    """True while the project has a queued, running or paused job."""
    return _live().filter(project_id=pk).exists()


def submit(kind, pk=None, *, priority=0, title='', **payload):
    """Queue a job. -> Job, or None if this project already has a live job."""
    from .models import Job, Project
    h = handler(kind)
    if h.needs_project and not pk:
        raise ValueError(f'job kind {kind!r} needs a project')
    if not title and pk:
        p = Project.objects.filter(pk=pk).only('title').first()
        title = (p and p.title) or f'Проект #{pk}'

    def create():
        with transaction.atomic():
            if pk and Job.objects.filter(project_id=pk, status__in=Job.LIVE).exists():
                return None
            return Job.objects.create(kind=kind, project_id=pk, payload=payload, priority=int(priority),
                                      resource=h.resource, title=title[:250], message='В очереди...',
                                      max_attempts=int(getattr(settings, 'JOB_MAX_ATTEMPTS', 3)))
    try:
        job = _retry(create)
    except IntegrityError:               # another process queued the same project at the same moment
        return None
    if job is None:
        return None
    if pk:
        set_state(pk, status='queued', cancel_requested=False, error='', message='В очереди...', stage='queued',
                  stage_progress=None)
    wake()
    return job


def enqueue(kind, pk, **payload):
    """Compatibility wrapper: queue a project job. Returns False if the project already has a live job."""
    return submit(kind, pk, **payload) is not None


def wake():
    _ensure_workers()
    _wake.set()


def request_cancel(pk):
    """Cancel the project's live job. Returns True if there was something to cancel."""
    job = live_job(pk)
    return bool(job and cancel_job(job.pk))


def release_script(script_id):
    """A script left 'generating' by a job that never ran (cancelled in the queue, lost) -> ready / draft again.
    -> True if it was released."""
    from .models import Job, Script
    if Job.objects.filter(payload__script_id=script_id, status__in=Job.LIVE).exists():
        return False
    sc = Script.objects.filter(pk=script_id, status='generating').first()
    if sc is None:
        return False
    Script.objects.filter(pk=sc.pk, status='generating').update(
        status='ready' if (sc.data or {}).get('scenes') else 'draft')
    return True


def cancel_job(job_id):
    from .models import Job
    job = Job.objects.filter(pk=job_id).first()
    if not job or job.status not in Job.LIVE:
        return False
    if job.status in (Job.QUEUED, Job.PAUSED):
        n = job_update(job_id, status=Job.CANCELLED, finished_at=timezone.now(), message='Отменено',
                       cancel_requested=True, pause_requested=False)
        if not n:
            return False
        if (job.payload or {}).get('script_id'):
            release_script((job.payload or {})['script_id'])
        if job.project_id:
            from .pipeline import after_cancel
            after_cancel(job.project_id)
            set_state(job.project_id, status='cancelled', cancel_requested=True, stage='', stage_progress=None,
                      message='Отменено до начала обработки' if job.status == Job.QUEUED else
                      'Обработка отменена. Уже готовые клипы сохранены.')
        return True
    job_update(job_id, cancel_requested=True, message='Останавливаем обработку...')
    if job.project_id:
        set_state(job.project_id, cancel_requested=True, message='Останавливаем обработку...')
    ev = _cancel.get(_key(job))
    if ev:
        ev.set()
    return True


def pause_job(job_id):
    """Queued -> paused at once; running -> paused at its next safe point."""
    from .models import Job
    job = Job.objects.filter(pk=job_id).first()
    if not job:
        return False
    if job.status == Job.QUEUED:
        ok = job_update(job_id, status=Job.PAUSED, message='Пауза (ожидает продолжения)')
        if ok and job.project_id:
            set_state(job.project_id, status='paused', stage='', stage_progress=None,
                      message='Пауза. Продолжите на странице «Очередь задач».')
        return bool(ok)
    if job.status == Job.RUNNING:
        job_update(job_id, pause_requested=True, message='Пауза после текущего шага...')
        ev = _pause.get(_key(job))
        if ev:
            ev.set()
        return True
    return False


def resume_job(job_id, manual=True):
    """Paused / interrupted / failed / cancelled -> queued again with its checkpoints (finished steps are skipped).
    A manual resume resets the attempt counter."""
    from .models import Job
    job = Job.objects.filter(pk=job_id).first()
    if not job or job.status not in (Job.PAUSED, Job.INTERRUPTED, Job.FAILED, Job.CANCELLED):
        return False
    fields = dict(status=Job.QUEUED, cancel_requested=False, pause_requested=False, error='', finished_at=None,
                  message='В очереди (продолжение)...')
    if manual:
        fields['attempts'] = 0
    try:
        n = _retry(lambda: Job.objects.filter(pk=job_id, status=job.status).update(updated_at=timezone.now(), **fields))
    except IntegrityError:               # the project got another live job meanwhile
        return False
    if n and job.project_id:
        set_state(job.project_id, status='queued', cancel_requested=False, error='', stage='queued',
                  stage_progress=None, message='В очереди (продолжение)...')
    if n:
        wake()
    return bool(n)


def retry_job(job_id):
    """Run again: unfinished jobs continue from their checkpoints; a completed job is queued anew (fresh run)."""
    from .models import Job
    job = Job.objects.filter(pk=job_id).first()
    if not job:
        return None
    if job.status == Job.COMPLETED:
        return submit(job.kind, job.project_id, priority=job.priority, title=job.title, **(job.payload or {}))
    return job if resume_job(job_id) else None


def set_priority(job_id, priority):
    from .models import Job
    return bool(job_update(job_id, priority=max(-100, min(100, int(priority))))) if \
        Job.objects.filter(pk=job_id).exists() else False


# ---------------------------------------------------------------- crash recovery
def mark_interrupted(pk):
    from .models import Clip
    set_state(pk, status='interrupted', stage='', stage_progress=None,
              message='Обработка была прервана (перезапуск сервера или сбой). Нажмите «Продолжить» — '
                      'скачанное видео и транскрипт будут использованы повторно.')
    for c in Clip.objects.filter(project_id=pk, status__in=('queued', 'rendering')):
        c.status = 'ready' if c.video else 'candidate'
        c.save(update_fields=['status', 'updated_at'])


def sweep_stale(force=False):
    """Running jobs without a fresh heartbeat belong to a dead process: mark them interrupted and, while attempts
    remain, queue them again (they continue from their checkpoints). -> number of interrupted jobs."""
    from .models import Job
    now = time.monotonic()
    if not force and now - _last_sweep[0] < 10:
        return 0
    _last_sweep[0] = now
    cutoff = timezone.now() - timedelta(seconds=stale_seconds())
    running = Job.objects.filter(status=Job.RUNNING).exclude(worker__startswith=PROCESS + ':')
    stale = [j for j in running if j.heartbeat_at is None or j.heartbeat_at < cutoff or owner_dead(j.worker)]
    auto = bool(getattr(settings, 'JOB_AUTO_RESUME', True))
    count = 0
    for job in stale:
        n = _retry(lambda: Job.objects.filter(pk=job.pk, status=Job.RUNNING, worker=job.worker).update(
            status=Job.INTERRUPTED, worker='', finished_at=timezone.now(), updated_at=timezone.now(),
            message='Прервано: приложение закрылось или упало во время обработки'))
        if not n:
            continue
        count += 1
        if job.project_id:
            mark_interrupted(job.project_id)
        if auto and job.attempts < job.max_attempts and not job.cancel_requested:
            resume_job(job.pk, manual=False)
            log.warning('job %s (%s) was interrupted; queued again (attempt %d of %d)', job.pk, job.kind,
                        job.attempts + 1, job.max_attempts)
        else:
            job_update(job.pk, message=f'Прервано {job.attempts} раз(а) подряд — автоматический перезапуск '
                                       f'остановлен. Нажмите «Повторить».' if job.attempts >= job.max_attempts
                       else 'Прервано. Нажмите «Продолжить».')
            log.warning('job %s (%s) was interrupted and is NOT restarted automatically', job.pk, job.kind)
    return count


def reap(pk):
    """If the DB says 'active' but no live job exists, mark the project interrupted."""
    from .models import Project
    sweep_stale()
    p = Project.objects.filter(pk=pk).only('status').first()
    if p and p.status in ACTIVE_STATUSES and not is_active(pk):
        mark_interrupted(pk)
        return True
    return False


def recover_once():
    """At process start: recover jobs of dead processes, mark orphaned projects, start workers for waiting jobs."""
    global _recovered
    if _recovered:
        return 0
    _recovered = True
    try:
        from .models import Job, Project
        jobs_n = sweep_stale(force=True)
        live = set(_live().exclude(project=None).values_list('project_id', flat=True))
        pks = [pk for pk in Project.objects.filter(status__in=ACTIVE_STATUSES).values_list('pk', flat=True)
               if pk not in live]
        for pk in pks:
            mark_interrupted(pk)
        if pks or jobs_n:
            log.warning('Recovered %d interrupted job(s) and %d orphaned project(s): %s', jobs_n, len(pks), pks)
        wake()                       # workers also keep checking for jobs of crashed processes
        return len(pks) + jobs_n
    except Exception as exc:  # table may not exist yet (before migrate)
        log.info('recover_once skipped: %s', exc)
        return 0


# ---------------------------------------------------------------- workers
def _ensure_workers():
    with _lock:
        _workers[:] = [t for t in _workers if t.is_alive()]
        while len(_workers) < max_heavy():
            t = threading.Thread(target=_worker_loop, name=f'clipforge-worker-{len(_workers)}', daemon=True)
            _workers.append(t)
            t.start()


def _claim(token):
    """Atomically take the best queued job (priority, then age). Heavy jobs only while fewer than
    MAX_CONCURRENT_JOBS heavy jobs run - checked in the same UPDATE, so it holds across processes."""
    from .models import Job
    table = Job._meta.db_table
    now = connection.ops.adapt_datetimefield_value(timezone.now())
    for pk, res in Job.objects.filter(status=Job.QUEUED).order_by('-priority', 'created_at', 'pk') \
            .values_list('pk', 'resource')[:20]:
        sql = (f"UPDATE {table} SET status='running', worker=%s, heartbeat_at=%s, started_at=%s, updated_at=%s, "
               f"attempts=attempts+1, pause_requested=0, finished_at=NULL WHERE id=%s AND status='queued'")
        params = [token, now, now, now, pk]
        if res == 'heavy':
            sql += f" AND (SELECT COUNT(*) FROM {table} WHERE status='running' AND resource='heavy') < %s"
            params.append(max_heavy())

        def run():
            with connection.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount
        if _retry(run):
            return Job.objects.get(pk=pk)
        if res == 'heavy':
            continue
    return None


def _worker_loop():
    token = f'{PROCESS}:{threading.current_thread().name[-2:]}'
    while True:
        try:
            close_old_connections()
            sweep_stale()
            job = _claim(token)
            if job is None:
                _wake.wait(IDLE_POLL_SECONDS)
                _wake.clear()
                continue
            _execute(job, token)
        except Exception:  # never let the worker die
            log.exception('job worker error')
            time.sleep(1)
        finally:
            close_old_connections()


def _beat(job_id, key, token, stop, rep=None):
    """Heartbeat (+ live operations for the UI) and pick up cancel/pause requests made by other processes."""
    from .models import Job
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            extra = {'activity': rep.activity()} if rep is not None else {}
            job_update(job_id, only_worker=token, heartbeat_at=timezone.now(), **extra)
            row = Job.objects.filter(pk=job_id).values('cancel_requested', 'pause_requested').first() or {}
            if row.get('cancel_requested') and key in _cancel:
                _cancel[key].set()
            if row.get('pause_requested') and key in _pause:
                _pause[key].set()
        except Exception as exc:
            log.debug('heartbeat failed: %s', exc)
        finally:
            close_old_connections()


def _execute(job, token):
    from .models import Job
    key = _key(job)
    with _lock:
        _active[key] = {'kind': job.kind, 'running': True, 'job': job.pk}
        _cancel[key] = threading.Event()
        _pause[key] = threading.Event()
    if job.cancel_requested:
        _cancel[key].set()
    stop = threading.Event()
    rep = Reporter(job.project_id, job=job)
    from . import resources
    resources.manager().bind_job(job.pk, job.priority, rep.cancelled)
    beat = threading.Thread(target=_beat, args=(job.pk, key, token, stop, rep), daemon=True, name=f'beat-{job.pk}')
    beat.start()
    t0 = time.monotonic()
    final = {}
    try:
        try:
            h = handler(job.kind)
            if job.project_id:
                set_state(job.project_id, cancel_requested=False)
            h.fn(job, rep)
            final = dict(status=Job.COMPLETED, message='Готово', progress=100, stage='', stage_progress=None)
        except JobPaused:
            log.info('job %s/%s paused', job.kind, job.pk)
            final = dict(status=Job.PAUSED, pause_requested=False, message='Пауза. Готовые шаги сохранены.')
            if job.project_id:
                from .pipeline import after_cancel
                after_cancel(job.project_id)
                set_state(job.project_id, status='paused', stage='', stage_progress=None,
                          message='Пауза. Готовые шаги сохранены — продолжите на странице «Очередь задач».')
        except JobCancelled:
            log.info('job %s/%s cancelled', job.kind, job.pk)
            final = dict(status=Job.CANCELLED, message='Отменено')
            if job.project_id:
                from .pipeline import after_cancel
                after_cancel(job.project_id)
                set_state(job.project_id, status='cancelled', stage='', stage_progress=None,
                          message='Обработка остановлена. Уже готовые клипы сохранены; можно продолжить.')
        except Exception as exc:
            log.exception('job %s/%s failed', job.kind, job.pk)
            final = dict(status=Job.FAILED, error=str(exc)[-2000:], message='Ошибка обработки', stage_progress=None)
            if job.project_id:
                set_state(job.project_id, status='error', stage_progress=None, error=str(exc)[-2000:],
                          message='Ошибка обработки')
    finally:
        stop.set()
        resources.manager().unbind_job(job.pk)
        cur = Job.objects.filter(pk=job.pk).values_list('run_seconds', flat=True).first() or 0.0
        if final.get('status') in (Job.COMPLETED, Job.FAILED, Job.CANCELLED):
            final['finished_at'] = timezone.now()
        if rep.current and rep._stage_started:
            rep.timings[rep.current] = round(rep.timings.get(rep.current, 0) + time.time() - rep._stage_started, 1)
        job_update(job.pk, only_worker=token, worker='', run_seconds=cur + (time.monotonic() - t0),
                   activity={'ops': [], 'timings': rep.timings, 'at': time.time()},
                   **(final or {'status': Job.INTERRUPTED, 'message': 'Прервано'}))
        with _lock:
            _active.pop(key, None)
            _cancel.pop(key, None)
            _pause.pop(key, None)
