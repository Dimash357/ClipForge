"""GPU / CPU resource scheduler and AI-model registry (one per ClipForge process).

Every heavy operation asks for what it needs - VRAM, RAM, CPU slots, NVENC / NVDEC sessions - and gets a lease:

    with resources.acquire(vram_mb=1500, cpu=2, label='Whisper') as lease: ...
    lease = resources.try_acquire(nvdec=1, vram_mb=150)        # None if busy -> do it on the CPU

* All-or-nothing grants: a request either gets everything at once or waits holding nothing -> no hold-and-wait.
  A thread that already holds a lease is never put to sleep for another one (it gets Unavailable and falls back
  to the CPU), so leases cannot deadlock. Waiters are served by job priority, then arrival; a request waiting
  longer than AGING_S blocks smaller later ones (no starvation).
* VRAM: the budget is the card's memory minus a mode-dependent reserve, minus what other programs use
  (nvidia-smi), minus our leases and resident models. A grant also needs the *measured* free memory, so a game
  or a browser on the GPU is respected. RAM is checked the same way against the system's available memory.
* Requests larger than the machine (e.g. any VRAM without a GPU) fail at once with Unavailable -> CPU fallback.
* Models are loaded lazily, reused, counted as resident VRAM/RAM and unloaded when idle (LRU), when another
  request needs the memory, after a CUDA out-of-memory error, or after every use in Low Resource mode.
* Modes: balanced (default), performance, low. Saved in clipforge_settings.json; CLIPFORGE_RESOURCE_MODE wins.

Leases are bound to the running job (jobs.py), inherit its priority and cancellation, and are released when the
job ends even if code forgot to. The heavy-job limit across processes stays in the job queue's atomic claim.
"""
import gc
import itertools
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings

log = logging.getLogger('studio.resources')

KINDS = ('cpu', 'ram_mb', 'vram_mb', 'nvenc', 'nvdec')
AGING_S = 20.0

MODES = {
    'balanced': {'label': 'Сбалансированный', 'heavy_jobs': 1, 'vram_reserve_mb': 1024, 'ram_reserve_mb': 3072,
                 'cpu_share': 1.0, 'nvenc': 2, 'nvdec': 2, 'keep_models': True, 'idle_unload_s': 900,
                 'note': 'одна тяжёлая задача, внутри неё параллельно всё совместимое; модели держатся 15 мин'},
    'performance': {'label': 'Максимальная производительность', 'heavy_jobs': 2, 'vram_reserve_mb': 512,
                    'ram_reserve_mb': 2048, 'cpu_share': 1.0, 'nvenc': 3, 'nvdec': 3, 'keep_models': True,
                    'idle_unload_s': 3600,
                    'note': 'две тяжёлые задачи одновременно, меньше запаса памяти, модели держатся час'},
    'low': {'label': 'Экономный', 'heavy_jobs': 1, 'vram_reserve_mb': 2048, 'ram_reserve_mb': 4096, 'cpu_share': 0.5,
            'nvenc': 1, 'nvdec': 1, 'keep_models': False, 'idle_unload_s': 0,
            'note': 'половина ядер, большой запас памяти, модели выгружаются сразу после использования'},
}
OOM_RX = re.compile(r'out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|'
                    r'failed to allocate|std::bad_alloc|MemoryError', re.I)


class Unavailable(RuntimeError):
    """The request cannot be granted (bigger than the machine, busy and not allowed to wait, or timed out)."""


class ResourceTimeout(Unavailable):
    pass


def is_oom(exc):
    return isinstance(exc, MemoryError) or bool(OOM_RX.search(str(exc)))


# ------------------------------------------------------------------ machine facts
def _ram_mb():
    """(total, available) MB."""
    try:
        if os.name == 'nt':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys // 2 ** 20, m.ullAvailPhys // 2 ** 20
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.split()[0]) // 1024
        return info['MemTotal'], info.get('MemAvailable', info.get('MemFree', 0))
    except Exception:
        return None, None


def _gpu():
    from . import sysload
    return sysload.gpu()


# ------------------------------------------------------------------ leases
@dataclass
class Lease:
    id: int
    req: dict
    label: str
    owner: object = None
    priority: int = 0
    granted_at: float = field(default_factory=time.time)
    thread: int = 0
    _mgr: object = None
    released: bool = False

    def release(self):
        if self._mgr is not None and not self.released:
            self._mgr._release(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


@dataclass
class _Ticket:
    seq: int
    req: dict
    priority: int
    since: float
    owner: object = None


@dataclass
class _Model:
    key: tuple
    obj: object
    device: str
    vram_mb: int
    ram_mb: int
    group: str
    unload: object = None
    refs: int = 0
    last_used: float = field(default_factory=time.time)
    loads: int = 1


class ResourceManager:
    def __init__(self):
        self._cv = threading.Condition(threading.RLock())
        self._leases = {}
        self._tickets = []
        self._seq = itertools.count(1)
        self._models = {}
        self._load_locks = {}
        self._local = threading.local()
        self._jobs = {}                       # job id -> {'priority', 'cancel'}
        self._facts = None
        self._facts_at = 0.0
        self._mode = None
        self.events = []                      # recent OOM / fallback events (for the UI)
        self.stats = {'granted': 0, 'waited': 0, 'unavailable': 0, 'timeouts': 0, 'oom': 0, 'evictions': 0}

    # ---------------------------------------------------------------- mode
    def _settings_file(self):
        return Path(getattr(settings, 'CLIPFORGE_SETTINGS_FILE', Path(settings.BASE_DIR) / 'clipforge_settings.json'))

    def mode(self):
        env = os.getenv('CLIPFORGE_RESOURCE_MODE', '').strip().lower()
        if env in MODES:
            return env
        if self._mode is None:
            try:
                self._mode = json.loads(self._settings_file().read_text(encoding='utf-8')).get('resource_mode')
            except Exception:
                self._mode = None
            if self._mode not in MODES:
                self._mode = 'balanced'
        return self._mode

    def mode_locked(self):
        return os.getenv('CLIPFORGE_RESOURCE_MODE', '').strip().lower() in MODES

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}')
        path = self._settings_file()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            data = {}
        data['resource_mode'] = mode
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, path)
        with self._cv:
            self._mode = mode
            self._facts = None
            self._sweep_models()
            self._cv.notify_all()
        log.info('resource mode: %s', mode)

    def profile(self):
        return MODES[self.mode()]

    # ---------------------------------------------------------------- capacities
    def facts(self, refresh=False):
        """Machine facts (cached 5 s): cores, RAM, GPU name/VRAM, NVENC/NVDEC presence."""
        now = time.monotonic()
        if self._facts is not None and not refresh and now - self._facts_at < 5:
            return self._facts
        total_ram, avail_ram = _ram_mb()
        g = _gpu() if getattr(settings, 'RESOURCE_GPU', 'auto') != 'off' else None
        f = {'cores': os.cpu_count() or 1, 'ram_total': total_ram, 'ram_avail': avail_ram,
             'gpu': g['name'] if g else None, 'vram_total': int(g['mem_total']) if g and g.get('mem_total') else 0,
             'vram_used': int(g['mem_used']) if g and g.get('mem_used') is not None else None}
        self._facts, self._facts_at = f, now
        return f

    def capacity(self):
        f, p = self.facts(), self.profile()
        gpu = f['vram_total'] > 0
        return {'cpu': max(1, int(round(f['cores'] * p['cpu_share']))),
                'ram_mb': max(512, (f['ram_total'] or 8192) - p['ram_reserve_mb']),
                'vram_mb': max(0, f['vram_total'] - p['vram_reserve_mb']) if gpu else 0,
                'nvenc': p['nvenc'] if gpu else 0, 'nvdec': p['nvdec'] if gpu else 0}

    def _used(self):
        u = {k: 0 for k in KINDS}
        for l in self._leases.values():
            for k, v in l.req.items():
                u[k] += v
        for m in self._models.values():
            u['vram_mb'] += m.vram_mb if m.device == 'cuda' else 0
            u['ram_mb'] += m.ram_mb
        return u

    def _ours_on_gpu(self):
        return self._used()['vram_mb']

    def _fits(self, req):
        cap, used, f, p = self.capacity(), self._used(), self.facts(), self.profile()
        for k, v in req.items():
            if v and used[k] + v > cap[k]:
                return False
        if req.get('vram_mb') and f['vram_used'] is not None:          # other programs on the GPU count too
            external = max(0, f['vram_used'] - self._ours_on_gpu())
            if f['vram_total'] - external - self._ours_on_gpu() - req['vram_mb'] < p['vram_reserve_mb']:
                return False
        if req.get('ram_mb') and f['ram_avail'] is not None:
            if f['ram_avail'] - req['ram_mb'] < p['ram_reserve_mb'] // 2:
                return False
        return True

    def _possible(self, req):
        cap = self.capacity()
        return all(not v or v <= cap[k] for k, v in req.items())

    # ---------------------------------------------------------------- job binding
    def bind_job(self, job_id, priority=0, cancel=None):
        """Called by the job runner in the worker thread: leases of this thread belong to the job."""
        self._jobs[job_id] = {'priority': priority, 'cancel': cancel}
        self._local.job = job_id

    @contextmanager
    def job_context(self, job_id):
        """Carry a job's priority / cancellation into a helper thread (e.g. the parallel screen scan)."""
        prev = getattr(self._local, 'job', None)
        self._local.job = job_id
        try:
            yield
        finally:
            self._local.job = prev

    def current_job(self):
        return getattr(self._local, 'job', None)

    def unbind_job(self, job_id):
        """Release everything the job still holds (safety net) and forget it."""
        with self._cv:
            for l in [l for l in self._leases.values() if l.owner == job_id]:
                log.warning('job %s ended holding lease %s (%s): released', job_id, l.id, l.label)
                self._release(l)
            self._jobs.pop(job_id, None)
        if getattr(self._local, 'job', None) == job_id:
            self._local.job = None

    # ---------------------------------------------------------------- acquire / release
    def _norm(self, req):
        out = {k: int(req.get(k) or 0) for k in KINDS}
        if any(v < 0 for v in out.values()):
            raise ValueError('negative resource request')
        return out

    def try_acquire(self, label='', **req):
        try:
            return self.acquire(label=label, wait=False, **req)
        except Unavailable:
            return None

    def acquire(self, label='', timeout=None, wait=True, cancel=None, priority=None, **req):
        """-> Lease. Raises Unavailable (impossible / busy without waiting / nested wait), ResourceTimeout,
        or JobCancelled when the owning job is cancelled while waiting."""
        from .procutil import JobCancelled
        req = self._norm(req)
        owner = self.current_job()
        ctx = self._jobs.get(owner) or {}
        prio = ctx.get('priority', 0) if priority is None else priority
        cancel = cancel or ctx.get('cancel')
        with self._cv:
            held = [l for l in self._leases.values() if l.thread == threading.get_ident()]
            if not self._possible(req):
                self.stats['unavailable'] += 1
                raise Unavailable(f'{label or "operation"}: needs {self._fmt(req)}, the machine has '
                                  f'{self._fmt(self.capacity())} in mode {self.mode()}')
            if self._fits(req) and not self._blocked_by_waiters(req, prio):
                return self._grant(req, label, owner, prio)
            if self._evict_for(req) and self._fits(req) and not self._blocked_by_waiters(req, prio):
                return self._grant(req, label, owner, prio)
            if not wait or held:
                self.stats['unavailable'] += 1
                raise Unavailable(f'{label or "operation"}: {self._fmt(req)} busy' +
                                  (' (this thread already holds a lease: no waiting, no deadlock)' if held else ''))
            t = _Ticket(next(self._seq), req, prio, time.monotonic(), owner)
            self._tickets.append(t)
            self.stats['waited'] += 1
            deadline = None if timeout is None else time.monotonic() + timeout
            try:
                while True:
                    if cancel and cancel():
                        raise JobCancelled()
                    if self._fits(req) and self._head_for(t):
                        return self._grant(req, label, owner, prio)
                    if self._evict_for(req):
                        continue
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        self.stats['timeouts'] += 1
                        raise ResourceTimeout(f'{label or "operation"}: waited {timeout:.0f} s for {self._fmt(req)}')
                    self._cv.wait(0.5 if left is None else min(0.5, left))
                    self.facts()
            finally:
                if t in self._tickets:
                    self._tickets.remove(t)
                self._cv.notify_all()

    def _order(self):
        return sorted(self._tickets, key=lambda t: (-t.priority, t.seq))

    def _head_for(self, t):
        """t may go if no ticket ahead of it (priority, age) also fits right now; an aged head blocks the rest."""
        now = time.monotonic()
        for o in self._order():
            if o is t:
                return True
            if self._fits(o.req) or now - o.since > AGING_S:
                return False
        return True

    def _blocked_by_waiters(self, req, prio):
        now = time.monotonic()
        return any((o.priority >= prio and self._fits(o.req)) or now - o.since > AGING_S for o in self._tickets)

    def _grant(self, req, label, owner, prio):
        l = Lease(next(self._seq), req, label, owner, prio, thread=threading.get_ident(), _mgr=self)
        self._leases[l.id] = l
        self.stats['granted'] += 1
        return l

    def _release(self, lease):
        with self._cv:
            if lease.released:
                return
            lease.released = True
            self._leases.pop(lease.id, None)
            self._cv.notify_all()

    @staticmethod
    def _fmt(req):
        names = {'cpu': 'CPU', 'ram_mb': 'RAM МБ', 'vram_mb': 'VRAM МБ', 'nvenc': 'NVENC', 'nvdec': 'NVDEC'}
        return ', '.join(f'{names[k]} {v}' for k, v in req.items() if v) or 'nothing'

    # ---------------------------------------------------------------- models
    @contextmanager
    def model(self, key, loader, *, device='cpu', vram_mb=0, ram_mb=0, group=None, unload=None, group_max=None):
        """Lazy-load (once, even with many threads), reuse and pin a model while in use.

        loader() -> object. VRAM/RAM estimates are accounted as resident memory; idle models are evicted (LRU) when
        memory is needed. group_max=1 keeps at most one model of the group resident (e.g. one Whisper size)."""
        key = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        with self._cv:
            lock = self._load_locks.setdefault(key, threading.Lock())
        with lock:
            with self._cv:
                m = self._models.get(key)
                if m is None:
                    need = {'vram_mb': vram_mb if device == 'cuda' else 0, 'ram_mb': ram_mb}
                    if group and group_max:
                        for other in [x for x in self._models.values() if x.group == group and x.refs == 0]:
                            self._unload(other, 'group limit')
                    if not self._possible(need):
                        raise Unavailable(f'model {key}: needs {self._fmt(need)}, machine has '
                                          f'{self._fmt(self.capacity())}')
                    if not self._fits(need):
                        self._evict_for(need)
                    if not self._fits(need) and device == 'cuda':
                        raise Unavailable(f'model {key}: not enough free VRAM for {vram_mb} МБ')
            if m is None:
                t0 = time.monotonic()
                obj = loader()
                with self._cv:
                    m = _Model(key, obj, device, int(vram_mb if device == 'cuda' else 0), int(ram_mb), group or '',
                               unload)
                    self._models[key] = m
                log.info('model %s loaded on %s in %.1f s', key, device, time.monotonic() - t0)
            with self._cv:
                m.refs += 1
                m.last_used = time.time()
        try:
            yield m.obj
        finally:
            with self._cv:
                m.refs -= 1
                m.last_used = time.time()
                if m.refs == 0 and not self.profile()['keep_models']:
                    self._unload(m, 'low resource mode')
                self._cv.notify_all()

    def loaded(self, key):
        key = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        return key in self._models

    def _unload(self, m, why):
        if self._models.get(m.key) is not m or m.refs:
            return False
        self._models.pop(m.key, None)
        try:
            if m.unload:
                m.unload(m.obj)
        except Exception as exc:
            log.warning('unloading %s failed: %s', m.key, exc)
        m.obj = None
        gc.collect()
        self.stats['evictions'] += 1
        log.info('model %s unloaded (%s)', m.key, why)
        self._cv.notify_all()
        return True

    def _evict_for(self, req):
        """Unload idle models (least recently used first) until req fits. -> True if something was unloaded."""
        if not (req.get('vram_mb') or req.get('ram_mb')):
            return False
        did = False
        for m in sorted([m for m in self._models.values() if m.refs == 0], key=lambda m: m.last_used):
            if self._fits(req):
                break
            if (req.get('vram_mb') and m.device == 'cuda') or req.get('ram_mb'):
                did |= self._unload(m, 'memory needed')
        return did

    def unload_models(self, group=None, why='requested'):
        with self._cv:
            return sum(self._unload(m, why) for m in list(self._models.values())
                       if m.refs == 0 and (group is None or m.group == group))

    def _sweep_models(self):
        ttl = self.profile()['idle_unload_s']
        now = time.time()
        for m in list(self._models.values()):
            if m.refs == 0 and (not self.profile()['keep_models'] or now - m.last_used > ttl):
                self._unload(m, 'idle')

    def sweep(self):
        with self._cv:
            self._sweep_models()

    def on_oom(self, where, exc=None):
        """A CUDA / RAM out-of-memory error: free every idle model, record the event."""
        self.stats['oom'] += 1
        self.events.append({'at': time.time(), 'what': 'oom', 'where': where, 'detail': str(exc)[:200] if exc else ''})
        del self.events[:-20]
        n = self.unload_models(why='out of memory')
        log.warning('out of memory in %s: %d idle model(s) unloaded', where, n)
        self.facts(refresh=True)

    def note(self, what, where, detail=''):
        self.events.append({'at': time.time(), 'what': what, 'where': where, 'detail': detail[:200]})
        del self.events[:-20]

    # ---------------------------------------------------------------- monitoring
    def snapshot(self):
        with self._cv:
            cap, used, f = self.capacity(), self._used(), self.facts()
            now = time.time()
            return {
                'mode': self.mode(), 'mode_locked': self.mode_locked(),
                'modes': {k: {'label': v['label'], 'note': v['note']} for k, v in MODES.items()},
                'machine': f, 'capacity': cap, 'used': used,
                'leases': [{'id': l.id, 'label': l.label, 'job': l.owner, 'req': {k: v for k, v in l.req.items() if v},
                            'held': round(now - l.granted_at, 1)} for l in self._leases.values()],
                'waiting': [{'job': t.owner, 'req': {k: v for k, v in t.req.items() if v}, 'priority': t.priority,
                             'waited': round(time.monotonic() - t.since, 1)} for t in self._order()],
                'models': [{'key': ' / '.join(map(str, m.key)), 'device': m.device, 'vram_mb': m.vram_mb,
                            'ram_mb': m.ram_mb, 'in_use': m.refs, 'idle': round(now - m.last_used),
                            'loads': m.loads} for m in self._models.values()],
                'stats': dict(self.stats), 'events': list(self.events[-8:])}


_manager = ResourceManager()


def manager():
    return _manager


def acquire(**kw):
    return _manager.acquire(**kw)


def try_acquire(**kw):
    return _manager.try_acquire(**kw)


def _reaper():
    while True:
        time.sleep(30)
        try:
            _manager.sweep()
        except Exception:
            pass


threading.Thread(target=_reaper, daemon=True, name='model-reaper').start()


def gpu_media_lease(label, nvenc=False, nvdec=False, cpu=2, wait=True, timeout=None):
    """Lease for one ffmpeg run that would like NVENC and/or NVDEC.
    -> (lease, use_nvenc, use_nvdec). If the GPU sessions are impossible, busy beyond `timeout`, or the thread
    already holds a lease, the run goes to the CPU (same filters; libx264 instead of NVENC) instead of failing."""
    mgr = _manager
    gpu_known = mgr.facts()['vram_total'] > 0
    want = {'nvenc': int(bool(nvenc)), 'nvdec': int(bool(nvdec)),
            'vram_mb': (300 if nvenc else 0) + (150 if nvdec else 0)}
    if gpu_known and (nvenc or nvdec):
        try:
            if timeout is None:
                timeout = float(os.getenv('CLIPFORGE_GPU_WAIT', '900'))
            return mgr.acquire(label=label, cpu=cpu, wait=wait, timeout=timeout, **want), bool(nvenc), bool(nvdec)
        except Unavailable as exc:
            mgr.note('cpu_fallback', label, str(exc))
            log.info('%s: GPU sessions unavailable (%s); using the CPU', label, exc)
            return cpu_lease(label + ' (CPU)', cpu, wait, timeout), False, False
    # no NVIDIA tools to account with (or nothing GPU-side asked): just the CPU share
    return cpu_lease(label, cpu, wait, timeout), bool(nvenc), bool(nvdec)


class _NullLease:
    label = ''

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def cpu_lease(label, cpu=1, wait=True, timeout=None):
    """CPU slots; a nested caller (already holding a lease) or an impossible request proceeds unaccounted
    rather than waiting - CPU work can always run, only more slowly."""
    try:
        return _manager.acquire(label=label, cpu=min(cpu, _manager.capacity()['cpu']), wait=wait, timeout=timeout)
    except ResourceTimeout:
        return _NullLease()
    except Unavailable:
        return _NullLease()
