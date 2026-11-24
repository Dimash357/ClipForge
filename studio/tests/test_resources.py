"""1D resource manager: capacities per mode, all-or-nothing leases, waiting by priority, no deadlock, cancel,
timeouts, external GPU use, model registry (lazy, shared, pinned, LRU eviction, group limit, low-mode unload,
OOM), job binding, CPU fallbacks, Whisper OOM handling, the job queue limit and the API."""
import json
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase, TestCase, override_settings

from studio import diagnostics, jobs, pipeline, resources
from studio.procutil import JobCancelled

GPU = {'cores': 16, 'ram_total': 24000, 'ram_avail': 16000, 'gpu': 'RTX 4060 Laptop GPU', 'vram_total': 8188,
       'vram_used': 300}
NOGPU = dict(GPU, gpu=None, vram_total=0, vram_used=None)


def manager(facts=GPU, mode='balanced'):
    m = resources.ResourceManager()
    m.facts = lambda refresh=False: dict(facts)
    m._mode = mode
    return m


def in_thread(fn):
    out = {}

    def run():
        try:
            out['value'] = fn()
        except BaseException as exc:
            out['error'] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


class Capacity(SimpleTestCase):
    def test_modes(self):
        b, p, low = manager(mode='balanced'), manager(mode='performance'), manager(mode='low')
        self.assertEqual(b.capacity(), {'cpu': 16, 'ram_mb': 20928, 'vram_mb': 7164, 'nvenc': 2, 'nvdec': 2})
        self.assertEqual(p.capacity()['vram_mb'], 7676)
        self.assertEqual(low.capacity()['cpu'], 8)
        self.assertEqual(low.capacity()['vram_mb'], 6140)
        self.assertEqual((b.profile()['heavy_jobs'], p.profile()['heavy_jobs']), (1, 2))

    def test_without_gpu_everything_gpu_side_is_unavailable_at_once(self):
        m = manager(NOGPU)
        self.assertEqual((m.capacity()['vram_mb'], m.capacity()['nvenc'], m.capacity()['nvdec']), (0, 0, 0))
        t = time.monotonic()
        with self.assertRaises(resources.Unavailable):
            m.acquire(vram_mb=100)
        self.assertLess(time.monotonic() - t, 0.5)                    # never waits for something that cannot come
        with m.acquire(cpu=4) as lease:
            self.assertEqual(lease.req['cpu'], 4)

    def test_other_programs_on_the_gpu_are_respected(self):
        busy = manager(dict(GPU, vram_used=6500))                       # e.g. a game holds 6.5 GB
        self.assertIsNone(busy.try_acquire(vram_mb=1000))
        self.assertIsNotNone(busy.try_acquire(vram_mb=500))

    def test_low_free_ram_is_respected(self):
        m = manager(dict(GPU, ram_avail=3000))
        self.assertIsNone(m.try_acquire(ram_mb=2000))


class Leases(SimpleTestCase):
    def test_all_or_nothing_and_release_wakes_waiters(self):
        m = manager()
        a = m.acquire(vram_mb=5000, nvenc=1, label='A')
        t, out = in_thread(lambda: m.acquire(vram_mb=3000, nvenc=1, label='B', timeout=5))
        time.sleep(0.3)
        self.assertNotIn('value', out)                                  # B waits, holding nothing
        self.assertEqual(m._used()['nvenc'], 1)
        a.release()
        t.join(5)
        self.assertEqual(out['value'].label, 'B')
        out['value'].release()
        self.assertEqual(m._used(), {k: 0 for k in resources.KINDS})

    def test_timeout(self):
        m = manager()
        with m.acquire(nvenc=2):
            t, out = in_thread(lambda: m.acquire(nvenc=1, timeout=0.3))
            t.join(3)
        self.assertIsInstance(out['error'], resources.ResourceTimeout)

    def test_priority_then_arrival(self):
        m = manager()
        hold = m.acquire(nvenc=2)
        order = []

        def ask(name, prio):
            with m.acquire(nvenc=2, priority=prio, timeout=10):
                order.append(name)
                time.sleep(0.05)
        ts = [in_thread(lambda: ask('low-1', 0))[0]]
        time.sleep(0.1)
        ts.append(in_thread(lambda: ask('low-2', 0))[0])
        time.sleep(0.1)
        ts.append(in_thread(lambda: ask('high', 5))[0])
        time.sleep(0.2)
        hold.release()
        for t in ts:
            t.join(10)
        self.assertEqual(order, ['high', 'low-1', 'low-2'])

    def test_a_holder_never_waits_so_leases_cannot_deadlock(self):
        m = manager()
        a = m.acquire(nvdec=1)
        b_holds, main_done = threading.Event(), threading.Event()
        res = {}

        def other():
            with m.acquire(nvdec=1):
                b_holds.set()
                try:
                    m.acquire(nvdec=1, timeout=30)                      # would be a cycle: refused at once
                except resources.Unavailable as exc:
                    res['b'] = exc
                main_done.wait(5)
        t = threading.Thread(target=other)
        t.start()
        b_holds.wait(5)
        t0 = time.monotonic()
        with self.assertRaises(resources.Unavailable):
            m.acquire(nvdec=1, timeout=30)
        self.assertLess(time.monotonic() - t0, 1)
        main_done.set()
        t.join(5)
        self.assertIn('already holds', str(res['b']))
        a.release()

    def test_cancel_while_waiting(self):
        m = manager()
        flag = threading.Event()
        with m.acquire(nvenc=2):
            t, out = in_thread(lambda: m.acquire(nvenc=1, cancel=flag.is_set))
            time.sleep(0.2)
            flag.set()
            t.join(3)
        self.assertIsInstance(out['error'], JobCancelled)
        self.assertEqual(m._tickets, [])

    def test_no_overbooking_under_concurrency(self):
        m = manager()
        cap = m.capacity()
        bad = []
        lock = threading.Lock()

        def worker(seed):
            rnd = random.Random(seed)
            for _ in range(60):
                req = {'cpu': rnd.randint(0, 8), 'vram_mb': rnd.choice([0, 1000, 3000]), 'nvdec': rnd.randint(0, 1)}
                try:
                    with m.acquire(timeout=5, **req):
                        with lock:
                            u = m._used()
                            if any(u[k] > cap[k] for k in cap):
                                bad.append(u)
                        time.sleep(rnd.random() / 200)
                except resources.Unavailable:
                    pass
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        self.assertEqual(bad, [])
        self.assertEqual(m._used(), {k: 0 for k in resources.KINDS})

    def test_job_binding_priority_and_safety_release(self):
        m = manager()
        m.bind_job(42, priority=7, cancel=lambda: False)
        lease = m.acquire(cpu=2, label='leaked')
        self.assertEqual((lease.owner, lease.priority), (42, 7))
        m.unbind_job(42)
        self.assertTrue(lease.released)
        self.assertEqual(m._used()['cpu'], 0)

    def test_cpu_fallbacks(self):
        m = manager(NOGPU)
        with mock.patch.object(resources, '_manager', m):
            lease, enc, dec = resources.gpu_media_lease('render', nvenc=True, nvdec=True)
            with lease:
                self.assertEqual((enc, dec), (True, True))             # nothing to account with: ffmpeg decides
        m = manager()
        with mock.patch.object(resources, '_manager', m):
            hold = m.acquire(nvenc=2)
            lease, enc, dec = resources.gpu_media_lease('render', nvenc=True, timeout=0.2)
            with lease:
                self.assertFalse(enc)                                   # busy beyond the wait -> CPU encoder
            hold.release()
            with resources.gpu_media_lease('render', nvenc=True)[0]:
                pass


class Models(SimpleTestCase):
    def test_lazy_shared_loaded_once(self):
        m = manager()
        calls = []

        def loader():
            calls.append(1)
            time.sleep(0.2)
            return object()
        got = []

        def use():
            with m.model('x', loader, device='cuda', vram_mb=1000) as obj:
                got.append(obj)
        ts = [threading.Thread(target=use) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len({id(o) for o in got}), 1)
        self.assertEqual(m._used()['vram_mb'], 1000)                   # resident, accounted

    def test_lru_eviction_never_touches_models_in_use(self):
        m = manager()
        unloaded = []
        with m.model('a', object, device='cuda', vram_mb=3000, unload=lambda o: unloaded.append('a')):
            with m.model('b', object, device='cuda', vram_mb=3000, unload=lambda o: unloaded.append('b')):
                pass
            with m.acquire(vram_mb=2000, timeout=1):                     # needs b's memory, a is pinned
                self.assertEqual(unloaded, ['b'])
        self.assertTrue(m.loaded('a'))

    def test_group_limit_and_low_mode(self):
        m = manager()
        with m.model(('whisper', 'small'), object, device='cuda', vram_mb=900, group='whisper', group_max=1):
            pass
        with m.model(('whisper', 'medium'), object, device='cuda', vram_mb=2100, group='whisper', group_max=1):
            pass
        self.assertEqual([x['key'] for x in m.snapshot()['models']], ['whisper / medium'])
        low = manager(mode='low')
        with low.model('x', object, ram_mb=100):
            self.assertTrue(low.loaded('x'))
        self.assertFalse(low.loaded('x'))                               # unloaded right after use

    def test_oom_and_idle_unload(self):
        m = manager()
        with m.model('a', object, device='cuda', vram_mb=1000):
            pass
        m.on_oom('test', RuntimeError('CUDA out of memory'))
        self.assertFalse(m.loaded('a'))
        self.assertEqual(m.snapshot()['events'][-1]['what'], 'oom')
        with m.model('b', object, ram_mb=10):
            pass
        m._models[('b',)].last_used -= 10_000
        m.sweep()
        self.assertFalse(m.loaded('b'))

    def test_model_bigger_than_the_card(self):
        with self.assertRaises(resources.Unavailable):
            with manager().model('huge', object, device='cuda', vram_mb=20000):
                pass

    def test_is_oom(self):
        for msg in ('CUDA failed with error out of memory', 'CUDA_ERROR_OUT_OF_MEMORY', 'std::bad_alloc'):
            self.assertTrue(resources.is_oom(RuntimeError(msg)))
        self.assertFalse(resources.is_oom(RuntimeError('cublas64_12.dll is not found')))


class Seg:
    def __init__(self, a, b, t):
        self.start, self.end, self.text, self.words = a, b, t, []


class WhisperOOM(SimpleTestCase):
    def setUp(self):
        diagnostics._whisper_gpu_failed = False
        pipeline._model_cache.clear()
        self.addCleanup(setattr, diagnostics, '_whisper_gpu_failed', False)
        self.addCleanup(pipeline._model_cache.clear)

    def test_out_of_memory_goes_to_cpu_but_keeps_the_gpu_for_next_time(self):
        calls = []

        class Model:
            def __init__(self, device):
                self.device = device

            def transcribe(self, *a, **k):
                calls.append(self.device)
                if self.device == 'cuda':
                    raise RuntimeError('CUDA failed with error out of memory')
                return iter([Seg(0, 1, 'раз')]), mock.Mock(language='ru')
        with mock.patch.object(diagnostics, 'whisper_settings', return_value=('cuda', 'float16', 'x')), \
                mock.patch.object(pipeline, '_get_model', side_effect=lambda n, d, c: Model(d)), \
                mock.patch.dict('os.environ', {'WHISPER_BATCH': '0'}):
            out = pipeline.transcribe('x.wav', 'auto')
        self.assertEqual(calls, ['cuda', 'cpu'])
        self.assertEqual(out[0]['text'], 'раз')
        self.assertFalse(diagnostics._whisper_gpu_failed)              # OOM is not a broken CUDA install

    def test_models_are_reused_between_transcriptions(self):
        made = []

        class Model:
            def transcribe(self, *a, **k):
                return iter([Seg(0, 1, 'раз')]), mock.Mock(language='ru')
        with mock.patch.object(diagnostics, 'whisper_settings', return_value=('cpu', 'int8', 'x')), \
                mock.patch.object(pipeline, '_get_model', side_effect=lambda *a: made.append(1) or Model()), \
                mock.patch.dict('os.environ', {'WHISPER_BATCH': '0'}):
            pipeline.transcribe('a.wav', 'auto')
            pipeline.transcribe('b.wav', 'auto')
        self.assertEqual(len(made), 1)


class QueueAndApi(TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ov = override_settings(CLIPFORGE_SETTINGS_FILE=self.dir / 's.json')
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.addCleanup(setattr, resources._manager, '_mode', None)
        resources._manager._mode = None

    def test_mode_sets_the_heavy_job_limit_and_persists(self):
        with mock.patch.dict('os.environ', {}, clear=False) as env:
            env.pop('CLIPFORGE_MAX_JOBS', None)
            env.pop('CLIPFORGE_RESOURCE_MODE', None)
            self.assertEqual(jobs.max_heavy(), 1)
            r = self.client.post('/api/resources/mode/', {'mode': 'performance'}, content_type='application/json')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(jobs.max_heavy(), 2)
            self.assertEqual(json.loads((self.dir / 's.json').read_text())['resource_mode'], 'performance')
            resources._manager._mode = None                              # a restart reads the file
            self.assertEqual(resources._manager.mode(), 'performance')
            env['CLIPFORGE_MAX_JOBS'] = '1'                              # an explicit limit still wins
            self.assertEqual(jobs.max_heavy(), 1)
        self.assertEqual(self.client.post('/api/resources/mode/', {'mode': 'turbo'},
                                          content_type='application/json').status_code, 400)

    def test_env_locks_the_mode(self):
        with mock.patch.dict('os.environ', {'CLIPFORGE_RESOURCE_MODE': 'low'}):
            self.assertEqual(resources._manager.mode(), 'low')
            self.assertEqual(self.client.post('/api/resources/mode/', {'mode': 'balanced'},
                                              content_type='application/json').status_code, 409)

    def test_snapshot_api(self):
        d = self.client.get('/api/resources/').json()
        for k in ('mode', 'modes', 'machine', 'capacity', 'used', 'leases', 'waiting', 'models', 'load'):
            self.assertIn(k, d)
        self.assertEqual(self.client.post('/api/resources/unload/').status_code, 200)
        self.assertContains(self.client.get('/queue/'), 'resources.js')
