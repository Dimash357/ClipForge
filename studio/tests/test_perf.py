"""Long-video performance work: parallel chunked screen scan (same result as one sequential pass), resume from the
partial cache, keyframe mode, ffmpeg stall watchdog, NVDEC detection + CPU fallback, live progress / ETA, CPU/GPU
load, and the API that shows them."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock, skipUnless

import numpy as np
from django.test import SimpleTestCase, TestCase

from studio import accel, jobs, motion_analysis as ma, procutil, scenes, sysload
from studio.models import Job, Project
from .helpers import HAVE_FFMPEG

needs_ffmpeg = skipUnless(HAVE_FFMPEG, 'needs ffmpeg')


def make_video(path, seconds, gop):
    """Small test video whose picture changes every few seconds (so the statistics differ frame to frame)."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'testsrc2=s=320x240:r=25:d={seconds}',
                    '-vf', "hue=h=t*40,drawbox=x='mod(t*60,300)':y=60:w=40:h=40:c=red:t=fill",
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-g', str(gop), '-keyint_min', str(gop),
                    '-sc_threshold', '0', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=120)
    return path


def reference_dense(path):
    """The v5.2 algorithm: ONE sequential ffmpeg pass, fps=1, 64x36 area scaling."""
    out = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-i', str(path), '-an', '-vf',
                          'fps=1,scale=64:36:flags=area,showinfo', '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24',
                          '-f', 'rawvideo', 'pipe:1'], capture_output=True, check=True)
    times = [float(x) for x in re.findall(r'pts_time:\s*([\d.]+)', out.stderr.decode())]
    frames = np.frombuffer(out.stdout, np.uint8).reshape(-1, 36, 64, 3)
    rows = [(t,) + ma._frame_row(f) for t, f in zip(times, frames)]
    return ma._profile(rows)


def same(a, b):
    return len(a.t) == len(b.t) and all(np.array_equal(getattr(a, k), getattr(b, k))
                                        for k in ('t', 'luma', 'sat', 'red', 'hist'))


@needs_ffmpeg
class ScreenScan(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dir = Path(tempfile.mkdtemp(prefix='cf_perf_'))
        cls.sparse = make_video(cls.dir / 'sparse.mp4', 40, 250)        # keyframe every 10 s -> dense mode
        cls.dense = make_video(cls.dir / 'dense.mp4', 40, 50)           # every 2 s -> keyframe mode
        cls.ref = reference_dense(cls.sparse)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)
        super().tearDownClass()

    def setUp(self):
        p = mock.patch.multiple(ma, CHUNK_MIN=6.0, CHUNK_MAX=6.0)        # 7 chunks for a 40 s video
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: os.environ.pop('CLIPFORGE_SCAN_WORKERS', None))

    def test_parallel_chunks_equal_one_sequential_pass(self):
        for workers in ('1', '3'):
            os.environ['CLIPFORGE_SCAN_WORKERS'] = workers
            sp = ma.scan_screen(self.sparse, 40.0, hw=False)
            self.assertEqual(ma.scan_screen.last_info['mode'], 'fps1')
            self.assertEqual(ma.scan_screen.last_info['chunks'], 7)
            self.assertTrue(same(sp, self.ref), workers)                   # bit-identical statistics

    def test_progress_is_real_and_monotonic(self):
        os.environ['CLIPFORGE_SCAN_WORKERS'] = '2'
        seen = []
        ma.scan_screen(self.sparse, 40.0, hw=False, progress=lambda f, info: seen.append(f))
        self.assertTrue(seen)
        self.assertEqual(seen[-1], 1.0)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in seen))

    def test_resume_from_partial_cache(self):
        os.environ['CLIPFORGE_SCAN_WORKERS'] = '1'
        part = self.dir / 'screen.part.npz'
        calls = []
        real = ma._chunk

        def counting(source, a, b, *args, **kw):
            calls.append(a)
            return real(source, a, b, *args, **kw)
        stop_after = [3]

        def cancel():
            return len(calls) > stop_after[0]
        with mock.patch.object(ma, '_chunk', side_effect=counting):
            with self.assertRaises(procutil.JobCancelled):
                ma.scan_screen(self.sparse, 40.0, hw=False, cancel=cancel, part_path=str(part))
        self.assertTrue(part.is_file())                                     # finished chunks were kept
        rows, done = ma._load_part(str(part), f'fps1:64:7:40.0')
        self.assertGreaterEqual(len(done), 2)
        first_run = len(calls)
        calls.clear()
        with mock.patch.object(ma, '_chunk', side_effect=counting):
            sp = ma.scan_screen(self.sparse, 40.0, hw=False, part_path=str(part))
        self.assertEqual(len(calls), 7 - len(done))                         # only the missing chunks
        self.assertLess(len(calls), first_run + 7)
        self.assertTrue(same(sp, self.ref))
        self.assertFalse(part.exists())

    def test_keyframe_mode_for_frequent_keyframes(self):
        sp = ma.scan_screen(self.dense, 40.0)
        self.assertEqual(ma.scan_screen.last_info['mode'], 'key')
        self.assertEqual(len(sp.t), 20)
        self.assertAlmostEqual(ma.keyframe_rate(self.dense), 0.5, delta=0.05)
        self.assertLess(ma.keyframe_rate(self.sparse), ma.KEY_DENSE)

    def test_nvdec_failure_falls_back_per_chunk(self):
        os.environ['CLIPFORGE_SCAN_WORKERS'] = '2'
        real = ma._chunk

        def gpu_breaks(source, a, b, width, t0, hw, *args):
            if hw:
                raise RuntimeError('CUDA_ERROR_NO_DEVICE')
            return real(source, a, b, width, t0, hw, *args)
        with mock.patch.object(ma, '_chunk', side_effect=gpu_breaks):
            sp = ma.scan_screen(self.sparse, 40.0, hw=True)
        self.assertTrue(ma.scan_screen.last_info.get('hw_failed'))
        self.assertTrue(same(sp, self.ref))

    def test_sample_frames_falls_back_to_cpu(self):
        cpu = list(scenes.sample_frames(self.sparse, 5.0, 7.0, 320, 240, fps=2.0, width=64, hw=False))
        with mock.patch.object(accel, 'download_format', return_value='nv12'):
            hw = list(scenes.sample_frames(self.sparse, 5.0, 7.0, 320, 240, fps=2.0, width=64, hw=True))
        self.assertEqual(len(cpu), 4)
        self.assertEqual([t for t, _ in hw], [t for t, _ in cpu])
        self.assertTrue(all(np.array_equal(a, b) for (_, a), (_, b) in zip(hw, cpu)))


class Watchdog(SimpleTestCase):
    def test_silent_process_is_killed_with_a_clear_error(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.PIPE)
        dog = procutil.Watchdog(proc, 'Тест', stall=1)
        t = time.time()
        self.assertEqual(proc.stdout.read(10), b'')                      # unblocked by the kill
        dog.stop()
        self.assertLess(time.time() - t, 10)
        with self.assertRaises(procutil.StallError) as cm:
            dog.check()
        self.assertIn('Тест', str(cm.exception))
        self.assertIn('Продолжить', str(cm.exception))

    def test_active_process_is_left_alone(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time,sys\nfor i in range(6):\n print(i, flush=True); '
                                 'time.sleep(0.4)'], stdout=subprocess.PIPE)
        dog = procutil.Watchdog(proc, 'Тест', stall=1)
        for line in proc.stdout:
            dog.touch()
        proc.wait()
        dog.stop()
        dog.check()
        self.assertEqual(proc.returncode, 0)


class Accel(SimpleTestCase):
    def setUp(self):
        accel._cache.clear()
        self.addCleanup(accel._cache.clear)
        self.f = Path(tempfile.mkdtemp()) / 'x.mp4'
        self.f.write_bytes(b'0' * 100)
        self.addCleanup(shutil.rmtree, self.f.parent, True)

    def test_detection_and_cache(self):
        ok = mock.Mock(returncode=0, stderr='')
        with mock.patch.object(accel, '_cuda_listed', return_value=True), \
                mock.patch.object(accel.subprocess, 'run', return_value=ok) as run:
            self.assertEqual(accel.decode_args(self.f), ['-hwaccel', 'cuda'])
            self.assertTrue(accel.nvdec_ok(self.f))
            self.assertEqual(run.call_count, 1)                           # tested once, then cached
            self.assertIn('-hwaccel_output_format', run.call_args[0][0])  # the test fails hard without NVDEC

    def test_failure_and_switch_off(self):
        bad = mock.Mock(returncode=1, stderr='Cannot load libcuda.so.1')
        with mock.patch.object(accel, '_cuda_listed', return_value=True), \
                mock.patch.object(accel.subprocess, 'run', return_value=bad):
            self.assertEqual(accel.decode_args(self.f), [])
        with self.settings(HW_DECODE='off'), mock.patch.object(accel.subprocess, 'run') as run:
            accel._cache.clear()
            self.assertFalse(accel.nvdec_ok(self.f))
            run.assert_not_called()


class LiveProgress(TestCase):
    def test_operations_eta_timings_and_messages(self):
        p = Project.objects.create(title='p')
        job = Job.objects.create(kind='full', project=p, status=Job.RUNNING)
        rep = jobs.Reporter(p.pk, job=job)
        rep.stage('transcribe', 'Распознаём речь...')
        rep.op('screen', 'Анализ экрана: 10%', frac=0.1)
        rep._ops['screen']['started'] -= 20                              # 20 s ago
        rep._stage_started -= 30
        rep.frac(0.5, 'Распознаём речь: 50%')
        act = Job.objects.get(pk=job.pk).activity
        ops = {o['key']: o for o in act['ops']}
        self.assertAlmostEqual(ops['screen']['eta'], 180, delta=3)       # 20 s for 10 % -> 180 s left
        self.assertAlmostEqual(ops['stage']['eta'], 30, delta=3)
        self.assertEqual(ops['stage']['label'], 'Распознаём речь: 50%')
        rep.op('screen', done=True)
        rep.stage('highlights', 'Ищем моменты...')
        self.assertIn('transcribe', rep.timings)
        self.assertIn('screen', rep.timings)
        # a new step's message is shown at once, even right after another write (the old "99 %" problem)
        rep.frac(0.6, 'Слушаем звук: 99%')
        rep.frac(0.65, 'Смотрим на экран: смерти, провалы...')
        self.assertTrue(Project.objects.get(pk=p.pk).message.startswith('Смотрим на экран'))

    def test_stall_warning(self):
        rep = jobs.Reporter(None)
        rep.stage('render', 'x')
        rep.last_progress -= 4000
        with self.settings(JOB_STALL_WARN_SECONDS=600):
            self.assertGreaterEqual(rep.activity()['stalled_for'], 4000)

    def test_status_api_shows_live_job_and_load(self):
        p = Project.objects.create(title='p', status='processing')
        Job.objects.create(kind='full', project=p, status=Job.RUNNING, worker=f'{jobs.PROCESS}:0',
                           heartbeat_at=__import__('django.utils.timezone', fromlist=['now']).now(),
                           activity={'ops': [{'key': 'stage', 'label': 'x', 'frac': 0.5, 'elapsed': 10, 'eta': 10,
                                              'started': time.time() - 10}], 'timings': {}, 'at': time.time()})
        d = self.client.get(f'/api/projects/{p.pk}/status/').json()
        self.assertTrue(d['job']['alive'])
        self.assertEqual(d['job']['ops'][0]['label'], 'x')
        self.assertIn('cpu', d['load'])
        self.assertEqual(self.client.get('/api/system/load/').status_code, 200)


class Load(SimpleTestCase):
    def test_snapshot(self):
        sysload._cache.update(at=0.0, value=None)
        sysload.cpu_percent()
        time.sleep(0.2)
        snap = sysload.snapshot()
        self.assertIn('cpu', snap)
        self.assertIn('gpu', snap)                                       # None without an NVIDIA driver
        if snap['cpu'] is not None:
            self.assertTrue(0 <= snap['cpu'] <= 100)
