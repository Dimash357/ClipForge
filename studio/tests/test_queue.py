"""Persistent job queue (1C): ordering, limits, duplicate protection, crash recovery with checkpoints, pause,
cancel, retry, attempt limits, SQLite lock retries, the queue page, and a real Stream Clipper resume."""
import os
import shutil
import tempfile
import threading
import time
from datetime import timedelta
from unittest import mock, skipUnless

from django.core.files import File
from django.db import IntegrityError, OperationalError
from django.test import TransactionTestCase, override_settings
from django.utils import timezone

from studio import exports, jobs, pipeline
from studio.models import Clip, Job, Project
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'

LOG = []                    # (job id, event)
GATE = threading.Event()    # handlers that wait for the test
RUNNING = [0, 0]            # current, max seen


def h_steps(job, rep):
    """Three checkpointed steps; skips the ones already done. payload: wait=True blocks before each step,
    fail_once=True fails the first run of step b."""
    for s in ('a', 'b', 'c'):
        if rep.done(s):
            LOG.append((job.pk, f'skip-{s}'))
            continue
        if job.payload.get('wait'):
            GATE.wait(10)
        if s == 'b' and job.payload.get('fail_once') and not rep.done('failed-once'):
            rep.checkpoint('failed-once')
            raise RuntimeError('boom in b')
        LOG.append((job.pk, s))
        rep.checkpoint(s)


def h_heavy(job, rep):
    RUNNING[0] += 1
    RUNNING[1] = max(RUNNING)
    LOG.append((job.pk, 'start'))
    try:
        GATE.wait(10)
        time.sleep(0.05)
    finally:
        RUNNING[0] -= 1


def h_cancellable(job, rep):
    LOG.append((job.pk, 'start'))
    for _ in range(400):
        rep.check()
        time.sleep(0.025)


jobs.register('t_steps', h_steps, label='test steps', needs_project=False)
jobs.register('t_heavy', h_heavy, label='test heavy', needs_project=False)
jobs.register('t_light', h_heavy, label='test light', resource='light', needs_project=False)
jobs.register('t_cancel', h_cancellable, label='test cancel', needs_project=False)


def hold():
    """Keep new jobs queued: no worker may claim anything (also the ones started by earlier tests)."""
    from contextlib import ExitStack
    st = ExitStack()
    st.enter_context(mock.patch.object(jobs, 'wake'))
    st.enter_context(mock.patch.object(jobs, '_claim', return_value=None))
    return st


def crashed_job(kind='t_steps', attempts=1, checkpoints=None, project=None, age=600, **payload):
    """A job left 'running' by a process that died `age` seconds ago."""
    return Job.objects.create(kind=kind, project=project, payload=payload, status=Job.RUNNING, attempts=attempts,
                              worker='deadhost:4242:abcdef:0', checkpoints=checkpoints or {},
                              heartbeat_at=timezone.now() - timedelta(seconds=age),
                              started_at=timezone.now() - timedelta(seconds=age + 5))


class QueueBase(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        jobs._pause.clear()
        LOG.clear()
        GATE.clear()
        RUNNING[:] = [0, 0]
        jobs._last_sweep[0] = 0.0
        self.addCleanup(GATE.set)

    def status(self, job, want, timeout=15):
        ok = wait_for(lambda: Job.objects.get(pk=job.pk).status == want, timeout=timeout)
        self.assertTrue(ok, f'job {job.pk}: {Job.objects.get(pk=job.pk).status} != {want}')
        return Job.objects.get(pk=job.pk)


# =====================================================================================================
class Basics(QueueBase):
    def test_steps_complete_with_checkpoints_and_time(self):
        job = jobs.submit('t_steps')
        job = self.status(job, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['a', 'b', 'c'])
        self.assertEqual(set(job.checkpoints), {'a', 'b', 'c'})
        self.assertEqual((job.attempts, job.worker, job.progress), (1, '', 100))
        self.assertIsNotNone(job.finished_at)
        self.assertGreater(job.run_seconds, 0)

    def test_priority_then_age(self):
        GATE.clear()
        first = jobs.submit('t_heavy')
        self.assertTrue(wait_for(lambda: (first.pk, 'start') in LOG))
        low = jobs.submit('t_heavy', priority=0)
        high = jobs.submit('t_heavy', priority=5)
        low2 = jobs.submit('t_heavy', priority=0)
        GATE.set()
        for j in (first, low, high, low2):
            self.status(j, Job.COMPLETED)
        starts = [j for j, e in LOG if e == 'start']
        self.assertEqual(starts, [first.pk, high.pk, low.pk, low2.pk])

    @override_settings(MAX_CONCURRENT_JOBS=1)
    def test_only_one_heavy_job_runs_at_a_time(self):
        js = [jobs.submit('t_heavy') for _ in range(3)]
        GATE.set()
        for j in js:
            self.status(j, Job.COMPLETED)
        self.assertEqual(RUNNING[1], 1)

    def test_claim_respects_the_heavy_limit_across_processes(self):
        Job.objects.create(kind='t_heavy', status=Job.RUNNING, worker='otherhost:1:x:0', heartbeat_at=timezone.now())
        heavy = Job.objects.create(kind='t_heavy', resource='heavy')
        light = Job.objects.create(kind='t_light', resource='light')
        got = jobs._claim('me:1:y:0')
        self.assertEqual(got.pk, light.pk)                  # the heavy one waits for the other process
        self.assertEqual(Job.objects.get(pk=heavy.pk).status, Job.QUEUED)

    def test_a_job_is_claimed_only_once(self):
        j = Job.objects.create(kind='t_light', resource='light')
        a = jobs._claim('p1:1:a:0')
        b = jobs._claim('p2:2:b:0')
        self.assertEqual((a.pk, b), (j.pk, None))
        self.assertEqual(Job.objects.get(pk=j.pk).worker, 'p1:1:a:0')
        Job.objects.filter(pk=j.pk).update(status=Job.COMPLETED)

    def test_unknown_kind_and_missing_project(self):
        with self.assertRaises(ValueError):
            jobs.submit('nope')
        with self.assertRaises(ValueError):
            jobs.submit('full')                              # Stream Clipper jobs need a project


class Duplicates(QueueBase):
    def test_one_live_job_per_project(self):
        p = Project.objects.create(title='p', status='done')
        with hold():                # keep it queued
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertFalse(jobs.enqueue('render', p.pk))
        self.assertEqual(Job.objects.filter(project=p).count(), 1)
        with self.assertRaises(IntegrityError):              # enforced by the database itself
            Job.objects.create(kind='full', project=p, status=Job.PAUSED)
        Job.objects.filter(project=p).update(status=Job.COMPLETED)
        with hold():
            self.assertTrue(jobs.enqueue('full', p.pk))     # finished jobs do not block a new one

    def test_sqlite_lock_is_retried_not_fatal(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise OperationalError('database is locked')
            return 'ok'
        self.assertEqual(jobs._retry(flaky), 'ok')
        with self.assertRaises(OperationalError):
            jobs._retry(lambda: (_ for _ in ()).throw(OperationalError('no such table: x')))


# =====================================================================================================
class Recovery(QueueBase):
    def test_crashed_job_resumes_and_skips_finished_steps(self):
        job = crashed_job(checkpoints={'a': {'at': 'x'}})
        self.assertEqual(jobs.sweep_stale(force=True), 1)
        job = self.status(job, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['skip-a', 'b', 'c'])
        self.assertEqual(job.attempts, 2)

    def test_recover_once_at_startup(self):
        p = Project.objects.create(title='p', status='processing')
        job = crashed_job(kind='t_steps', checkpoints={'a': {}, 'b': {}})
        orphan = Project.objects.create(title='o', status='fetching')            # legacy: no job record
        jobs._recovered = False
        self.assertEqual(jobs.recover_once(), 3)              # p and orphan have no live job + 1 crashed job
        self.status(job, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['skip-a', 'skip-b', 'c'])
        self.assertEqual(Project.objects.get(pk=orphan.pk).status, 'interrupted')
        self.assertEqual(Project.objects.get(pk=p.pk).status, 'interrupted')

    def test_dead_owner_on_this_computer_is_recovered_at_once(self):
        import subprocess
        import sys
        dead = subprocess.Popen([sys.executable, '-c', 'pass'])
        dead.wait()                                           # a real process that is gone now
        self.assertTrue(jobs.owner_dead(f'{jobs.HOST}:{dead.pid}:abc123:0'))
        self.assertTrue(jobs.owner_dead(f'{jobs.HOST}:{os.getpid()}:oldrun:0'))       # same pid, earlier run
        self.assertFalse(jobs.owner_dead(f'{jobs.PROCESS}:0'))
        self.assertFalse(jobs.owner_dead(f'otherpc:{dead.pid}:abc123:0'))              # other computer: heartbeat only
        job = crashed_job(age=1)                              # heartbeat still fresh...
        Job.objects.filter(pk=job.pk).update(worker=f'{jobs.HOST}:{dead.pid}:abc123:0')
        self.assertEqual(jobs.sweep_stale(force=True), 1)     # ...but its process is dead
        self.status(job, Job.COMPLETED)

    def test_fresh_heartbeat_is_left_alone(self):
        job = crashed_job(age=2)
        self.assertEqual(jobs.sweep_stale(force=True), 0)
        self.assertEqual(Job.objects.get(pk=job.pk).status, Job.RUNNING)
        Job.objects.filter(pk=job.pk).update(status=Job.COMPLETED)

    def test_no_endless_restarts(self):
        job = crashed_job(attempts=3)                         # crashed on its 3rd start (max_attempts = 3)
        self.assertEqual(jobs.sweep_stale(force=True), 1)
        job = Job.objects.get(pk=job.pk)
        self.assertEqual(job.status, Job.INTERRUPTED)
        self.assertIn('остановлен', job.message)
        self.assertTrue(jobs.resume_job(job.pk))              # a manual retry is allowed and resets the counter
        job = self.status(job, Job.COMPLETED)
        self.assertEqual(job.attempts, 1)

    @override_settings(JOB_AUTO_RESUME=False)
    def test_auto_resume_can_be_switched_off(self):
        job = crashed_job()
        jobs.sweep_stale(force=True)
        self.assertEqual(Job.objects.get(pk=job.pk).status, Job.INTERRUPTED)


# =====================================================================================================
class Control(QueueBase):
    def test_pause_at_a_safe_point_then_resume(self):
        job = jobs.submit('t_steps', wait=True)
        self.assertTrue(wait_for(lambda: Job.objects.get(pk=job.pk).status == Job.RUNNING))
        self.assertTrue(jobs.pause_job(job.pk))
        GATE.set()                                            # step a finishes, its checkpoint is the safe point
        job = self.status(job, Job.PAUSED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['a'])
        self.assertIn('a', job.checkpoints)
        self.assertTrue(jobs.resume_job(job.pk))
        self.status(job, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['a', 'skip-a', 'b', 'c'])

    def test_pause_and_cancel_while_queued(self):
        with hold():
            a = jobs.submit('t_steps')
            b = jobs.submit('t_steps')
        self.assertTrue(jobs.pause_job(a.pk))
        self.assertTrue(jobs.cancel_job(b.pk))
        self.assertEqual(Job.objects.get(pk=a.pk).status, Job.PAUSED)
        self.assertEqual(Job.objects.get(pk=b.pk).status, Job.CANCELLED)
        self.assertFalse(jobs.cancel_job(b.pk))
        self.assertTrue(jobs.cancel_job(a.pk))                # a paused job can be cancelled too

    def test_cancel_running(self):
        job = jobs.submit('t_cancel')
        self.assertTrue(wait_for(lambda: (job.pk, 'start') in LOG))
        t = time.time()
        self.assertTrue(jobs.cancel_job(job.pk))
        job = self.status(job, Job.CANCELLED)
        self.assertLess(time.time() - t, 3)
        self.assertTrue(job.cancel_requested)

    def test_cancel_requested_by_another_process(self):
        job = jobs.submit('t_cancel')
        self.assertTrue(wait_for(lambda: (job.pk, 'start') in LOG))
        Job.objects.filter(pk=job.pk).update(cancel_requested=True)     # e.g. the queue page of a second server
        self.status(job, Job.CANCELLED, timeout=jobs.HEARTBEAT_SECONDS + 5)

    def test_failed_job_is_retried_from_its_checkpoint(self):
        job = jobs.submit('t_steps', fail_once=True)
        job = self.status(job, Job.FAILED)
        self.assertIn('boom in b', job.error)
        self.assertIsNotNone(jobs.retry_job(job.pk))
        job = self.status(job, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == job.pk], ['a', 'skip-a', 'b', 'c'])
        self.assertEqual(job.error, '')

    def test_completed_job_retry_is_a_new_run(self):
        job = self.status(jobs.submit('t_steps'), Job.COMPLETED)
        new = jobs.retry_job(job.pk)
        self.assertNotEqual(new.pk, job.pk)
        self.status(new, Job.COMPLETED)
        self.assertEqual([e for j, e in LOG if j == new.pk], ['a', 'b', 'c'])


class Page(QueueBase):
    def test_queue_page_and_api(self):
        p = Project.objects.create(title='Мой стрим', status='done')
        st = hold()
        st.__enter__()
        self.addCleanup(st.__exit__, None, None, None)
        jobs.enqueue('full', p.pk)
        job = Job.objects.get(project=p)
        self.assertContains(self.client.get('/queue/'), 'Очередь задач')
        d = self.client.get('/api/jobs/').json()
        self.assertEqual(d['live'][0]['title'], 'Мой стрим')
        self.assertEqual(d['live'][0]['status'], 'queued')
        r = self.client.post(f'/api/jobs/{job.pk}/priority/', {'priority': 7}, content_type='application/json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Job.objects.get(pk=job.pk).priority, 7)
        self.assertEqual(self.client.post(f'/api/jobs/{job.pk}/pause/').status_code, 200)
        self.assertEqual(Project.objects.get(pk=p.pk).status, 'paused')
        self.assertEqual(self.client.post(f'/api/jobs/{job.pk}/pause/').status_code, 409)
        self.assertEqual(self.client.post(f'/api/jobs/{job.pk}/explode/').status_code, 404)
        self.assertEqual(self.client.post(f'/api/jobs/{job.pk}/cancel/').status_code, 200)
        self.assertEqual(Project.objects.get(pk=p.pk).status, 'cancelled')
        self.assertEqual(self.client.get('/api/jobs/').json()['recent'][0]['status'], 'cancelled')


# =====================================================================================================
@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class StreamClipperResume(QueueBase):
    """A full Stream Clipper job 'crashes' after its first clip: the resumed job keeps that clip, renders only the
    rest and does not transcribe or pick moments again."""

    def setUp(self):
        super().setUp()
        self.media = tempfile.mkdtemp()
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def test_resume_after_crash_skips_finished_clips(self):
        p = Project.objects.create(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode='center',
                                   options=exports.resolve({'preset': '720p', 'quality': 'small'}))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            job = self.status(Job.objects.get(project=p), Job.COMPLETED, timeout=300)
        self.assertTrue(set(job.checkpoints) >= {'source', 'transcript', 'clips', 'rendered'}, job.checkpoints)
        done = list(p.clips.order_by('position'))
        kept = {c.pk: c.video.name for c in done}
        # rewind to "crashed while rendering the last clip": the done clips are checkpointed, one more was in progress
        todo = Clip.objects.create(project=p, start=9.0, end=15.0, position=len(done), status='rendering',
                                   title='Highlight x')
        cps = dict(job.checkpoints, clips={'ids': list(kept) + [todo.pk], 'review': False},
                   rendered={'ids': list(kept)})
        Job.objects.filter(pk=job.pk).update(status=Job.RUNNING, worker='deadhost:1:x:0', checkpoints=cps,
                                             heartbeat_at=timezone.now() - timedelta(minutes=5), attempts=1)
        Project.objects.filter(pk=p.pk).update(status='processing')
        with mock.patch.object(pipeline, 'transcribe', side_effect=AssertionError('transcribed again')), \
                mock.patch.object(pipeline, 'pick_highlights', side_effect=AssertionError('picked again')):
            self.assertEqual(jobs.sweep_stale(force=True), 1)
            job = self.status(job, Job.COMPLETED, timeout=300)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        for c in p.clips.filter(pk__in=list(kept)):
            self.assertEqual(c.video.name, kept[c.pk])                   # not rendered again
        todo.refresh_from_db()
        self.assertEqual(todo.status, 'ready')                           # the unfinished one is rendered now
        self.assertTrue(todo.video)
        self.assertEqual(p.clips.count(), len(kept) + 1)                 # no duplicates, nothing deleted
        self.assertEqual(job.attempts, 2)
        self.assertIn(todo.pk, job.checkpoints['rendered']['ids'])
