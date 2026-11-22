import json
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

from django.core.files.base import ContentFile
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import Client, TestCase, TransactionTestCase, override_settings

from studio import jobs, pipeline
from studio.models import Clip, Project
from studio.procutil import JobCancelled, run_ffmpeg


def wait_for(cond, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class JobRunner(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()

    def test_duplicate_enqueue_rejected(self):
        p = Project.objects.create(title='a')
        gate = threading.Event()
        with mock.patch.object(pipeline, 'run_job', lambda *a, **k: gate.wait(5)):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertFalse(jobs.enqueue('full', p.pk))
            gate.set()
            self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk)))

    def test_only_one_heavy_job_at_a_time(self):
        a, b = Project.objects.create(title='a'), Project.objects.create(title='b')
        state = {'now': 0, 'max': 0}
        lock = threading.Lock()

        def fake(kind, pk, payload, rep):
            with lock:
                state['now'] += 1
                state['max'] = max(state['max'], state['now'])
            time.sleep(0.3)
            with lock:
                state['now'] -= 1
            jobs.set_state(pk, status='done')

        with mock.patch.object(pipeline, 'run_job', fake):
            jobs.enqueue('full', a.pk)
            jobs.enqueue('full', b.pk)
            self.assertTrue(wait_for(lambda: not jobs.is_active(a.pk) and not jobs.is_active(b.pk)))
        self.assertEqual(state['max'], 1)
        self.assertEqual(Project.objects.get(pk=a.pk).status, 'done')

    def test_cancel_running_job(self):
        p = Project.objects.create(title='a')

        def fake(kind, pk, payload, rep):
            rep.stage('render', 'working')
            for _ in range(200):
                rep.check()
                time.sleep(0.05)

        with mock.patch.object(pipeline, 'run_job', fake):
            jobs.enqueue('full', p.pk)
            self.assertTrue(wait_for(lambda: Project.objects.get(pk=p.pk).stage == 'render'))
            self.assertTrue(jobs.request_cancel(p.pk))
            self.assertTrue(wait_for(lambda: Project.objects.get(pk=p.pk).status == 'cancelled'))
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk)))

    def test_cancel_queued_job_never_runs(self):
        a, b = Project.objects.create(title='a'), Project.objects.create(title='b')
        gate, ran = threading.Event(), []

        def fake(kind, pk, payload, rep):
            ran.append(pk)
            gate.wait(5)

        with mock.patch.object(pipeline, 'run_job', fake):
            jobs.enqueue('full', a.pk)
            self.assertTrue(wait_for(lambda: a.pk in ran))
            jobs.enqueue('full', b.pk)
            jobs.request_cancel(b.pk)
            self.assertEqual(Project.objects.get(pk=b.pk).status, 'cancelled')
            gate.set()
            self.assertTrue(wait_for(lambda: not jobs.is_active(b.pk) and not jobs.is_active(a.pk)))
        self.assertNotIn(b.pk, ran)

    def test_failure_is_recorded_not_stuck(self):
        p = Project.objects.create(title='a')

        def boom(*a, **k):
            raise RuntimeError('ffmpeg exploded')

        with mock.patch.object(pipeline, 'run_job', boom):
            jobs.enqueue('full', p.pk)
            self.assertTrue(wait_for(lambda: Project.objects.get(pk=p.pk).status == 'error'))
        self.assertIn('ffmpeg exploded', Project.objects.get(pk=p.pk).error)

    def test_orphaned_projects_recovered(self):
        a = Project.objects.create(title='a', status='processing')
        b = Project.objects.create(title='b', status='fetching')
        c = Project.objects.create(title='c', status='done')
        Clip.objects.create(project=a, start=0, end=5, status='rendering')
        jobs._recovered = False
        self.assertEqual(jobs.recover_once(), 2)
        self.assertEqual(Project.objects.get(pk=a.pk).status, 'interrupted')
        self.assertEqual(Project.objects.get(pk=b.pk).status, 'interrupted')
        self.assertEqual(Project.objects.get(pk=c.pk).status, 'done')
        self.assertEqual(Clip.objects.get(project=a).status, 'candidate')

    def test_reap_marks_dead_jobs(self):
        p = Project.objects.create(title='a', status='processing')
        self.assertTrue(jobs.reap(p.pk))
        self.assertEqual(Project.objects.get(pk=p.pk).status, 'interrupted')

    def test_real_progress_never_claims_more_than_known(self):
        p = Project.objects.create(title='a')
        rep = jobs.Reporter(p.pk)
        rep.stage('transcribe', 'x')
        self.assertIsNone(Project.objects.get(pk=p.pk).stage_progress)     # unknown stays unknown
        rep.frac(0.5, 'half')
        self.assertEqual(Project.objects.get(pk=p.pk).stage_progress, 0.5)


class FfmpegCancel(TestCase):
    def test_cancel_kills_ffmpeg_quickly(self):
        flag = {'stop': False}
        threading.Timer(1.0, lambda: flag.update(stop=True)).start()
        t0 = time.time()
        with self.assertRaises(JobCancelled):
            run_ffmpeg(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=1280x720:r=30',
                        '-t', '600', '-c:v', 'libx264', '-preset', 'ultrafast', '-f', 'null', '-'], duration=600,
                       cancel_check=lambda: flag['stop'])
        self.assertLess(time.time() - t0, 8)

    def test_failure_reports_stderr_tail(self):
        with self.assertRaises(RuntimeError) as cm:
            run_ffmpeg(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', '/nonexistent/file.mp4', '-f', 'null', '-'])
        self.assertIn('nonexistent', str(cm.exception))


class Migration(TransactionTestCase):
    def test_v4_rows_survive_upgrade(self):
        ex = MigrationExecutor(connection)
        ex.migrate([('studio', '0002_edit_modes')])
        old = ex.loader.project_state([('studio', '0002_edit_modes')]).apps
        P, C = old.get_model('studio', 'Project'), old.get_model('studio', 'Clip')
        p = P.objects.create(title='legacy', status='done', edit_mode='gameplay_facecam')
        C.objects.create(project_id=p.pk, start=1, end=30, score=61, title='old', video='renders/x.mp4')
        ex = MigrationExecutor(connection)
        ex.migrate(ex.loader.graph.leaf_nodes())
        self.assertEqual(Project.objects.get(pk=p.pk).edit_mode, 'gameplay_facecam')
        c = Clip.objects.get(project_id=p.pk)
        self.assertEqual((c.status, c.video.name, c.scores), ('ready', 'renders/x.mp4', {}))


class Views(TestCase):
    def setUp(self):
        self.c = Client()
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)

    def _project_with_clip(self):
        p = Project.objects.create(title='t', status='done', stage='')
        p.source.save('src.mp4', ContentFile(b'0' * 100), save=True)
        c = Clip.objects.create(project=p, start=1, end=20, score=70, title='Clip', status='ready')
        c.video.save('c.mp4', ContentFile(bytes(range(256)) * 40), save=True)
        return p, c

    def test_index_and_pages_render(self):
        self.assertEqual(self.c.get('/').status_code, 200)
        self.assertEqual(self.c.get('/history/').status_code, 200)
        p, _ = self._project_with_clip()
        self.assertEqual(self.c.get(f'/projects/{p.pk}/').status_code, 200)
        with mock.patch('studio.diagnostics.full_report', return_value={
                'python': '3', 'platform': 'x', 'ffmpeg': {'ffmpeg': True, 'ffprobe': True, 'libass': True, 'version': 'v'},
                'nvenc': {'ok': False, 'detail': 'no'}, 'cuda': {'count': 0}, 'gpu': {}, 'libs': {'cv2': '4'},
                'whisper': {'device': 'cpu', 'compute': 'int8', 'note': 'n', 'model': 'small'},
                'encoder': {'name': 'libx264', 'note': ''}, 'disk_free_gb': 10, 'max_jobs': 1, 'ollama': False}):
            self.assertEqual(self.c.get('/system/').status_code, 200)

    def test_create_project_validates_and_enqueues(self):
        with mock.patch('studio.jobs.enqueue') as enq:
            r = self.c.post('/', {'youtube_url': 'https://evil.com/x', 'rights': 'on', 'auto_fetch': 'on'})
            self.assertEqual(r.status_code, 200)
            self.assertIn('error', r.context)
            r = self.c.post('/', {'youtube_url': 'https://youtu.be/dQw4w9WgXcQ', 'auto_fetch': 'on'})
            self.assertIn('право', r.context['error'])
            self.assertFalse(enq.called)
            r = self.c.post('/', {'youtube_url': 'https://youtu.be/dQw4w9WgXcQ', 'auto_fetch': 'on', 'rights': 'on',
                                  'edit_mode': 'director', 'subtitle_style': 'gaming', 'preset': '720p', 'present_review_first': '1',
                                  'review_first': 'on', 'present_captions': '1', 'clip_length': 'long', 'evil_field': 'x'})
            self.assertEqual(r.status_code, 302)
            self.assertTrue(enq.called)
        p = Project.objects.latest('pk')
        self.assertEqual((p.edit_mode, p.clip_min, p.clip_max), ('director', 45, 90))
        self.assertEqual((p.options['subtitle_style'], p.options['preset'], p.options['review_first'], p.options['captions']),
                         ('gaming', '720p', True, False))
        self.assertNotIn('evil_field', p.options)

    def test_status_shape_is_backward_compatible(self):
        p, c = self._project_with_clip()
        j = self.c.get(f'/api/projects/{p.pk}/status/').json()
        for k in ('status', 'progress', 'message', 'error'):
            self.assertIn(k, j)
        clip = j['clips'][0]
        for k in ('id', 'title', 'start', 'end', 'score', 'url'):      # v4 keys
            self.assertIn(k, clip)
        self.assertEqual(len(j['stages']), 8)
        self.assertTrue(all(s['state'] == 'done' for s in j['stages']))

    def test_render_endpoint_validates(self):
        p, c = self._project_with_clip()
        post = lambda body: self.c.post(f'/api/projects/{p.pk}/render/', json.dumps(body), content_type='application/json')
        self.assertEqual(post({'clips': []}).status_code, 400)
        self.assertEqual(post({'clips': [{'id': 99999}]}).status_code, 400)
        with mock.patch('studio.jobs.enqueue', return_value=True) as enq:
            r = post({'clips': [{'id': c.pk, 'start': 5, 'end': 4}], 'mode': 'rm -rf', 'overrides': {'preset': '720p', 'hack': 1}})
            self.assertEqual(r.status_code, 200)
            kw = enq.call_args.kwargs
            self.assertEqual(kw['bounds'], {})                      # invalid bounds ignored
            self.assertNotIn('mode', kw)                            # invalid mode ignored
            self.assertEqual(kw['overrides'], {'preset': '720p'})   # unknown option dropped
        with mock.patch('studio.jobs.enqueue', return_value=False):
            self.assertEqual(post({'clips': [{'id': c.pk}]}).status_code, 409)

    def test_cancel_and_resume_endpoints(self):
        p = Project.objects.create(title='x', status='error')
        with mock.patch('studio.jobs.enqueue', return_value=True) as enq:
            self.assertEqual(self.c.post(f'/api/projects/{p.pk}/resume/', '{}', content_type='application/json').status_code, 200)
            self.assertEqual(enq.call_args.args[0], 'resume')
        p.status = 'processing'
        p.save()
        self.assertEqual(self.c.post(f'/api/projects/{p.pk}/resume/', '{}', content_type='application/json').status_code, 409)
        self.assertEqual(self.c.post(f'/api/projects/{p.pk}/cancel/').json(), {'ok': False})   # nothing live in this process

    def test_history_filters(self):
        Project.objects.create(title='alpha game', status='done', edit_mode='gameplay')
        Project.objects.create(title='beta talk', status='error', edit_mode='smart')
        r = self.c.get('/history/', {'q': 'alpha'})
        self.assertEqual([i['p'].title for i in r.context['items']], ['alpha game'])
        r = self.c.get('/history/', {'status': 'error'})
        self.assertEqual([i['p'].title for i in r.context['items']], ['beta talk'])
        r = self.c.get('/history/', {'mode': 'gameplay'})
        self.assertEqual(len(r.context['items']), 1)

    def test_media_range_requests(self):
        _, c = self._project_with_clip()
        url = c.video.url
        full = self.c.get(url)
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full['Accept-Ranges'], 'bytes')
        data = b''.join(full.streaming_content)
        part = self.c.get(url, HTTP_RANGE='bytes=10-19')
        self.assertEqual(part.status_code, 206)
        self.assertEqual(part['Content-Range'], f'bytes 10-19/{len(data)}')
        self.assertEqual(b''.join(part.streaming_content), data[10:20])
        self.assertEqual(b''.join(self.c.get(url, HTTP_RANGE='bytes=-5').streaming_content), data[-5:])
        self.assertEqual(self.c.get(url, HTTP_RANGE=f'bytes={len(data) + 5}-').status_code, 416)
        self.assertIn('attachment', self.c.get(url + '?download=1')['Content-Disposition'])

    def test_media_path_traversal_blocked(self):
        secret = Path(self.media).parent / 'secret.txt'
        secret.write_text('top secret')
        self.addCleanup(secret.unlink)
        for bad in ('../secret.txt', '..%2Fsecret.txt', 'renders/../../secret.txt', '/etc/passwd'):
            self.assertEqual(self.c.get('/media/' + bad).status_code, 404, bad)
