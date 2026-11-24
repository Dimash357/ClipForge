"""2D Clip Selection Studio: state validation, original/undo, draft vertical preview (no full render), one job
for bulk apply, restore, ZIP export with texts, the page and the status fields."""
import io
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from unittest import mock, skipUnless

from django.conf import settings
from django.core.files import File
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import clipstudio, exports, jobs, pipeline
from studio.models import Clip, Job, Project
from .helpers import HAVE_FFMPEG, sample_video
from .test_jobs_views import wait_for
from .test_smartedit import paused_transcript


class FakeClip:
    def __init__(self, **kw):
        self.start, self.end = kw.get('start', 2.0), kw.get('end', 12.0)
        self.meta = kw.get('meta', {})
        self.render_info = kw.get('render_info', {'mode': 'center', 'opts': {}})
        self.scores = kw.get('scores', {'auto_bounds': [1.0, 11.0]})


class State(SimpleTestCase):
    def test_clean_keeps_only_valid_changes(self):
        c = FakeClip(meta={'hooks': {'variants': [{'id': 'orig'}, {'id': 'teaser-1'}], 'selected': None}})
        st = clipstudio.clean({'start': 3, 'end': 14, 'mode': 'smart', 'hook': 'teaser-1',
                               'edit': {'mode': 'dynamic', 'restore': ['pause-1']},
                               'overrides': {'subtitle_style': 'bold', 'evil': 1, 'caption_words': 99}}, c)
        self.assertEqual((st['start'], st['end'], st['mode'], st['hook']), (3.0, 14.0, 'smart', 'teaser-1'))
        self.assertEqual(st['edit'], {'mode': 'dynamic', 'restore': ['pause-1']})
        self.assertEqual(st['overrides'], {'subtitle_style': 'bold', 'caption_words': 6})       # clamped, no junk
        bad = clipstudio.clean({'start': 10, 'end': 5, 'mode': 'nope', 'hook': 'missing', 'edit': {'mode': 'wild'}}, c)
        self.assertEqual((bad['start'], bad['end'], bad['mode'], bad['hook']), (2.0, 12.0, 'center', None))
        self.assertEqual(bad['edit']['mode'], None)

    def test_original_comes_from_the_found_bounds(self):
        o = clipstudio.original_state(FakeClip(render_info={'mode': 'smart', 'opts': {'subtitle_style': 'bold'}}))
        self.assertEqual((o['start'], o['end'], o['mode'], o['overrides'], o['hook']), (1.0, 11.0, None, {}, None))
        kept = {'start': 5.0, 'end': 9.0}
        self.assertEqual(clipstudio.original_state(FakeClip(meta={'original': kept})), kept)

    def test_texts_for_publishing(self):
        c = FakeClip(meta={'short': {'title': 'Т', 'description': 'Д', 'hashtags': ['игры', '#стрим']}})
        c.title = 'x'
        self.assertEqual(clipstudio.texts(c), ('Т', 'Д', '#игры #стрим'))
        self.assertEqual(clipstudio._slug('a/b:c*?"<>|d'), 'a b c d')


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height:format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True).stdout.split()
    w, h = out[0].split(',')[:2]
    return int(w), int(h), float(out[-1])


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class Studio(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)
        p = Project.objects.create(title='Стрим', clip_count=2, clip_min=6, clip_max=10, edit_mode='center',
                                   options=exports.resolve({'preset': '720p', 'quality': 'small'}))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=paused_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.wait(p)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        self.p = p
        self.c = p.clips.order_by('position').first()

    def wait(self, p):
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))

    def test_page_and_status_fields(self):
        r = self.client.get(f'/projects/{self.p.pk}/studio/')
        self.assertContains(r, 'clipstudio.js')
        self.assertContains(self.client.get(f'/projects/{self.p.pk}/'), 'Clip Studio')
        c = self.client.get(f'/api/projects/{self.p.pk}/status/').json()['clips'][0]
        for k in ('history', 'edited', 'edit_mode', 'output_duration', 'width'):
            self.assertIn(k, c)

    def test_draft_preview_is_vertical_cached_and_changes_nothing(self):
        before = (self.c.video.name, dict(self.c.render_info))
        r = self.client.post(f'/api/clips/{self.c.pk}/preview/', {'edit': {'mode': 'dynamic'},
                                                                   'overrides': {'subtitle_style': 'bold'}},
                             content_type='application/json')
        self.assertEqual(r.status_code, 200, r.content)
        d = r.json()
        path = Path(settings.MEDIA_ROOT) / d['url'][len(settings.MEDIA_URL):]
        w, h, dur = probe(path)
        self.assertEqual((w, h), (360, 640))
        self.assertAlmostEqual(dur, d['info']['duration'], delta=0.15)
        again = self.client.post(f'/api/clips/{self.c.pk}/preview/', {'edit': {'mode': 'dynamic'},
                                                                       'overrides': {'subtitle_style': 'bold'}},
                                 content_type='application/json').json()
        self.assertTrue(again['info']['cached'])
        self.c.refresh_from_db()
        self.assertEqual((self.c.video.name, self.c.render_info), before)
        self.assertFalse(Job.objects.filter(project=self.p, kind='render').exists())     # no job, no full render

    def test_apply_undo_and_original(self):
        c = self.c
        orig = (c.start, c.end)
        r = self.client.post(f'/api/projects/{self.p.pk}/apply/', {'clips': [
            {'id': c.pk, 'start': c.start + 0.5, 'end': c.end, 'overrides': {'subtitle_style': 'bold'},
             'edit': {'mode': 'natural', 'restore': []}}]}, content_type='application/json')
        self.assertEqual(r.status_code, 200, r.content)
        self.wait(self.p)
        c.refresh_from_db()
        self.assertEqual(c.status, 'ready', c.error)
        self.assertAlmostEqual(c.start, orig[0] + 0.5, delta=0.01)
        self.assertEqual(c.render_info['opts'], {'subtitle_style': 'bold'})
        self.assertEqual(c.meta['edit']['mode'], 'natural')
        self.assertEqual(len(c.meta['history']), 1)
        self.assertIn('original', c.meta)
        # undo the last applied change
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/restore/', {'to': 'previous'},
                                          content_type='application/json').status_code, 200)
        self.wait(self.p)
        c.refresh_from_db()
        self.assertAlmostEqual(c.start, orig[0], delta=0.01)
        self.assertEqual(c.render_info['opts'], {})
        self.assertEqual(c.meta['history'], [])
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/restore/', {'to': 'previous'},
                                          content_type='application/json').status_code, 400)
        # back to the original clip
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/restore/', {'to': 'original'},
                                          content_type='application/json').status_code, 200)
        self.wait(self.p)
        c.refresh_from_db()
        ab = c.scores['auto_bounds']
        self.assertEqual((round(c.start, 2), round(c.end, 2)), (round(ab[0], 2), round(ab[1], 2)))
        self.assertNotIn('edit', c.render_info)

    def test_bulk_apply_is_one_job_and_busy_is_refused(self):
        clips = list(self.p.clips.all())
        n0 = Job.objects.filter(project=self.p).count()
        with mock.patch.object(jobs, 'wake'), mock.patch.object(jobs, '_claim', return_value=None):
            r = self.client.post(f'/api/projects/{self.p.pk}/apply/', {'clips': [
                {'id': c.pk, 'overrides': {'caption_censor': True}} for c in clips]}, content_type='application/json')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(Job.objects.filter(project=self.p).count(), n0 + 1)
            job = Job.objects.filter(project=self.p).latest('created_at')
            self.assertEqual(sorted(job.payload['clip_ids']), sorted(c.pk for c in clips))
            self.assertEqual(self.client.post(f'/api/projects/{self.p.pk}/apply/', {'clips': [{'id': clips[0].pk}]},
                                              content_type='application/json').status_code, 409)
            Job.objects.filter(pk=job.pk).update(status=Job.CANCELLED)
        self.assertEqual(self.client.post(f'/api/projects/{self.p.pk}/apply/', {'clips': []},
                                          content_type='application/json').status_code, 400)

    def test_export_zip_with_texts(self):
        c = self.c
        c.meta = dict(c.meta, short={'title': 'Лучший момент', 'description': 'Описание', 'hashtags': ['стрим']})
        c.save()
        ids = list(self.p.clips.values_list('pk', flat=True))
        r = self.client.get(f'/api/projects/{self.p.pk}/export/?ids={",".join(map(str, ids))}')
        self.assertEqual(r.status_code, 200)
        z = zipfile.ZipFile(io.BytesIO(b''.join(r.streaming_content)))
        names = z.namelist()
        self.assertIn('shorts.csv', names)
        self.assertEqual(sum(n.endswith('.mp4') for n in names), len(ids))
        txt = next(n for n in names if 'Лучший момент' in n and n.endswith('.txt'))
        self.assertEqual(z.read(txt).decode('utf-8'), 'Лучший момент\n\nОписание\n\n#стрим\n')
        self.assertIn('Лучший момент', z.read('shorts.csv').decode('utf-8-sig'))
        Clip.objects.filter(project=self.p).update(status='error')
        self.assertEqual(self.client.get(f'/api/projects/{self.p.pk}/export/?ids={ids[0]}').status_code, 400)
