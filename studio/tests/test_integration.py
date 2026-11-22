"""End-to-end tests with real ffmpeg on a synthetic video (game+webcam -> talking head -> gameplay -> screen)."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock, skipUnless

from django.core.files import File
from django.test import TestCase, TransactionTestCase, override_settings

from studio import diagnostics, editor, exports, jobs, pipeline, quality, scenes
from studio.models import Clip, Project
from studio.procutil import JobCancelled
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'


def ffprobe(path):
    d = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height,duration',
                                            '-of', 'json', str(path)]))
    v = next(s for s in d['streams'] if s['codec_type'] == 'video')
    a = next((s for s in d['streams'] if s['codec_type'] == 'audio'), None)
    return int(v['width']), int(v['height']), float(v['duration']), (float(a['duration']) if a else None)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class SceneDirector(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.src = sample_video()
        cls.vinfo = editor.probe_video(cls.src)

    def test_detects_four_scene_types_in_order(self):
        a = scenes.analyze(str(self.src), 0, 16, 1920, 1080)
        self.assertIsNotNone(a['cam'])
        self.assertEqual(a['cam']['corner'], 'top_right')
        self.assertAlmostEqual(a['cam']['x'], 0.781, delta=0.03)
        self.assertAlmostEqual(a['cam']['w'], 0.219, delta=0.04)
        plan = scenes.plan_scenes(a, 16.0)
        self.assertEqual([s['type'] for s in plan], ['game_cam', 'talking', 'gameplay', 'screen'])
        for s, expect in zip(plan, (4, 8, 12, 16)):
            self.assertAlmostEqual(s['end'], expect, delta=1.2)
            self.assertFalse(s['fallback'])
        self.assertTrue(all(s['end'] - s['start'] >= scenes.MIN_SEG - 0.1 for s in plan))   # no flicker

    def _render(self, mode, **opt):
        opts = exports.resolve({'subtitle_style': 'bold', 'quality': 'small', 'preset': '720p', **opt})
        clip = {'start': 0.0, 'end': 16.0}
        td = Path(tempfile.mkdtemp())
        plan = editor.prepare(str(self.src), clip, mode, 'auto', opts, self.vinfo)
        dest = td / 'o.mp4'
        info = editor.render(str(self.src), clip, fake_transcript(), dest, td, opts, plan, self.vinfo, mode)
        return dest, info, plan

    def test_every_mode_produces_correct_vertical_video(self):
        for mode in exports.MODE_KEYS:
            dest, info, _ = self._render(mode)
            w, h, vd, ad = ffprobe(dest)
            self.assertEqual((w, h), (720, 1280), mode)
            self.assertAlmostEqual(vd, 16.0, delta=0.15, msg=mode)
            self.assertIsNotNone(ad, mode)
            self.assertAlmostEqual(ad, vd, delta=0.3, msg=f'A/V sync {mode}')

    def test_director_output_passes_quality_validation(self):
        dest, info, _ = self._render('director')
        q = quality.validate_clip(dest, 16.0, source_has_audio=True, caption_report=info['captions'], text='x',
                                  siblings=[], clip_min=5, clip_max=30)
        bad = [c for c in q['checks'] if c['status'] == 'fail']
        self.assertEqual(bad, [], q)
        self.assertGreaterEqual(q['score'], 80)

    def test_gameplay_facecam_has_no_black_bars_or_duplicate_face(self):
        dest, info, plan = self._render('gameplay_facecam', preset='1080p')
        err = quality.run_detectors(dest)
        black, _ = quality.parse_detect(err, 16)
        self.assertEqual(black, [], 'black frames/bars in split layout')
        self.assertEqual(info['cam']['source'], 'detected')
        # the single mask over the overlay must hide the small face in the game panel: only ONE large face should remain
        td = Path(tempfile.mkdtemp())
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(dest), '-frames:v', '1', str(td / 'f.png')], check=True)
        import cv2
        im = cv2.imread(str(td / 'f.png'))
        top = cv2.cvtColor(im[:int(im.shape[0] * 0.66)], cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(scenes.detect_faces_gray(top)), 0, 'face is still visible in the gameplay panel')
        bottom = cv2.cvtColor(im[int(im.shape[0] * 0.66):], cv2.COLOR_BGR2GRAY)
        self.assertGreaterEqual(len(scenes.detect_faces_gray(bottom)), 1, 'facecam face lost')

    def test_captions_inside_safe_zone_for_every_layout(self):
        _, info, plan = self._render('director')
        H = 1280
        for typ in {s['type'] for s in plan['segments']}:
            _, mv = __import__('studio.layouts', fromlist=['x']).seg_layout(
                {'type': typ}, {'W': 720, 'H': H, 'src_w': 1920, 'src_h': 1080, 'bg': 'blur', 'cam': plan['cam']}, 'a', 'b')
            self.assertGreaterEqual(mv, 0.2 * H - 1)

    def test_nvenc_failure_falls_back_to_cpu(self):
        if diagnostics.nvenc_works()['ok']:
            self.skipTest('real NVENC works here')
        with mock.patch.object(diagnostics, 'choose_encoder', return_value=('h264_nvenc', 'forced')):
            dest, info, _ = self._render('gameplay')
        self.assertEqual(info['encoder'], 'libx264')
        self.assertTrue(info.get('encoder_fallback'))
        self.assertTrue(dest.exists() and dest.stat().st_size > 1000)

    def test_cancel_during_scene_analysis(self):
        calls = {'n': 0}

        def cancel():
            calls['n'] += 1
            return calls['n'] > 3
        with self.assertRaises(JobCancelled):
            scenes.analyze(str(self.src), 0, 16, 1920, 1080, cancel=cancel)

    def test_analysis_failure_degrades_to_safe_layout(self):
        opts = exports.resolve({})
        with mock.patch.object(scenes, 'analyze', side_effect=RuntimeError('decoder crashed')):
            plan = editor.prepare(str(self.src), {'start': 0, 'end': 16}, 'director', 'auto', opts, self.vinfo)
        self.assertEqual(plan['segments'][0]['type'], 'gameplay')
        self.assertTrue(plan['segments'][0]['fallback'])


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class FullPipeline(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def make_project(self, **kw):
        p = Project(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode=kw.pop('mode', 'gameplay'),
                    options=kw.pop('options', {'preset': '720p', 'quality': 'small', 'subtitle_style': 'clean'}), **kw)
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        return p

    def run_job(self, kind, pk, **payload):
        self.assertTrue(jobs.enqueue(kind, pk, **payload))
        self.assertTrue(wait_for(lambda: not jobs.is_active(pk), timeout=240))

    def test_full_flow_rerender_resume_without_duplicates_or_retranscription(self):
        p = self.make_project()
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()) as tr:
            self.run_job('full', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'done', p.error)
            self.assertEqual(tr.call_count, 1)
            clips = list(p.clips.order_by('position'))
            self.assertGreaterEqual(len(clips), 1)
            for c in clips:
                self.assertEqual(c.status, 'ready')
                self.assertTrue(Path(c.video.path).is_file())
                self.assertTrue(Path(c.poster.path).is_file())
                self.assertIsNotNone(c.quality.get('score'))
                self.assertTrue(c.meta['titles'])
                self.assertTrue(c.reason)
                self.assertEqual(ffprobe(c.video.path)[:2], (720, 1280))
                self.assertTrue(5 <= c.end - c.start <= 10)
            first = clips[0]
            old_file, old_n = first.video.path, Clip.objects.count()

            # re-render ONE clip with other settings + new bounds -> same row, old file replaced, no new transcription
            self.run_job('render', p.pk, clip_ids=[first.pk], bounds={str(first.pk): [1.0, 8.0]},
                         overrides={'subtitle_style': 'karaoke'}, mode='smart')
            first.refresh_from_db()
            self.assertEqual(Clip.objects.count(), old_n)
            self.assertEqual((first.start, first.end), (1.0, 8.0))
            self.assertEqual(first.render_info['mode'], 'smart')
            self.assertEqual(first.render_info['captions']['style'], 'karaoke')
            self.assertFalse(Path(old_file).exists(), 'replaced render must be deleted')
            self.assertTrue(Path(first.video.path).is_file())

            # reprocess the whole project: clips replaced (not duplicated), transcript + source reused
            src_path = p.source.path
            self.run_job('reprocess', p.pk)
            self.assertEqual(Clip.objects.filter(project=p).count(), len(clips))
            self.assertEqual(tr.call_count, 1, 'must not transcribe again')
            self.assertTrue(Path(src_path).is_file(), 'source must never be deleted')

    def test_review_flow_candidates_then_render_selected(self):
        p = self.make_project(options={'preset': '720p', 'quality': 'small', 'review_first': True})
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()):
            self.run_job('full', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'review')
            cands = list(p.clips.all())
            self.assertTrue(cands and all(c.status == 'candidate' and not c.video for c in cands))
            self.run_job('render', p.pk, clip_ids=[cands[0].pk], bounds={}, discard_others=True)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        self.assertEqual(p.clips.count(), 1)
        self.assertEqual(p.clips.get().status, 'ready')

    def test_cancel_is_safe_and_resume_reuses_transcript(self):
        p = self.make_project()
        started = {}
        real_render = editor.render

        def slow_render(*a, **k):
            started['t'] = True
            for _ in range(400):
                k['rep'].check()
                time.sleep(0.05)
            return real_render(*a, **k)

        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()) as tr:
            with mock.patch.object(editor, 'render', slow_render):
                jobs.enqueue('full', p.pk)
                self.assertTrue(wait_for(lambda: started.get('t'), timeout=120))
                jobs.request_cancel(p.pk)
                self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=30))
            p.refresh_from_db()
            self.assertEqual(p.status, 'cancelled')
            self.assertFalse(Clip.objects.filter(project=p, status__in=('queued', 'rendering')).exists())
            self.assertTrue(Path(p.source.path).is_file())
            self.run_job('resume', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'done', p.error)
            self.assertEqual(tr.call_count, 1)
        self.assertEqual(Clip.objects.filter(project=p, status='ready').count(), p.clips.count())

    def test_new_renderer_failure_falls_back_to_v4_renderer(self):
        p = self.make_project(mode='center')
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()), \
                mock.patch.object(editor, 'render', side_effect=RuntimeError('filtergraph broke')):
            self.run_job('full', p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        c = p.clips.first()
        self.assertEqual(c.render_info['fallback_renderer'], 'v4')
        self.assertEqual(ffprobe(c.video.path)[:2], (1080, 1920))

    def test_missing_speech_gives_clear_error(self):
        p = self.make_project()
        with mock.patch.object(pipeline, 'transcribe', return_value=[]):
            self.run_job('full', p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, 'error')
        self.assertIn('Речь не найдена', p.error)

    def test_cross_project_transcript_cache(self):
        a = self.make_project()
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()) as tr:
            self.run_job('full', a.pk)
            b = self.make_project()
            self.run_job('full', b.pk)
            self.assertEqual(tr.call_count, 1, 'same video -> transcript reused across projects')
        b.refresh_from_db()
        self.assertEqual(b.status, 'done', b.error)
