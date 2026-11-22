import os
import sys
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

from django.core.files.base import ContentFile
from django.test import SimpleTestCase, TestCase, override_settings

from studio import apps, diagnostics, pipeline
from studio.models import Project
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video


@skipUnless(HAVE_FFMPEG and os.getenv('CLIPFORGE_SKIP_SLOW') != '1', 'needs ffmpeg')
class LegacyRenderersStillWork(TestCase):
    """The proven v4 render() must keep working untouched: it is the safety net for the new engine."""

    def test_all_four_v4_modes(self):
        import json
        import subprocess
        src = sample_video()
        clip = {'start': 0.0, 'end': 6.0}
        for mode in ('center', 'smart', 'gameplay', 'gameplay_facecam'):
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                dest = td / 'o.mp4'
                pipeline.render(src, clip, fake_transcript(), dest, td, mode, 'auto')
                d = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,duration',
                                                        '-of', 'json', str(dest)]))
                v = d['streams'][0]
                self.assertEqual((v['width'], v['height']), (1080, 1920), mode)


class FakeSeg:
    def __init__(self, s, e, t):
        self.start, self.end, self.text, self.words = s, e, t, []


class TranscribeDeviceFallback(SimpleTestCase):
    def setUp(self):
        diagnostics._whisper_gpu_failed = False
        pipeline._model_cache.clear()
        self.addCleanup(setattr, diagnostics, '_whisper_gpu_failed', False)

    def test_cuda_failure_restarts_on_cpu_and_is_remembered(self):
        calls = []

        class Model:
            def __init__(self, device):
                self.device = device

            def transcribe(self, *a, **k):
                calls.append(self.device)

                def gen():
                    if self.device == 'cuda':
                        raise RuntimeError('Library cublas64_12.dll is not found or cannot be loaded')
                    yield FakeSeg(0, 2, 'привет мир')
                return gen(), mock.Mock(language='ru')

        with mock.patch.object(diagnostics, 'whisper_settings', side_effect=lambda: ('cuda', 'float16', 'x')
                               if not diagnostics._whisper_gpu_failed else ('cpu', 'int8', 'x')), \
                mock.patch.object(pipeline, '_get_model', side_effect=lambda n, d, c: Model(d)):
            out = pipeline.transcribe('x.mp4', 'auto')
        self.assertEqual(calls, ['cuda', 'cpu'])
        self.assertEqual(out[0]['text'], 'привет мир')
        self.assertTrue(diagnostics._whisper_gpu_failed)
        self.assertEqual(pipeline.transcribe.last_info['device'], 'cpu')
        self.assertEqual(diagnostics.whisper_settings()[0], 'cpu')       # no more CUDA attempts this session

    def test_cpu_error_is_not_swallowed(self):
        class Model:
            def transcribe(self, *a, **k):
                raise RuntimeError('corrupt file')
        with mock.patch.object(diagnostics, 'whisper_settings', return_value=('cpu', 'int8', 'x')), \
                mock.patch.object(pipeline, '_get_model', return_value=Model()):
            with self.assertRaises(RuntimeError):
                pipeline.transcribe('x.mp4', 'auto')

    def test_explicit_env_override(self):
        with mock.patch.dict(os.environ, {'WHISPER_DEVICE': 'cpu'}):
            self.assertEqual(diagnostics.whisper_settings()[:2], ('cpu', 'int8'))
        with mock.patch.dict(os.environ, {'WHISPER_DEVICE': 'cuda', 'WHISPER_COMPUTE': 'int8_float16'}):
            self.assertEqual(diagnostics.whisper_settings()[:2], ('cuda', 'int8_float16'))


class AutoreloadGuard(SimpleTestCase):
    def test_only_the_serving_child_starts_workers(self):
        with mock.patch.object(sys, 'argv', ['manage.py', 'runserver']):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('RUN_MAIN', None)
                self.assertFalse(apps.is_serving_process(), 'autoreload parent must stay passive')
            with mock.patch.dict(os.environ, {'RUN_MAIN': 'true'}):
                self.assertTrue(apps.is_serving_process())
        with mock.patch.object(sys, 'argv', ['manage.py', 'runserver', '--noreload']):
            self.assertTrue(apps.is_serving_process())
        for cmd in ('migrate', 'test', 'shell', 'check'):
            with mock.patch.object(sys, 'argv', ['manage.py', cmd]):
                self.assertFalse(apps.is_serving_process(), cmd)


class DownloadReuse(TestCase):
    def test_same_video_is_not_downloaded_twice(self):
        from studio import youtube_download as yd
        url = 'https://youtu.be/dQw4w9WgXcQ'
        with tempfile.TemporaryDirectory() as media, override_settings(MEDIA_ROOT=media):
            a = Project.objects.create(title='a', youtube_url=url)
            a.source.save('dQw4w9WgXcQ.mp4', ContentFile(b'x' * 50), save=True)
            b = Project.objects.create(title='b', youtube_url=url)
            with mock.patch.object(yd, '_download', side_effect=AssertionError('must not download again')):
                yd.fetch_source(b.pk)
            b.refresh_from_db()
            self.assertEqual(b.source.name, a.source.name)
