"""Source videos: never accept a file without a video stream; long streams fit the size limit."""
import subprocess
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

from django.conf import settings
from django.core.files import File
from django.test import SimpleTestCase, TestCase, override_settings

from studio import pipeline, youtube_download as yd
from studio.models import Project
from studio.procutil import describe_streams, has_video
from .helpers import HAVE_FFMPEG


def _audio_only(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=d=6', '-c:a', 'aac', str(path)], check=True)
    return path


def _video(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:d=6', '-f', 'lavfi',
                    '-i', 'sine=d=6', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', str(path)],
                   check=True)
    return path


class FormatSelection(SimpleTestCase):
    """The yt-dlp format string, evaluated by yt-dlp's own selector, can never pick an audio-only format."""

    def _pick(self, formats):
        import yt_dlp
        ydl = yt_dlp.YoutubeDL({'quiet': True, 'format': yd.FORMAT})
        sel = ydl.build_format_selector(yd.FORMAT)
        ctx = {'formats': formats, 'incomplete_formats': False, 'has_merged_format': False}
        return list(sel(ctx))

    def _fmt(self, fid, vcodec='none', acodec='none', height=None, ext='mp4'):
        return {'format_id': fid, 'vcodec': vcodec, 'acodec': acodec, 'height': height, 'ext': ext,
                'url': 'https://x/' + fid, 'protocol': 'https', 'tbr': (height or 1) * 10, 'video_ext': ext if vcodec != 'none' else 'none',
                'audio_ext': ext if acodec != 'none' else 'none'}

    def test_merges_1080_video_with_audio(self):
        picked = self._pick([self._fmt('140', acodec='mp4a', ext='m4a'),
                             self._fmt('137', vcodec='avc1', height=1080), self._fmt('299', vcodec='avc1', height=1440)])
        self.assertTrue(picked)
        ids = picked[0].get('format_id')
        self.assertEqual(ids, '137+140')

    def test_audio_only_list_selects_nothing(self):
        picked = self._pick([self._fmt('140', acodec='mp4a', ext='m4a'), self._fmt('251', acodec='opus', ext='webm')])
        self.assertEqual(picked, [], 'an audio-only format must never be chosen')

    def test_limits_follow_settings(self):
        self.assertGreaterEqual(yd.MAX_BYTES, 4 * 1024 ** 3)
        self.assertGreaterEqual(yd.DOWNLOAD_TIMEOUT, 3600)


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class StreamChecks(SimpleTestCase):
    def test_has_video(self):
        td = Path(tempfile.mkdtemp())
        self.assertFalse(has_video(_audio_only(td / 'a.mp4')))
        self.assertTrue(has_video(_video(td / 'v.mp4')))
        self.assertIn('audio:aac', describe_streams(td / 'a.mp4'))

    def test_probe_explains(self):
        td = Path(tempfile.mkdtemp())
        with self.assertRaises(ValueError) as cm:
            pipeline.probe(_audio_only(td / 'a.mp4'))
        self.assertIn('только audio', str(cm.exception))
        self.assertIn('Продолжить', str(cm.exception))


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class BrokenSourceRecovery(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media, BASE_DIR=Path(self.media))
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.td = Path(tempfile.mkdtemp())

    def _project(self, url='https://www.youtube.com/watch?v=dQw4w9WgXcQ'):
        p = Project.objects.create(title='t', youtube_url=url)
        with open(_audio_only(self.td / 'old.mp4'), 'rb') as f:
            p.source.save('dQw4w9WgXcQ.mp4', File(f), save=True)
        return p

    def test_audio_only_youtube_source_is_downloaded_again_and_old_file_kept(self):
        p = self._project()
        old_path = Path(p.source.path)
        good = _video(self.td / 'good.mp4')

        def fake_fetch(pk, rep=None):
            with open(good, 'rb') as f:
                Project.objects.get(pk=pk).source.save('dQw4w9WgXcQ.mp4', File(f), save=True)
        rep = mock.Mock()
        with mock.patch.object(yd, 'fetch_source', side_effect=fake_fetch) as fetch:
            path = pipeline._ensure_source(p, rep)
        fetch.assert_called_once()
        self.assertTrue(has_video(path))
        self.assertTrue(old_path.is_file(), 'the old file is never deleted')

    def test_uploaded_audio_only_file_gives_clear_error(self):
        p = self._project(url='')
        with self.assertRaises(RuntimeError) as cm:
            pipeline._ensure_source(p, mock.Mock())
        self.assertIn('нет видеодорожки', str(cm.exception))

    def test_download_without_video_is_rejected_with_hint_and_log(self):
        p = Project.objects.create(title='t', youtube_url='https://www.youtube.com/watch?v=dQw4w9WgXcQ')

        def fake_download(pk, video_id, destination, rep=None):
            _audio_only(destination / f'{video_id}.mp4')
            (destination / 'yt_dlp.log').write_text('WARNING: No supported JavaScript runtime could be found')
            yd._keep_log(pk, destination / 'yt_dlp.log')
        with mock.patch.object(yd, '_download', side_effect=fake_download), \
                mock.patch('shutil.which', return_value=None):
            with self.assertRaises(RuntimeError) as cm:
                yd.fetch_source(p.pk)
        msg = str(cm.exception)
        self.assertIn('без видеодорожки', msg)
        self.assertIn('deno', msg)
        self.assertIn(f'ytdlp-{p.pk}.log', msg)
        self.assertTrue((Path(self.media) / 'logs' / f'ytdlp-{p.pk}.log').is_file())
        p.refresh_from_db()
        self.assertFalse(p.source)

    def test_broken_source_of_another_project_is_not_reused(self):
        other = self._project()
        p = Project.objects.create(title='t2', youtube_url=other.youtube_url)
        with mock.patch.object(yd, '_download', side_effect=RuntimeError('download attempted')):
            with self.assertRaises(RuntimeError) as cm:
                yd.fetch_source(p.pk)
        self.assertIn('download attempted', str(cm.exception))
