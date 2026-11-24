"""v5.4: YuNet face detector, YAMNet sound events, caption font + swearing filter, batched Whisper."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

import cv2
import numpy as np
from django.test import SimpleTestCase, TestCase, override_settings

from studio import (audio_analysis as aa, captions, editor, exports, faces, layouts, model_files, pipeline, scenes,
                    shorts_meta as sm, sound_events as se)
from studio.models import Clip, Project
from . import synth
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'


def astronaut():
    from skimage import data
    return data.astronaut()[:, :, ::-1].copy()


# =====================================================================================================
# YuNet
# =====================================================================================================
class FaceDetector(SimpleTestCase):
    def setUp(self):
        faces._state.update(backend=None, error=None)
        self.addCleanup(faces._state.update, backend=None, error=None)

    def test_yunet_is_the_default_and_bundled(self):
        self.assertTrue(faces.MODEL.is_file())
        self.assertEqual(faces.backend(), 'yunet')

    def test_finds_faces_that_haar_misses(self):
        img = astronaut()
        hard = {'dark': (img * 0.18).astype(np.uint8), 'small': cv2.resize(img, (90, 90)),
                'tilted': cv2.warpAffine(img, cv2.getRotationMatrix2D((256, 256), 25, 1.0), (512, 512))}
        for name, im in hard.items():
            self.assertEqual(len(scenes.haar_faces(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))), 0, f'haar {name}')
            self.assertEqual(len(faces.detect(im)), 1, f'yunet {name}')

    def test_eye_landmarks(self):
        f = faces.detect(astronaut())[0]
        x, y, w, h = f['box']
        (rx, ry), (lx, ly) = f['eyes']
        self.assertTrue(x < rx < lx < x + w)
        self.assertTrue(y < ry < y + h * 0.6)

    def test_no_false_faces_on_gameplay_or_ui(self):
        self.assertEqual(faces.detect(synth.game_frame(1.0)), [])
        self.assertEqual(faces.detect(synth.screen_frame(1.0)), [])

    def test_haar_can_be_forced_and_is_the_fallback(self):
        with mock.patch.dict(os.environ, {'CLIPFORGE_FACE_DETECTOR': 'haar'}):
            self.assertEqual(faces.backend(), 'haar')
            self.assertEqual(faces.detect(astronaut())[0]['eyes'], None)
        faces._state.update(backend=None)
        with mock.patch.object(faces, 'MODEL', Path('/nonexistent/model.onnx')):
            self.assertEqual(faces.backend(), 'haar')
            self.assertIn('не найдена', faces.status()['error'])
            self.assertEqual(len(faces.detect(astronaut())), 1)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class EyesInReactionLayout(SimpleTestCase):
    def test_webcam_eye_line_is_measured(self):
        a = scenes.analyze(str(sample_video()), 0, 4, 1920, 1080)
        cam = a['cam']
        self.assertIsNotNone(cam)
        self.assertIn('eye_rel', cam)
        self.assertTrue(0.2 < cam['eye_rel'] < 0.8)

    def test_measured_eyes_go_to_the_upper_third(self):
        W, Hc = 1080, 768
        sw, sh, x, y = layouts.reaction_cam_crop(W, Hc, 800, 600, 0.5, 0.5, 0.45, eye_y=0.42)
        self.assertAlmostEqual((0.42 * sh - y) / Hc, 1 / 3, delta=0.02)


# =====================================================================================================
# YAMNet sound events
# =====================================================================================================
def timeline(n=40, **groups):
    base = {g: np.zeros(n, np.float32) for g in se.GROUPS}
    for k, spans in groups.items():
        for a, b, v in spans:
            base[k][a:b] = v
    return se.SoundTimeline(base)


class SoundTimelineRules(SimpleTestCase):
    def test_thresholds_and_merging(self):
        tl = timeline(laughter=[(10, 14, 0.6), (15, 17, 0.5)], cheer=[(30, 32, 0.2)])
        ev = [e for e in tl.events]
        self.assertEqual([e['kind'] for e in ev], ['laughter'])           # 1-frame gap bridged, weak cheer ignored
        self.assertAlmostEqual(ev[0]['start'], 10 * se.HOP, delta=0.01)

    def test_lookalikes_block_false_explosions(self):
        blast = timeline(explosion=[(10, 11, 0.9)])
        self.assertEqual([e['kind'] for e in blast.events], ['explosion'])
        steps = timeline(explosion=[(10, 11, 0.9)], x_explosion=[(13, 18, 0.55)])   # footsteps around it
        self.assertEqual(steps.events, [])
        frog = timeline(laughter=[(5, 8, 0.5)], x_laughter=[(5, 8, 0.9)])
        self.assertEqual(frog.events, [])

    def test_attach_relabels_and_adds_events(self):
        # loudness profile: calm speech, a loud burst at 20 s, a quiet bit at 30 s
        x = np.concatenate([synth.speech_like(20), synth.scream_like(2), synth.speech_like(8),
                            0.3 * synth.laugh_like(2), synth.speech_like(8)])
        p = aa.profile_from_samples(x)
        self.assertTrue(any(e['kind'] in ('scream', 'laughter', 'loud') and 19 <= e['start'] <= 21 for e in p.events))
        n = int(len(x) / aa.SR / se.HOP) + 2
        # classifier: the burst is NOT a scream (e.g. music), the quiet part IS laughter; an explosion in quiet talk
        tl = timeline(n, laughter=[(int(30.2 / se.HOP), int(31.8 / se.HOP), 0.7)], explosion=[(int(10 / se.HOP), int(10 / se.HOP) + 1, 0.9)])
        p.attach_sound(tl)
        burst = [e for e in p.events if 19 <= e['start'] <= 21]
        self.assertTrue(burst and all(e['kind'] not in ('scream', 'laughter') for e in burst), burst)
        self.assertTrue(any(e['kind'] == 'laughter' and 29.5 <= e['start'] <= 31 for e in p.events), p.events)
        self.assertFalse(any(e['kind'] == 'explosion' for e in p.events), 'a quiet "explosion" is not one')

    def test_scores_use_new_kinds(self):
        f = {'top': 0.5, 'mean': 0.3, 'dyn': 5, 'silence': 0.0, 'events': [], 'strong': 1, 'peak_rel': 0.5,
             'kinds': {'explosion': 1, 'gunshot': 2}, 'longest_pause': 0, 'dramatic_pause': False, 'spike': False}
        sc, facts = aa.audio_score(f)
        self.assertIn('explosion', dict(facts))
        self.assertIn('gunshot', dict(facts))


@skipUnless(model_files.is_ready(), 'YAMNet not downloaded (setup.bat downloads it)')
class YamnetModel(SimpleTestCase):
    def test_real_model_sanity(self):
        sr = se.SR
        t = np.arange(sr * 3) / sr
        quiet = se.classify_samples(np.zeros(sr * 3, np.float32))
        self.assertEqual(quiet.events, [])
        tone = se.classify_samples((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        self.assertEqual(tone.events, [], 'a test tone is not laughter, a scream or an explosion')

    @skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
    def test_file_and_memory_agree(self):
        x = np.concatenate([synth.speech_like(20), synth.scream_like(2), synth.speech_like(20)])
        td = Path(tempfile.mkdtemp())
        mem = se.classify_samples(x)
        fil = se.classify_file(synth.make_av(td / 'a.mp4', x), 42)
        n = min(mem.n, fil.n)
        self.assertLessEqual(abs(mem.n - fil.n), 1)
        self.assertLess(float(np.abs(mem.groups['speech'][:n] - fil.groups['speech'][:n]).max()), 0.05)


class ModelDownload(SimpleTestCase):
    def test_hash_mismatch_is_rejected(self):
        td = Path(tempfile.mkdtemp())
        spec = dict(model_files.YAMNET, size=5, sha256='0' * 64)

        class R:
            def __init__(self):
                self.chunks = [b'abcde', b'']

            def read(self, n):
                return self.chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with mock.patch.object(model_files, 'MODELS_DIR', td), mock.patch('urllib.request.urlopen', return_value=R()):
            with self.assertRaises(RuntimeError):
                model_files.download(spec)
        self.assertEqual(list(td.iterdir()), [], 'nothing is kept after a bad download')

    def test_offline_keeps_heuristics(self):
        with mock.patch.object(se, '_session', None), mock.patch.object(model_files, 'is_ready', return_value=False), \
                mock.patch.object(model_files, 'download', side_effect=OSError('offline')):
            self.assertIsNone(se.load())
            notes = []
            self.assertIsNone(pipeline._sound_timeline('x.mp4', None, 10, None, None, None, notes))
            self.assertIn('sound_model_unavailable', notes)

    def test_tests_never_download(self):
        from django.conf import settings
        self.assertFalse(settings.SOUND_MODEL_DOWNLOAD)


# =====================================================================================================
# captions: font + swearing filter
# =====================================================================================================
class Censor(SimpleTestCase):
    def test_words(self):
        self.assertEqual(captions.censor_text('Бля, меня убили, сука!'), 'Б**, меня убили, с***!')
        self.assertEqual(captions.censor_text('What the fuck'), 'What the f***')
        for clean in ('Сукачёв', 'блин', 'бляха', 'блюдо', 'убили', '****'):
            self.assertEqual(captions.censor_word(clean), clean)

    def test_segments_and_timing_kept(self):
        segs = [{'start': 0, 'end': 2, 'text': 'бля это жесть', 'words': [
            {'start': 0.0, 'end': 0.3, 'text': 'бля'}, {'start': 0.4, 'end': 0.7, 'text': 'это'},
            {'start': 0.8, 'end': 1.2, 'text': 'жесть'}]}]
        out = captions.censor_segments(segs)
        self.assertEqual([w['text'] for w in out[0]['words']], ['б**', 'это', 'жесть'])
        self.assertEqual([w['start'] for w in out[0]['words']], [0.0, 0.4, 0.8])
        self.assertEqual(segs[0]['words'][0]['text'], 'бля', 'the stored transcript is never changed')

    def test_options(self):
        o = exports.resolve({'caption_font': 'Comic Sans', 'caption_censor': 'on'})
        self.assertEqual(o['caption_font'], 'style')
        self.assertTrue(o['caption_censor'])
        self.assertEqual(exports.resolve({'caption_font': 'Montserrat'})['caption_font'], 'Montserrat')


class Fonts(SimpleTestCase):
    def test_style_lines(self):
        segs = [{'start': 0, 'end': 2, 'text': 'а б в', 'words': [{'start': i * 0.4, 'end': i * 0.4 + 0.3, 'text': w}
                                                                  for i, w in enumerate(['раз', 'два', 'три'])]}]
        for font, (name, bold) in captions.FONTS.items():
            dest = Path(tempfile.mkdtemp()) / 'a.ass'
            rep = captions.build_ass(segs, 0, 2, dest, style_id='stream', font=font)
            style = next(l for l in dest.read_text(encoding='utf-8').splitlines() if l.startswith('Style:')).split(',')
            self.assertEqual(style[1], name)
            self.assertEqual(style[7], '-1' if bold else '0')
            self.assertEqual(rep['font'], name)

    def test_montserrat_ships_with_cyrillic(self):
        f = captions.FONTS_DIR / 'Montserrat-Black.ttf'
        self.assertTrue(f.is_file())
        self.assertTrue((captions.FONTS_DIR / 'OFL-Montserrat.txt').is_file())

    def test_graph_uses_relative_fontsdir(self):
        g, _ = layouts.build_graph([{'start': 0, 'end': 2, 'type': 'center'}],
                                   {'W': 720, 'H': 1280, 'src_w': 1920, 'src_h': 1080, 'bg': 'blur'}, fontsdir='fonts')
        self.assertIn('ass=captions.ass:fontsdir=fonts', g)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class CaptionRender(TestCase):
    def test_render_with_font_and_censor(self):
        src = str(sample_video())
        vinfo = editor.probe_video(src)
        words = [{'start': 0.3 + i * 0.4, 'end': 0.6 + i * 0.4, 'text': w} for i, w in enumerate('Бля меня опять убили'.split())]
        tr = [{'start': 0.3, 'end': 2.0, 'text': 'Бля меня опять убили', 'words': words}]
        opts = exports.resolve({'preset': '720p', 'quality': 'small', 'caption_font': 'Montserrat', 'caption_censor': True})
        td = Path(tempfile.mkdtemp())
        plan = editor.prepare(src, {'start': 0, 'end': 3}, 'center', 'auto', opts, vinfo)
        info = editor.render(src, {'start': 0, 'end': 3}, tr, td / 'o.mp4', td, opts, plan, vinfo, 'center')
        ass = (td / 'captions.ass').read_text(encoding='utf-8')
        self.assertIn('Б**', ass)
        self.assertNotIn('БЛЯ', ass)
        self.assertIn('Montserrat Black', ass)
        self.assertTrue((td / 'fonts' / 'Montserrat-Black.ttf').is_file())
        self.assertTrue(info['captions']['censor'])
        self.assertGreater((td / 'o.mp4').stat().st_size, 1000)


class ShortsCensor(TestCase):
    def test_titles_follow_the_filter(self):
        p = Project.objects.create(title='t', status='done', options={'caption_censor': True},
                                   transcript_meta={'detected_language': 'ru'})
        c = Clip.objects.create(project=p, start=0, end=10, status='ready', text='Бля! Меня опять убили! Сука, как так?')
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            s = sm.ensure_for_clip(p, c)
        self.assertNotIn('БЛЯ', s['title'])
        self.assertNotIn('СУКА', s['title'] + s['description'].upper())


# =====================================================================================================
# Whisper batched decoding
# =====================================================================================================
class Seg:
    def __init__(self, a, b, text):
        self.start, self.end, self.text = a, b, text
        self.words = []


class WhisperBatch(SimpleTestCase):
    def setUp(self):
        pipeline._model_cache.clear()

    def _run(self, device, env=None, batched_fails=False):
        calls = {'batched': [], 'seq': 0}

        class Model:
            def transcribe(self, *a, **k):
                calls['seq'] += 1
                return iter([Seg(0, 1, 'раз')]), mock.Mock(language='ru')

        class Batched:
            def __init__(self, model):
                pass

            def transcribe(self, path, batch_size=0, **k):
                calls['batched'].append(batch_size)
                if batched_fails:
                    raise RuntimeError('CUDA out of memory')
                return iter([Seg(1, 2, 'два'), Seg(0, 1, 'раз')]), mock.Mock(language='ru')
        import faster_whisper
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch('studio.diagnostics.whisper_settings', return_value=(device, 'x', 'n')), \
                mock.patch.object(pipeline, '_get_model', return_value=Model()), \
                mock.patch.object(faster_whisper, 'BatchedInferencePipeline', Batched):
            out = pipeline.transcribe('a.wav', 'auto')
        return out, calls

    def test_gpu_uses_batches(self):
        out, calls = self._run('cuda')
        self.assertEqual(calls['batched'], [16])
        self.assertEqual(calls['seq'], 0)
        self.assertEqual([s['text'] for s in out], ['раз', 'два'])        # chronological
        self.assertEqual(pipeline.transcribe.last_info['batch'], 16)

    def test_cpu_stays_sequential_unless_asked(self):
        _, calls = self._run('cpu')
        self.assertEqual((calls['batched'], calls['seq']), ([], 1))
        _, calls = self._run('cpu', {'WHISPER_BATCH': '8'})
        self.assertEqual(calls['batched'], [8])
        _, calls = self._run('cuda', {'WHISPER_BATCH': '0'})
        self.assertEqual((calls['batched'], calls['seq']), ([], 1))

    def test_batch_failure_falls_back(self):
        out, calls = self._run('cuda', batched_fails=True)
        self.assertEqual(calls['batched'], [16, 8])          # out of memory: one retry with half the batch (1D)
        self.assertEqual(calls['seq'], 1)
        self.assertEqual(out[0]['text'], 'раз')
        self.assertEqual(pipeline.transcribe.last_info['batch'], 0)

    def test_batch_size_parsing(self):
        with mock.patch.dict(os.environ, {'WHISPER_BATCH': 'abc'}):
            self.assertEqual(pipeline.whisper_batch_size('cuda'), 0)
        with mock.patch.dict(os.environ, {'WHISPER_BATCH': '999'}):
            self.assertEqual(pipeline.whisper_batch_size('cpu'), 64)
