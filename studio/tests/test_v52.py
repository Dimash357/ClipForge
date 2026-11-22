"""v5.2 stream editor: moment cues, interest scoring, Reaction Priority, action tracking, stream captions,
loudness/format targets, analysis cache, manual bounds."""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

import numpy as np
from django.core.files import File
from django.test import SimpleTestCase, TestCase, TransactionTestCase, override_settings

from studio import (analysis_cache, audio_analysis as aa, captions, editor, exports, highlights, jobs, layouts,
                    moments, motion_analysis, pipeline, quality, scenes)
from studio.models import Project
from . import synth
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for
from .test_v51 import NEUTRAL, audio_for, transcript

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'


# =====================================================================================================
# speech cues
# =====================================================================================================
class Cues(SimpleTestCase):
    def test_categories(self):
        cases = {
            'ЧТО?! Ты видел это?': 'exclaim', 'Да ладно, не может быть': 'exclaim', 'No way, did you see that': 'exclaim',
            'Бля, ну как так': 'profanity', 'What the fuck was that': 'profanity', 'Это ****** просто': 'profanity',
            'Меня опять убили на мосту': 'fail', 'Миссия провалена, я разбился': 'fail', 'I died again': 'fail',
            'Спасибо за донат, 500 рублей': 'donation', 'Thanks for the bits': 'donation',
            'Нет, чат, вы не правы': 'chat', 'Это баг какой-то, он летает': 'absurd',
            'Ахахаха': 'laugh_text', 'lmao': 'laugh_text',
        }
        for text, cat in cases.items():
            self.assertIn(cat, moments.sentence_cues(text), text)

    def test_plain_speech_has_no_cues(self):
        for text in NEUTRAL:
            self.assertEqual(moments.sentence_cues(text), {}, text)

    def test_reaction_score_grows_with_cues(self):
        self.assertEqual(moments.reaction_score({}), 30)
        one = moments.reaction_score({'exclaim': 1})
        two = moments.reaction_score({'exclaim': 1, 'fail': 1})
        self.assertTrue(30 < one < two <= 100)

    def test_loudness_alone_is_down_weighted(self):
        loud = moments.fuse([{'t': 10, 'end': 11, 'src': 'audio', 'kind': 'loud', 'w': 0.8}])[0]
        real = moments.fuse([{'t': 10, 'end': 11, 'src': 'audio', 'kind': 'loud', 'w': 0.8},
                             {'t': 10.5, 'end': 12, 'src': 'text', 'kind': 'fail', 'w': 1.0}])[0]
        self.assertLess(loud['w'], 0.5)
        self.assertGreater(real['w'], 2.0)            # two modalities agree -> bonus
        self.assertEqual(real['srcs'], ['audio', 'text'])

    def test_key_words_cover_whole_cue_phrases(self):
        keys = moments.key_words('Да ладно! Меня убили, 300 рублей')
        for w in ('да', 'ладно', 'убили', '300'):
            self.assertIn(w, keys)
        self.assertNotIn('меня', keys)


# =====================================================================================================
# event-anchored windows and interest scoring
# =====================================================================================================
def _stream(reaction_text, at=3, scream=True, before=NEUTRAL, after=NEUTRAL, gap=3.0):
    """neutral talk -> a reaction sentence (+ scream) -> neutral talk. Returns segs, duration, profile, react_seg."""
    sents = list(before) + [reaction_text] + list(after)
    segs, dur = transcript(sents, 0.0, gaps={len(before) - 1: gap})
    react = segs[len(before)]
    evs = [(react['start'] - 0.2, synth.scream_like(2.0))] if scream else []
    return segs, dur, audio_for(segs, dur, evs), react


class InterestScoring(SimpleTestCase):
    def test_fail_reaction_beats_neutral_talk_and_hooks_fast(self):
        segs, dur, prof, react = _stream('ЧТО?! Меня убили! Да ладно!')
        best = highlights.find_highlights(segs, dur, 8, 20, 1, audio=prof)[0]
        self.assertTrue(best['start'] <= react['start'] <= best['end'])
        self.assertLessEqual(react['start'] - best['start'], 3.5, 'hook must land in the first seconds')
        self.assertGreaterEqual(react['start'] - best['start'], 0.5, 'the cause before the reaction must be kept')
        self.assertTrue(0 <= best['scores']['interest'] <= 100)
        self.assertGreater(best['scores']['reaction'], 55)
        self.assertIn('фейл', best['reason'])

    def test_not_chosen_by_loudness_alone(self):
        # A: neutral words + very loud unclassified noise.  B: emotional fail reaction with a quieter scream.
        sents_a, t = transcript(NEUTRAL, 0.0)
        noise = (0.6 * np.random.default_rng(5).standard_normal(int(4 * aa.SR))).astype(np.float32)
        sents_b, dur = transcript(['Ну всё, иду туда.', 'ЧТО?! Да ладно!', 'Меня убили, серьёзно?!'] + NEUTRAL[:4], t + 3)
        segs = sents_a + sents_b
        prof = audio_for(segs, dur, [(sents_a[3]['start'], noise), (sents_b[1]['start'] - 0.1, 0.45 * synth.scream_like(1.5))])
        best = highlights.find_highlights(segs, dur, 8, 18, 1, audio=prof)[0]
        self.assertGreaterEqual(best['end'], sents_b[1]['start'], best)
        self.assertGreater(best['start'], sents_a[2]['end'] - 0.5, best)

    def test_screen_fail_event_counts(self):
        segs, dur = transcript(NEUTRAL * 2, 0.0)
        mid = segs[11]['start']
        t = np.arange(0, dur, 1.0)
        luma = np.where(np.abs(t - mid) < 0.6, 5.0, 120.0)
        sp = motion_analysis.ScreenProfile(t, luma, np.full(len(t), 0.4), np.zeros(len(t)), np.zeros(len(t)))
        self.assertTrue(any(e['kind'] == 'fail_screen' for e in sp.events))
        best = highlights.find_highlights(segs, dur, 8, 16, 1, screen=sp)[0]
        self.assertTrue(best['start'] <= mid <= best['end'], (best['start'], best['end'], mid))
        self.assertIn('fail_screen', dict(best['facts']))

    def test_auto_length_follows_content(self):
        segs, dur, prof, react = _stream('ЧТО?! Меня убили! Да ладно!')
        res = highlights.find_highlights(segs, dur, 12, 60, 2, audio=prof, auto_length=True)
        self.assertTrue(res)
        for c in res:
            self.assertGreaterEqual(c['end'] - c['start'], 6.0)
            self.assertLessEqual(c['end'] - c['start'], 61.0)
            self.assertEqual(c['auto_bounds'], [c['start'], c['end']])

    def test_boundaries_never_cut_words(self):
        segs, dur, prof, _ = _stream('Ахахаха, что это было?! Это баг!')
        words = [w for s in segs for w in s['words']]
        for c in highlights.find_highlights(segs, dur, 8, 20, 3, audio=prof):
            for w in words:
                self.assertFalse(w['start'] + 0.01 < c['start'] < w['end'] - 0.01, (c['start'], w))
                self.assertFalse(w['start'] + 0.01 < c['end'] < w['end'] - 0.01, (c['end'], w))

    def test_selected_moments_do_not_repeat(self):
        segs, dur, prof, _ = _stream('ЧТО?! Меня убили! Да ладно!')
        res = highlights.find_highlights(segs, dur, 6, 12, 4, audio=prof)
        for i, a in enumerate(res):
            for b in res[i + 1:]:
                ov = max(0, min(a['end'], b['end']) - max(a['start'], b['start']))
                self.assertLessEqual(ov, 0.2 * min(a['end'] - a['start'], b['end'] - b['start']) + 0.01)


# =====================================================================================================
# captions
# =====================================================================================================
class StreamCaptions(SimpleTestCase):
    def _build(self, text, **kw):
        words, t = [], 0.0
        for w in text.split():
            words.append({'start': round(t, 2), 'end': round(t + 0.3, 2), 'text': w})
            t += 0.34
        dest = Path(tempfile.mkdtemp()) / 'a.ass'
        rep = captions.build_ass([{'start': 0, 'end': t, 'text': text, 'words': words}], 0, t, dest, style_id='stream', **kw)
        events = [l for l in dest.read_text(encoding='utf-8').splitlines() if l.startswith('Dialogue:')]
        return rep, events, words, dest.read_text(encoding='utf-8')

    def test_one_word_per_event_in_sync(self):
        rep, events, words, _ = self._build('Сейчас мы пойдём на мост и посмотрим')
        self.assertEqual(len(events), len(words))
        for ev, w in zip(events, words):
            a = ev.split(',')[1]
            h, m, s = a.split(':')
            self.assertAlmostEqual(float(s), w['start'], delta=0.02)
        self.assertEqual(rep['max_words'], 1)

    def test_style_is_big_bold_white_with_black_outline(self):
        _, _, _, ass = self._build('Привет всем')
        style = next(l for l in ass.splitlines() if l.startswith('Style:')).split(',')
        self.assertGreaterEqual(int(style[2]), 96)                 # size
        self.assertEqual(style[3], '&H00FFFFFF')                    # white
        self.assertEqual(style[5], '&H00000000')                    # black outline colour
        self.assertEqual(style[7], '-1')                            # bold
        self.assertGreaterEqual(int(style[16]), 8)                  # thick outline

    def test_key_words_yellow_others_white(self):
        _, events, _, _ = self._build('Ну и что ЧТО?! Меня убили на мосту')
        texts = [re.sub(r'\{[^}]*\}', '', e.split(',,', 1)[1].split(',', 4)[-1]) for e in events]
        marked = {t for t, e in zip(texts, events) if '1c&H004DE1FF' in e}
        self.assertIn('УБИЛИ', marked)
        self.assertIn('ЧТО?!', marked)
        self.assertNotIn('НА', marked)
        self.assertNotIn('МЕНЯ', marked)
        self.assertLess(len(marked), len(texts))

    def test_language_is_kept(self):
        _, events, _, _ = self._build('Привет чат, сегодня играем')
        joined = ' '.join(events)
        self.assertIn('ПРИВЕТ', joined)
        self.assertIsNone(re.search(r'[A-Z]{3,}', re.sub(r'\{[^}]*\}|Dialogue|S0', '', joined)))

    def test_defaults(self):
        o = exports.resolve({})
        self.assertEqual((o['subtitle_style'], o['fps'], o['preset'], o['loudness']), ('stream', '30', '1080p', 'stream'))
        self.assertEqual(exports.dims(o), (1080, 1920))


# =====================================================================================================
# Reaction Priority geometry
# =====================================================================================================
CAM = {'x': .78, 'y': 0, 'w': .22, 'h': .29, 'corner': 'top_right', 'confidence': .8, 'face_rel': (0.45, 0.55),
       'face_h': 0.5}


class ReactionLayout(SimpleTestCase):
    def test_panel_shares(self):
        for W, H in ((1080, 1920), (720, 1280)):
            g, margin = layouts.reaction_layout(W, H, 1920, 1080, 'a', 'b', CAM)
            self.assertIn(f'crop={W}:{layouts.ev(0.4 * H)}', g)
            self.assertIn('[campanel][gpanel]vstack', g)
            text_bottom = H - margin
            Hc = layouts.ev(0.4 * H)
            self.assertGreater(text_bottom, 0.75 * Hc, 'captions must not cover the face / eyes')
            geo = layouts.cam_geometry(W, H, 1920, 1080, CAM, 'balanced', panel_h=Hc, game_anchor=0.22)
            self.assertLessEqual(text_bottom, Hc + geo['game_y'] + 1, 'captions must not cover the game')

    def test_face_centred_eyes_on_upper_third(self):
        W, Hc = 1080, 768
        cw, ch = 800, 600                          # roomy webcam crop
        sw, sh, x, y = layouts.reaction_cam_crop(W, Hc, cw, ch, 0.5, 0.45, 0.45)
        self.assertAlmostEqual((0.5 * sw - x) / W, 0.5, delta=0.02)
        eyes = (0.45 - 0.12 * 0.45) * sh - y
        self.assertAlmostEqual(eyes / Hc, 1 / 3, delta=0.02)
        # face near an edge: crop is clamped inside the image (no black borders)
        sw, sh, x, y = layouts.reaction_cam_crop(W, Hc, cw, ch, 0.05, 0.1, 0.45)
        self.assertEqual((x, y), (0, 0))
        self.assertTrue(x + W <= sw and y + Hc <= sh)

    def test_action_segment_layout(self):
        ctx = {'W': 1080, 'H': 1920, 'src_w': 1920, 'src_h': 1080, 'bg': 'blur'}
        path = [(0.0, 100.0, 0.0), (5.0, 900.0, 0.0)]
        g, margin = layouts.seg_layout({'type': 'action', 'path': path}, ctx, 'a', 'b')
        self.assertIn("crop=608:1080:x='", g)
        self.assertGreaterEqual(margin, 0.2 * 1920)


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class ReactionGraphsRunInFfmpeg(SimpleTestCase):
    def test_reaction_and_action_graphs(self):
        with tempfile.TemporaryDirectory() as td:
            captions.build_ass([], 0, 1, Path(td) / 'captions.ass', 720, 1280, 'stream')
            segs_list = [[{'start': 0, 'end': 1, 'type': 'reaction_cam'}],
                         [{'start': 0, 'end': 1, 'type': 'action', 'path': [(0, 10, 0), (1, 600, 0)]}]]
            for src in ('1920x1080', '1440x1080'):
                sw, sh = map(int, src.split('x'))
                for cam in (CAM, scenes.manual_cam('bottom_left'), scenes.manual_cam('top_left')):
                    for fit in ('full', 'balanced', 'fill'):
                        for segs in segs_list:
                            ctx = {'W': 720, 'H': 1280, 'src_w': sw, 'src_h': sh, 'bg': 'blur', 'cam': cam, 'cam_fit': fit}
                            graph, _ = layouts.build_graph(segs, ctx)
                            r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                                                f'testsrc2=size={src}:rate=5:duration=0.4', '-filter_complex', graph,
                                                '-map', '[vout]', '-f', 'null', '-'], cwd=td, capture_output=True, text=True)
                            self.assertEqual(r.returncode, 0, (src, cam['corner'], fit, segs[0]['type'], r.stderr[-300:]))


# =====================================================================================================
# rendering targets (1080x1920, 30 FPS, H.264/AAC, -14 LUFS) and Reaction / action renders
# =====================================================================================================
def _moving_box_video(path, fps=60, dur=6, with_audio=True):
    """Dark game-like frame with one bright object moving left -> right, + a quiet tone."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', f'color=c=0x202830:s=1280x720:r={fps}:d={dur}',
           '-f', 'lavfi', '-i', f'color=c=white:s=110x110:r={fps}:d={dur}']
    if with_audio:
        cmd += ['-f', 'lavfi', '-i', f'sine=frequency=330:duration={dur},volume=0.05']
    cmd += ['-filter_complex', "[0][1]overlay=x='60+t*180':y=300[v]", '-map', '[v]']
    if with_audio:
        cmd += ['-map', '2:a', '-c:a', 'aac']
    cmd += ['-c:v', 'libx264', '-preset', 'ultrafast', '-g', str(fps), str(path)]
    subprocess.run(cmd, check=True)
    return path


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class StreamRender(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.td = Path(tempfile.mkdtemp())
        cls.box = _moving_box_video(cls.td / 'box.mp4')
        cls.mix = sample_video()

    def _render(self, src, mode, start=0.0, end=6.0, **opt):
        vinfo = editor.probe_video(src)
        opts = exports.resolve(opt)
        clip = {'start': start, 'end': end}
        td = Path(tempfile.mkdtemp())
        plan = editor.prepare(str(src), clip, mode, 'auto', opts, vinfo)
        dest = td / 'o.mp4'
        info = editor.render(str(src), clip, fake_transcript(), dest, td, opts, plan, vinfo, mode)
        return dest, info, plan, opts

    def test_export_targets_and_quality_checks(self):
        dest, info, plan, opts = self._render(self.box, 'center')
        st = quality.probe_streams(dest)
        self.assertEqual((st['w'], st['h'], st['vcodec'], st['acodec']), (1080, 1920, 'h264', 'aac'))
        self.assertAlmostEqual(st['fps'], 30.0, delta=0.1)              # 60 fps source -> 30
        self.assertEqual(info['loudness']['mode'], 'two-pass')
        q = quality.validate_clip(dest, 6.0, caption_report=info['captions'], text='x', clip_min=5, clip_max=30,
                                  expect={'size': (1080, 1920), 'fps': 30, 'lufs': -14.0})
        checks = {c['id']: c for c in q['checks']}
        self.assertEqual(checks['format']['status'], 'ok', checks['format'])
        self.assertEqual(checks['loudness']['status'], 'ok', checks['loudness'])
        lufs, peak = quality.parse_loudness(quality.run_detectors(dest))
        self.assertAlmostEqual(lufs, -14.0, delta=1.5)
        self.assertLessEqual(peak, -0.5)

    def test_loudness_can_be_turned_off(self):
        dest, info, _, _ = self._render(self.box, 'center', preset='720p', loudness='off')
        self.assertIsNone(info['loudness']['target'])
        lufs, _ = quality.parse_loudness(quality.run_detectors(dest))
        self.assertLess(lufs, -25)                                     # the quiet source stays quiet

    def test_no_webcam_means_full_frame_game_following_the_action(self):
        dest, info, plan, _ = self._render(self.box, 'reaction', preset='720p')
        seg = plan['segments'][0]
        self.assertEqual(seg['type'], 'action')
        xs = [p[1] for p in seg['path']]
        self.assertGreater(xs[-1] - xs[0], 150, 'crop window must follow the object to the right')
        self.assertTrue(any('Вебка не найдена' in n for n in plan['notes']))
        self.assertEqual(quality.probe_streams(dest)['h'], 1280)

    def test_gameplay_priority_without_webcam_also_follows_action(self):
        _, _, plan, _ = self._render(self.box, 'gameplay_facecam', preset='720p')
        self.assertEqual(plan['segments'][0]['type'], 'action')

    def test_reaction_priority_face_centred_on_top(self):
        import cv2
        dest, info, plan, _ = self._render(self.mix, 'reaction', 0.0, 4.0, preset='720p')
        self.assertEqual(plan['segments'][0]['type'], 'reaction_cam')
        f = Path(tempfile.mkdtemp()) / 'f.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(dest), '-frames:v', '1', str(f)], check=True)
        im = cv2.imread(str(f))
        Hc = layouts.ev(0.4 * 1280)
        top = cv2.cvtColor(im[:Hc], cv2.COLOR_BGR2GRAY)
        faces = scenes.detect_faces_gray(top)
        self.assertTrue(faces, 'webcam face must be in the top panel')
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        self.assertAlmostEqual((x + w / 2) / 720, 0.5, delta=0.12)
        self.assertLess(y + 0.45 * h, Hc * 0.55, 'eyes belong to the upper part of the webcam panel')
        game = cv2.cvtColor(im[Hc:], cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(scenes.detect_faces_gray(game)), 0, 'no duplicate webcam in the game panel')


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
def cam_stream_video():
    """A 'stream': gameplay with the webcam overlay on screen the whole time (top right)."""
    if not hasattr(cam_stream_video, 'path'):
        cam_stream_video.path = synth.make_video(Path(tempfile.mkdtemp()) / 'stream.mp4', timeline=[(12, 'game_cam')])
    return cam_stream_video.path


class SourceWebcam(SimpleTestCase):
    def test_whole_video_webcam_detection(self):
        cam = scenes.detect_source_cam(str(cam_stream_video()), 12.0, 1920, 1080)
        self.assertIsNotNone(cam)
        self.assertEqual(cam['corner'], 'top_right')
        self.assertIn('face_h', cam)

    def test_no_webcam(self):
        td = Path(tempfile.mkdtemp())
        self.assertIsNone(scenes.detect_source_cam(str(_moving_box_video(td / 'b.mp4', fps=30)), 6.0, 1280, 720))


# =====================================================================================================
# pipeline: processing order, analysis cache, manual bounds
# =====================================================================================================
class SnapBounds(SimpleTestCase):
    def test_snap_never_cuts_words(self):
        segs = fake_transcript()
        w = segs[1]['words'][2]
        s, e = pipeline.snap_bounds(segs, w['start'] + 0.1, segs[3]['words'][1]['start'] + 0.1)
        self.assertLessEqual(s, w['start'])
        self.assertGreaterEqual(e, segs[3]['words'][1]['end'])


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class PipelineOrderAndCache(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def run_job(self, kind, pk, **payload):
        self.assertTrue(jobs.enqueue(kind, pk, **payload))
        self.assertTrue(wait_for(lambda: not jobs.is_active(pk), timeout=300))

    def test_order_cache_and_fast_reedit(self):
        p = Project(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode='reaction',
                    options=exports.resolve({'preset': '720p', 'quality': 'small'}))
        with open(cam_stream_video(), 'rb') as f:
            p.source.save('stream.mp4', File(f), save=True)
        seen = {}

        def fake_transcribe(path, lang, rep=None, duration=None):
            seen['path'] = str(path)
            return fake_transcript(12.0)
        real_analyze = aa.analyze_file
        calls = {'audio': 0}

        def counting(*a, **k):
            calls['audio'] += 1
            return real_analyze(*a, **k)
        with mock.patch.object(pipeline, 'transcribe', side_effect=fake_transcribe), \
                mock.patch.object(aa, 'analyze_file', side_effect=counting):
            self.run_job('full', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'done', p.error)
            self.assertTrue(seen['path'].endswith('audio16k.wav'), 'Whisper must read the extracted mono 16 kHz track')
            cache = analysis_cache.cache_dir(p.source_fingerprint, create=False)
            out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=channels,sample_rate',
                                           '-of', 'json', str(cache / 'audio16k.wav')])
            st = json.loads(out)['streams'][0]
            self.assertEqual((st['channels'], st['sample_rate']), (1, '16000'))
            for name in ('audio.npz', 'screen.npz'):
                self.assertTrue((cache / name).is_file(), name)
            self.assertTrue(list((cache / 'scenes').glob('*.json')))
            a = p.analysis
            self.assertEqual(a['probe']['width'], 1920)
            self.assertEqual(a['cam']['corner'], 'top_right')
            self.assertTrue(a['candidates'])
            for c in p.clips.all():
                self.assertIn('interest', c.scores)
                self.assertEqual(len(c.scores['auto_bounds']), 2)
                self.assertEqual(c.render_info['segments'][0]['type'], 'reaction_cam')
                ids = {k['id'] for k in c.quality['checks']}
                self.assertTrue({'format', 'loudness'} <= ids)
            self.assertEqual(calls['audio'], 1)
            # re-pick moments: audio / screen / transcript come from the cache
            self.run_job('reprocess', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'done', p.error)
            self.assertEqual(calls['audio'], 1, 'audio must not be analysed again')
            # manual bounds with word snapping
            c = p.clips.order_by('position').first()
            self.run_job('render', p.pk, clip_ids=[c.pk], bounds={str(c.pk): [1.05, 7.5]}, snap=True)
            c.refresh_from_db()
            words = [w for s in fake_transcript(12.0) for w in s['words']]
            self.assertFalse(any(w['start'] < c.start < w['end'] for w in words))
            self.assertLessEqual(c.start, 1.05)
