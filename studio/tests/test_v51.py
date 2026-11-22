"""v5.1: Gameplay + Facecam geometry / duplicate-webcam removal and the multimodal highlight scorer."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

import numpy as np
from django.test import SimpleTestCase, TestCase

from studio import audio_analysis as aa
from studio import captions, editor, exports, highlights, layouts, motion_analysis, pipeline, scenes
from studio.procutil import JobCancelled
from . import synth
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'
CAM_TR = {'x': .78, 'y': 0, 'w': .22, 'h': .29, 'corner': 'top_right', 'confidence': .8}
CAM_BL = {'x': 0, 'y': .68, 'w': .24, 'h': .32, 'corner': 'bottom_left', 'confidence': .8}


# =====================================================================================================
# Gameplay + Facecam layout
# =====================================================================================================
class CamGeometry(SimpleTestCase):
    def test_game_aspect_is_preserved_never_stretched(self):
        for src in ((1920, 1080), (1440, 1080), (2560, 1080), (1280, 720)):
            for fit in layouts.CAM_GAME_FIT:
                for W, H in ((1080, 1920), (720, 1280)):
                    g = layouts.cam_geometry(W, H, *src, CAM_TR, fit)
                    x0, y0, vw, vh = g['crop']
                    gw, gh = g['game']
                    self.assertAlmostEqual(vw / vh, gw / gh, delta=0.02 * gw / gh, msg=(src, fit, W))
                    self.assertLessEqual(gw, W)
                    self.assertLessEqual(gh, g['Hg'])
                    self.assertEqual(g['Hc'] + g['Hg'], H)
                    self.assertTrue(0 <= x0 and x0 + vw <= src[0])

    def test_side_crop_respects_budget(self):
        for fit, budget in layouts.CAM_GAME_FIT.items():
            g = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR, fit)
            self.assertLessEqual(g['crop_frac'], budget + 0.01, fit)
        full = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR, 'full')
        self.assertEqual(full['crop'], (0, 0, 1920, 1080))
        self.assertEqual(full['game'][0], 1080)           # the whole game spans the full width

    def test_game_is_bigger_than_plain_fit(self):
        full = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR, 'full')
        bal = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR, 'balanced')
        self.assertGreater(bal['game'][1], full['game'][1] * 1.15)

    def test_crop_is_spent_on_the_webcam_side(self):
        g = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR, 'balanced')
        self.assertEqual(g['crop'][0], 0)                   # right-side cam: the left edge is kept intact
        g = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_BL, 'balanced')
        x0, _, vw, _ = g['crop']
        self.assertEqual(x0 + vw, 1920)                     # left-side cam: the right edge is kept intact

    def test_cam_fully_cropped_needs_no_patch(self):
        small = {'x': .88, 'y': 0, 'w': .12, 'h': .2, 'corner': 'top_right', 'confidence': .8}
        g = layouts.cam_geometry(1080, 1920, 1920, 1080, small, 'fill')
        self.assertIsNone(g['hide'])
        graph, _ = layouts.cam_layout(1080, 1920, 1920, 1080, 'a', 'b', small, fit='fill')
        self.assertNotIn('[mask]', graph)
        self.assertNotIn('boxblur=2:2,scale=1080:1920', graph)

    def test_webcam_panel_is_sharp_and_bounded(self):
        graph, margin = layouts.cam_layout(1080, 1920, 1920, 1080, 'a', 'b', CAM_TR)
        cam_chain = next(p for p in graph.split(';') if p.startswith('[cam]'))
        self.assertIn('flags=lanczos', cam_chain)
        self.assertIn('unsharp=', cam_chain)
        self.assertNotIn('blur', cam_chain)
        g = layouts.cam_geometry(1080, 1920, 1920, 1080, CAM_TR)
        self.assertLessEqual(g['Hc'], layouts.CAM_PANEL_MAX * 1920 + 2)
        self.assertGreaterEqual(margin, g['Hc'])           # captions sit above the webcam panel

    def test_mirror_fill_is_default_and_blur_is_opt_in(self):
        graph, _ = layouts.cam_layout(1080, 1920, 1920, 1080, 'a', 'b', CAM_TR, fit='full')
        self.assertIn('hflip', graph)
        self.assertIn('vflip', graph)
        self.assertIn('blend=all_expr', graph)
        self.assertNotIn('eq=brightness=-0.10', graph)
        legacy, _ = layouts.cam_layout(1080, 1920, 1920, 1080, 'a', 'b', CAM_TR, fit='full', fill='blur')
        self.assertIn('eq=brightness=-0.10', legacy)
        self.assertNotIn('hflip', legacy)

    def test_mirror_needs_room_else_falls_back(self):
        huge = {'x': .4, 'y': 0, 'w': .6, 'h': .6, 'corner': 'top_right', 'confidence': .8}
        graph, _ = layouts.cam_layout(1080, 1920, 1920, 1080, 'a', 'b', huge, fit='full')
        self.assertIn('[mask]', graph)                      # still hidden (one mirror direction or legacy fill)

    def test_options_whitelisted(self):
        o = exports.resolve({'cam_fit': 'evil', 'cam_fill': 'blur', 'highlight_audio': 'off'})
        self.assertEqual(o['cam_fit'], 'balanced')
        self.assertEqual(o['cam_fill'], 'blur')
        self.assertFalse(o['highlight_audio'])
        self.assertTrue(o['highlight_motion'])


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class CamGraphsRunInFfmpeg(SimpleTestCase):
    """Every geometry / fill / corner / preset combination must be a valid ffmpeg filtergraph."""

    def test_all_combinations(self):
        with tempfile.TemporaryDirectory() as td:
            captions.build_ass([], 0, 1, Path(td) / 'captions.ass', 720, 1280, 'clean')
            for src in ('1920x1080', '1440x1080'):
                sw, sh = map(int, src.split('x'))
                for cam in (CAM_TR, CAM_BL, scenes.manual_cam('top_left'), scenes.manual_cam('bottom_right')):
                    for fit in ('full', 'balanced', 'fill'):
                        for fill in ('mirror', 'blur'):
                            for pos in ('bottom', 'top'):
                                ctx = {'W': 720, 'H': 1280, 'src_w': sw, 'src_h': sh, 'bg': 'blur', 'cam': cam,
                                       'cam_fit': fit, 'cam_fill': fill, 'cam_position': pos}
                                graph, _ = layouts.build_graph([{'start': 0, 'end': 1, 'type': 'game_cam'}], ctx)
                                r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                                                    f'testsrc2=size={src}:rate=5:duration=0.4', '-filter_complex', graph,
                                                    '-map', '[vout]', '-f', 'null', '-'], cwd=td, capture_output=True, text=True)
                                self.assertEqual(r.returncode, 0, (src, cam['corner'], fit, fill, pos, r.stderr[-400:]))


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class CamRender(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.src = sample_video()
        cls.vinfo = editor.probe_video(cls.src)

    def _frame(self, **opt):
        import cv2
        opts = exports.resolve({'preset': '720p', 'quality': 'high', **opt})
        clip = {'start': 0.0, 'end': 4.0}
        td = Path(tempfile.mkdtemp())
        plan = editor.prepare(str(self.src), clip, 'gameplay_facecam', 'auto', opts, self.vinfo)
        dest = td / 'o.mp4'
        editor.render(str(self.src), clip, fake_transcript(), dest, td, opts, plan, self.vinfo, 'gameplay_facecam')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(dest), '-frames:v', '1', str(td / 'f.png')],
                       check=True)
        return cv2.imread(str(td / 'f.png')), plan

    def test_duplicate_webcam_is_rebuilt_not_blurred(self):
        import cv2
        sharp_vals = {}
        for fill in ('mirror', 'blur'):
            im, plan = self._frame(cam_fit='full', cam_fill=fill)
            g = layouts.cam_geometry(720, 1280, 1920, 1080, plan['cam'], 'full')
            gw, gh = g['game']
            s = gw / 1920
            x, y, w, h = g['hide']
            y0 = g['game_y']
            roi = im[y0 + int(y * s) + 4:y0 + int((y + h) * s) - 4, int(x * s) + 4:int((x + w) * s) - 4]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            self.assertEqual(len(scenes.detect_faces_gray(cv2.cvtColor(im[:g['Hg']], cv2.COLOR_BGR2GRAY))), 0, fill)
            sharp_vals[fill] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.assertGreater(sharp_vals['mirror'], 3 * sharp_vals['blur'], sharp_vals)

    def test_webcam_panel_has_face_and_detail(self):
        import cv2
        im, plan = self._frame()
        g = layouts.cam_geometry(720, 1280, 1920, 1080, plan['cam'], 'balanced')
        panel = cv2.cvtColor(im[g['Hg']:], cv2.COLOR_BGR2GRAY)
        self.assertGreaterEqual(len(scenes.detect_faces_gray(panel)), 1)
        # no leftover overlay border (bright vertical line) on the panel's edges
        self.assertLess(float(panel[:, :3].mean()), float(panel[:, 6:40].mean()) + 25)


# =====================================================================================================
# Audio reaction analysis
# =====================================================================================================
def _mix():
    """40 s calm speech, 1 s silence, 2 s scream, 20 s speech, 3 s laughter, 20 s speech."""
    return np.concatenate([synth.speech_like(40), synth.silence(1), synth.scream_like(2), synth.speech_like(20),
                           synth.laugh_like(3), synth.speech_like(20)])


class AudioEvents(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.p = aa.profile_from_samples(_mix())

    def test_scream_and_laughter_found_at_the_right_time(self):
        kinds = {e['kind']: e for e in self.p.events}
        self.assertIn('scream', kinds)
        self.assertIn('laughter', kinds)
        self.assertAlmostEqual(kinds['scream']['start'], 41.0, delta=0.4)
        self.assertAlmostEqual(kinds['scream']['end'], 43.0, delta=0.4)
        self.assertAlmostEqual(kinds['laughter']['start'], 63.0, delta=0.6)
        self.assertTrue(kinds['scream'].get('after_pause'), 'pause before the scream not recognised')
        self.assertFalse([e for e in self.p.events if e['end'] < 40], 'calm speech must not produce events')

    def test_pause_detected(self):
        self.assertTrue(any(abs(a - 40.0) < 0.3 and b - a >= 0.7 for a, b in self.p.pauses), self.p.pauses)

    def test_reaction_span_scores_higher_than_calm_span(self):
        calm, _ = aa.audio_score(aa.span_features(self.p, 5, 20))
        scream, facts = aa.audio_score(aa.span_features(self.p, 35, 50))
        laugh, lf = aa.audio_score(aa.span_features(self.p, 58, 72))
        self.assertGreater(scream, calm + 30)
        self.assertGreater(laugh, calm + 20)
        self.assertIn('scream', dict(facts))
        self.assertIn('laughter', dict(lf))

    def test_volume_spike(self):
        x = np.concatenate([synth.speech_like(20), 0.6 * synth.scream_like(0.5), synth.speech_like(10)])
        p = aa.profile_from_samples(x)
        f = aa.span_features(p, 15, 25)
        self.assertTrue(f['spike'] or f['strong'], p.events)

    def test_empty_and_tiny_inputs(self):
        p = aa.profile_from_samples(np.zeros(aa.SR * 3, np.float32))
        self.assertEqual(p.events, [])
        self.assertIsNone(aa.span_features(p, 1.0, 1.1))
        self.assertEqual(aa.audio_score(None), (None, []))


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class AudioFromFile(SimpleTestCase):
    def test_chunked_ffmpeg_decode_matches_in_memory_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.make_av(Path(td) / 'a.mp4', _mix())
            with mock.patch.object(aa, 'CHUNK_S', 7):            # force many chunk boundaries
                prog = []
                p = aa.analyze_file(path, 86, progress=prog.append)
            self.assertTrue(prog and prog[-1] > 0.9)
            kinds = {e['kind']: e for e in p.events}
            self.assertIn('scream', kinds)
            self.assertAlmostEqual(kinds['scream']['start'], 41.0, delta=0.5)
            ref = aa.profile_from_samples(_mix())
            n = min(len(ref.db), len(p.db))
            self.assertLess(abs(len(ref.db) - len(p.db)), 40)
            self.assertLess(float(np.median(np.abs(ref.db[:n] - p.db[:n]))), 3.0)   # aac is lossy, timing intact

    def test_no_audio_track_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'v.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=160x90:duration=3',
                            '-c:v', 'libx264', '-preset', 'ultrafast', str(path)], check=True)
            self.assertIsNone(aa.analyze_file(path, 3))

    def test_broken_file_raises_and_pipeline_degrades(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / 'bad.mp4'
            bad.write_bytes(b'not a video at all' * 100)
            with self.assertRaises(RuntimeError):
                aa.analyze_file(bad, 10)
            audio, motion, notes = pipeline.highlight_signals(bad, 10, exports.resolve({}))
            self.assertIsNone(audio)
            self.assertIn('audio_failed', notes)

    def test_cancel(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.make_av(Path(td) / 'a.mp4', _mix())
            with mock.patch.object(aa, 'CHUNK_S', 5):
                with self.assertRaises(JobCancelled):
                    aa.analyze_file(path, 86, cancel=lambda: True)


# =====================================================================================================
# Multimodal highlight scoring
# =====================================================================================================
NEUTRAL = ['Сегодня мы снова заходим на карту возле старого моста.', 'Команда собирается у точки и ждёт сигнала.',
           'Противник держит позицию на втором этаже здания.', 'Мы проверяем углы и медленно двигаемся вперёд.',
           'Патроны заканчиваются поэтому нужно экономить выстрелы.', 'Справа слышны шаги кого-то из соперников.',
           'Напарник прикрывает меня с длинной дистанции.', 'Сейчас попробуем обойти их через подвал.']


def transcript(sentences, t0=0.0, gaps=None, wdur=0.34):
    """Word-timed transcript; gaps: {sentence_index: extra seconds of silence after it}."""
    segs, t = [], t0
    for k, s in enumerate(sentences):
        words, st = [], t
        for w in s.split():
            words.append({'start': round(t, 2), 'end': round(t + 0.3, 2), 'text': w})
            t += wdur
        segs.append({'start': round(st, 2), 'end': round(t, 2), 'text': s, 'words': words})
        t += 0.45 + (gaps or {}).get(k, 0.0)
    return segs, t


def audio_for(segs, total, events=()):
    """Speech under every sentence, silence elsewhere, plus (start, signal) events."""
    x = synth.silence(total + 1)
    for s in segs:
        a = int(s['start'] * aa.SR)
        sp = synth.speech_like(s['end'] - s['start'])
        x[a:a + len(sp)] = sp
    for start, sig in events:
        a = int(start * aa.SR)
        x[a:a + len(sig)] = sig[:len(x) - a]
    return aa.profile_from_samples(x)


class MultimodalHighlights(SimpleTestCase):
    def _two_halves(self, ev_half=1):
        segs_a, t = transcript(NEUTRAL, 0.0, gaps={3: 2.5})
        segs_b, t2 = transcript(NEUTRAL, t + 3.0, gaps={3: 2.5})
        segs = segs_a + segs_b
        half = segs_b if ev_half == 1 else segs_a
        scream_at = half[3]['end'] + 0.3
        prof = audio_for(segs, t2, [(scream_at, synth.scream_like(2.0))])
        return segs, t2, prof, scream_at

    def test_text_only_ranking_is_unchanged_without_audio(self):
        segs = fake_transcript()
        a = highlights.find_highlights(segs, 16.0, 5, 9, 2)
        b = highlights.find_highlights(segs, 16.0, 5, 9, 2, audio=None, motion=None)
        self.assertEqual([(c['start'], c['end'], c['score']) for c in a], [(c['start'], c['end'], c['score']) for c in b])
        self.assertNotIn('audio', a[0]['scores'])

    def test_emotional_reaction_wins_over_identical_text(self):
        for half in (0, 1):
            segs, dur, prof, scream_at = self._two_halves(half)
            res = highlights.find_highlights(segs, dur, 8, 16, 1, audio=prof)
            self.assertTrue(res)
            best = res[0]
            self.assertTrue(best['start'] <= scream_at <= best['end'], (half, best['start'], best['end'], scream_at))
            self.assertIn('audio', best['scores'])
            self.assertIn('крик', best['reason'])

    def test_reaction_tail_is_not_cut(self):
        segs, dur, prof, scream_at = self._two_halves(1)
        best = highlights.find_highlights(segs, dur, 8, 16, 1, audio=prof)[0]
        self.assertGreaterEqual(best['end'], scream_at + 2.0 - 0.1)        # the whole 2 s scream is inside
        words = [w for s in segs for w in s['words']]
        for w in words:                                                     # no word is cut in half
            self.assertFalse(w['start'] < best['start'] < w['end'] or w['start'] < best['end'] < w['end'], w)

    def test_lead_in_before_opening_reaction_is_kept(self):
        segs_a, t = transcript(NEUTRAL[:4], 0.0)
        segs_b, t2 = transcript(NEUTRAL[4:] + NEUTRAL[:2], t + 4.0)
        segs = segs_a + segs_b
        burst = segs_b[0]['start'] - 0.1
        prof = audio_for(segs, t2, [(burst, synth.scream_like(1.5))])
        res = highlights.find_highlights(segs, t2, 8, 18, 2, audio=prof)
        hit = next(c for c in res if c['start'] <= segs_b[0]['start'] + 0.5 <= c['end'])
        self.assertLessEqual(hit['start'], burst - 1.0)                      # starts before the burst, with a lead-in
        self.assertGreaterEqual(hit['start'], segs_a[-1]['end'])           # ...but never inside earlier words
        self.assertIn('kept_lead_in', dict(hit['facts']))

    def test_repetitive_and_monotone_fragments_lose(self):
        rep = ['Да да да да да да да да.', 'Ну да да да да да да.', 'Да да да ну да да да.', 'Да да да да да да ну.']
        segs_r, t = transcript(rep * 2, 0.0)
        segs_n, t2 = transcript(NEUTRAL, t + 2.0)
        segs = segs_r + segs_n
        prof = audio_for(segs, t2)
        best = highlights.find_highlights(segs, t2, 8, 16, 1, audio=prof)[0]
        self.assertGreaterEqual(best['start'], segs_n[0]['start'] - 0.5)

    def test_motion_is_folded_in_and_failures_are_ignored(self):
        segs, dur, prof, _ = self._two_halves(1)

        class Fake(motion_analysis.MotionProfile):
            pass

        def motion(spans):
            ts = np.arange(0, dur, 1 / motion_analysis.FPS)
            mo = np.where(ts < dur / 2, 30.0, 1.0)                    # first half: intense action
            return Fake(ts, mo, np.zeros(len(ts), bool))
        res = highlights.find_highlights(segs, dur, 8, 16, 2, audio=prof, motion=motion)
        self.assertTrue(any('action' in c['scores'] for c in res))

        def broken(spans):
            raise RuntimeError('decoder exploded')
        res2 = highlights.find_highlights(segs, dur, 8, 16, 2, audio=prof, motion=broken)
        self.assertTrue(res2)
        self.assertFalse(any('action' in c['scores'] for c in res2))

        def cancelled(spans):
            raise JobCancelled()
        with self.assertRaises(JobCancelled):
            highlights.find_highlights(segs, dur, 8, 16, 2, audio=prof, motion=cancelled)

    def test_scores_bounded_and_reason_present(self):
        segs, dur, prof, _ = self._two_halves(1)
        for c in highlights.find_highlights(segs, dur, 8, 16, 3, audio=prof):
            for k in ('hook', 'context', 'engagement', 'completeness', 'overall', 'audio'):
                self.assertTrue(0 <= c['scores'][k] <= 100, (k, c['scores']))
            self.assertTrue(c['reason'])
            self.assertNotIn('_af', c)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class GameplayMotion(SimpleTestCase):
    def test_gameplay_moves_more_than_ui_and_cuts_are_found(self):
        src = sample_video()            # 0-4 game+cam, 4-8 talking, 8-12 gameplay, 12-16 screen
        mp = motion_analysis.analyze_spans(str(src), [(2.0, 6.0), (8.5, 15.5)], 1920, 1080)
        game = mp.span(8.5, 11.8)
        screen = mp.span(12.3, 15.5)
        self.assertGreater(game['mean'], 3 * max(screen['mean'], 0.01))
        self.assertGreaterEqual(mp.span(2.0, 6.0)['cuts'], 1)
        g_sc, _ = motion_analysis.action_score(game)
        s_sc, s_facts = motion_analysis.action_score(screen)
        self.assertGreater(g_sc, s_sc)
        self.assertIn('static_picture', dict(s_facts))

    def test_spans_are_merged_and_capped(self):
        self.assertEqual(motion_analysis.merge_spans([(10, 20), (0, 5), (19, 30)]), [(0, 5), (10, 30)])
        mp = motion_analysis.analyze_spans(str(sample_video()), [(0, 16)], 1920, 1080, max_total=2.0)
        self.assertLessEqual(mp.t.max() if len(mp.t) else 0, 2.1)
