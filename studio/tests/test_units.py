import json
import pathlib
import tempfile

from django.core.exceptions import ValidationError
from django.test import SimpleTestCase

from studio import captions, exports, highlights, layouts, metadata, quality, scenes
from studio import textutil as tx
from studio.youtube import parse_youtube_url
from .helpers import fake_transcript


class YoutubeUrl(SimpleTestCase):
    def test_valid(self):
        self.assertEqual(parse_youtube_url('https://www.youtube.com/watch?v=dQw4w9WgXcQ'), 'dQw4w9WgXcQ')
        self.assertEqual(parse_youtube_url('https://youtu.be/dQw4w9WgXcQ'), 'dQw4w9WgXcQ')
        self.assertEqual(parse_youtube_url('https://www.youtube.com/shorts/dQw4w9WgXcQ'), 'dQw4w9WgXcQ')

    def test_rejects_unsafe(self):
        for bad in ('http://www.youtube.com/watch?v=dQw4w9WgXcQ', 'https://evil.com/watch?v=dQw4w9WgXcQ',
                    'https://www.youtube.com/watch?v=short', 'https://user:pw@www.youtube.com/watch?v=dQw4w9WgXcQ',
                    'file:///etc/passwd', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ; rm -rf /'):
            with self.assertRaises(ValidationError, msg=bad):
                parse_youtube_url(bad)


class ExportOptions(SimpleTestCase):
    def test_whitelist(self):
        o = exports.resolve({'preset': 'evil', 'quality': 'high', 'caption_words': '99', 'fps': '60', 'unknown': 1,
                             'subtitle_style': '../../x', 'captions': 'off'})
        self.assertEqual(o['preset'], '1080p')
        self.assertEqual(o['quality'], 'high')
        self.assertEqual(o['caption_words'], 6)
        self.assertEqual(o['subtitle_style'], exports.DEFAULTS['subtitle_style'])
        self.assertFalse(o['captions'])
        self.assertNotIn('unknown', o)

    def test_dims_and_layering(self):
        self.assertEqual(exports.dims(exports.resolve({'preset': '720p'})), (720, 1280))
        self.assertEqual(exports.resolve({'quality': 'small'}, {'quality': 'high'})['quality'], 'high')


class Highlights(SimpleTestCase):
    def test_sentences_and_boundaries(self):
        segs = fake_transcript()
        sents = tx.build_sentences(tx.flat_words(segs))
        self.assertGreaterEqual(len(sents), 6)
        self.assertTrue(sents[0]['text'].endswith('?'))

    def test_clips_start_and_end_on_sentence_edges(self):
        segs = fake_transcript()
        res = highlights.find_highlights(segs, 16.0, 5, 9, 2)
        self.assertTrue(res)
        starts = [s['start'] for s in segs]
        for c in res:
            self.assertTrue(any(abs(c['start'] - st) <= 0.3 for st in starts), c)
            self.assertLessEqual(c['end'] - c['start'], 9.9)
            self.assertGreaterEqual(c['end'] - c['start'], 4.9)
            for k in ('hook', 'context', 'engagement', 'completeness', 'overall'):
                self.assertTrue(0 <= c['scores'][k] <= 100)
            self.assertIn('Выбран', c['reason'])

    def test_no_overlap_and_diversity(self):
        res = highlights.find_highlights(fake_transcript(), 16.0, 4, 7, 3)
        for i, a in enumerate(res):
            for b in res[i + 1:]:
                ov = max(0, min(a['end'], b['end']) - max(a['start'], b['start']))
                self.assertLessEqual(ov, 0.2 * min(a['end'] - a['start'], b['end'] - b['start']) + 1e-6)

    def test_tiny_transcript_returns_empty_so_legacy_fallback_runs(self):
        self.assertEqual(highlights.find_highlights([{'start': 0, 'end': 2, 'text': 'hi there', 'words': []}], 5, 20, 55, 3), [])

    def test_weak_start_penalised(self):
        good = [{'start': 0, 'end': 8, 'text': '', 'words': [
            {'start': i * .3, 'end': i * .3 + .25, 'text': w} for i, w in enumerate(
                'Why is this happening? Nobody tells you the secret behind it. Watch what happens next, it is amazing!'.split())]}]
        sents = tx.build_sentences(tx.flat_words(good))
        f = highlights._sentence_features(sents[0])
        self.assertTrue(f['q_start'])
        weak = highlights._sentence_features({'text': 'And then he said that it was fine', 'words': ['x'] * 8, 'start': 0, 'end': 3})
        self.assertTrue(weak['cont_start'])


class Captions(SimpleTestCase):
    def _words(self):
        words, t = [], 0.0
        for w in ('Почему большинство игроков проигрывает на этой карте? Я расскажу вам секрет, о котором никто '
                  'не говорит! Невероятнонеоправданнодлинноеслово тест').split():
            words.append({'start': round(t, 2), 'end': round(t + .3, 2), 'text': w})
            t += .34
        return [{'start': 0, 'end': t, 'text': '', 'words': words}], t

    def test_all_styles_generate_and_respect_width(self):
        segs, t = self._words()
        for sid in captions.STYLES:
            with tempfile.TemporaryDirectory() as td:
                dest = pathlib.Path(td) / 'a.ass'
                rep = captions.build_ass(segs, 0, t, dest, 1080, 1920, sid, margin_for=lambda x: 420)
                text = dest.read_text(encoding='utf-8')
                self.assertIn('[Events]', text)
                self.assertIn('Dialogue:', text)
                self.assertEqual(rep['est_overflow'], 0, sid)
                self.assertLessEqual(rep['max_words'], 6)

    def test_word_highlight_only_with_reliable_timings(self):
        segs, t = self._words()
        with tempfile.TemporaryDirectory() as td:
            dest = pathlib.Path(td) / 'a.ass'
            self.assertEqual(captions.build_ass(segs, 0, t, dest, style_id='karaoke')['mode'], 'word')
            bad = [{'start': 0, 'end': 5, 'text': 'один два три четыре пять шесть', 'words': []}]
            rep = captions.build_ass(bad, 0, 5, dest, style_id='karaoke')
            self.assertEqual(rep['mode'], 'segment')
            self.assertGreater(rep['events'], 0)

    def test_position_follows_layout(self):
        segs, t = self._words()
        with tempfile.TemporaryDirectory() as td:
            dest = pathlib.Path(td) / 'a.ass'
            captions.build_ass(segs, 0, t, dest, margin_for=lambda x: 640 if x < 2 else 410)
            styles = [l for l in dest.read_text(encoding='utf-8').splitlines() if l.startswith('Style:')]
            self.assertEqual(len(styles), 2)

    def test_legacy_captions_unchanged_by_default(self):
        from studio import pipeline
        segs, t = self._words()
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / 'a.ass'
            pipeline.captions_ass(segs, 0, t, a)
            self.assertIn(',70,70,285,1', a.read_text(encoding='utf-8'))

    def test_punctuation_cleanup(self):
        words = [{'text': ',привет'}, {'text': 'мир,'}]
        self.assertEqual(captions._clean_group_text(words), ['привет', 'мир'])


class SceneLogic(SimpleTestCase):
    def test_flicker_removed(self):
        labels = ['gameplay'] * 10 + ['talking'] * 2 + ['gameplay'] * 8 + ['talking'] * 14
        runs = scenes.smooth_labels(labels, [.8] * len(labels))
        self.assertEqual([r[0] for r in runs], ['gameplay', 'talking'])

    def test_low_confidence_uses_fallback(self):
        samples = [{'t': i / 2, 'faces': [], 'motion': 12.0, 'uniform': .1, 'edge': .05} for i in range(20)]
        # force low-confidence classification
        orig = scenes.classify_sample
        scenes.classify_sample = lambda s, cam: ('talking', 0.3)
        try:
            plan = scenes.plan_scenes({'samples': samples, 'fps': 2.0, 'cam': None}, 10.0, fallback='smart')
        finally:
            scenes.classify_sample = orig
        self.assertEqual(plan[0]['type'], 'smart')
        self.assertTrue(plan[0]['fallback'])

    def test_camera_speed_limited_and_deadzone(self):
        def sample(t, cx):
            return {'t': t, 'faces': [{'cx': cx, 'cy': .5, 'w': .1, 'h': .17, 'area': .0189}]}
        # subject jumps from 30% to 80% of the frame at t=3
        samples = [sample(i / 2, .3 if i < 6 else .8) for i in range(24)]
        path, found = scenes.track_path(samples, 0, 12, 1920, 1080, 606, 1080)
        self.assertTrue(found)
        for (t0, x0, _), (t1, x1, _) in zip(path, path[1:]):
            self.assertLessEqual(abs(x1 - x0) / (t1 - t0), 0.35 * 606 + 1, 'camera moved faster than the limit')
        small = [sample(i / 2, .5 + (0.01 if i % 2 else -0.01)) for i in range(12)]
        p2, _ = scenes.track_path(small, 0, 6, 1920, 1080, 606, 1080)
        self.assertLess(max(x for _, x, _ in p2) - min(x for _, x, _ in p2), 15)   # jitter inside dead zone ignored

    def test_two_faces_framed_together_when_they_fit(self):
        s = [{'t': 0.0, 'faces': [{'cx': .42, 'cy': .5, 'w': .05, 'h': .09, 'area': .0045},
                                  {'cx': .5, 'cy': .5, 'w': .05, 'h': .09, 'area': .0045}]}]
        tg = scenes._choose_targets(s, 0, 1, 606 / 1920)
        self.assertAlmostEqual(tg[0][1][0], .46, places=2)

    def test_path_simplification(self):
        pts = [(i * .5, 100 + (i if i < 6 else 6) * 20, 0) for i in range(14)]
        self.assertLessEqual(len(scenes.simplify_path(pts)), 4)


class Layouts(SimpleTestCase):
    cam = {'x': .78, 'y': 0, 'w': .22, 'h': .29, 'corner': 'top_right', 'confidence': .8}

    def ctx(self, **kw):
        c = {'W': 1080, 'H': 1920, 'src_w': 1920, 'src_h': 1080, 'bg': 'blur', 'cam': self.cam}
        c.update(kw)
        return c

    def test_graph_ends_with_ass_and_yuv420(self):
        g, m = layouts.build_graph([{'start': 0, 'end': 5, 'type': 'gameplay'}], self.ctx())
        self.assertTrue(g.endswith('[vout]'))
        self.assertIn('ass=captions.ass', g)
        self.assertIn('format=yuv420p', g)

    def test_cam_layout_has_no_black_padding_and_hides_duplicate(self):
        g, margin_for = layouts.build_graph([{'start': 0, 'end': 5, 'type': 'game_cam'}], self.ctx())
        mv = margin_for(1.0)
        self.assertNotIn('pad=', g)
        self.assertNotIn(':black', g)
        self.assertIn('[mask]', g)
        self.assertGreater(mv, 0.2 * 1920)
        g2, _ = layouts.build_graph([{'start': 0, 'end': 5, 'type': 'game_cam'}], self.ctx(hide_cam=False))
        self.assertNotIn('[mask]', g2)

    def test_captions_sit_in_safe_zone(self):
        for typ in ('gameplay', 'game_cam', 'screen', 'talking'):
            _, margin = layouts.seg_layout({'type': typ}, self.ctx(), 'a', 'b')
            self.assertGreaterEqual(margin, 0.2 * 1920 - 1, typ)
            self.assertLessEqual(margin, 0.45 * 1920, typ)

    def test_fade_transition_uses_xfade_with_scene_offsets(self):
        segs = [{'start': 0, 'end': 4, 'type': 'gameplay'}, {'start': 4, 'end': 9, 'type': 'screen'}]
        g, _ = layouts.build_graph(segs, self.ctx(), transition='fade')
        self.assertIn('xfade=transition=fade:duration=0.25:offset=4', g)
        self.assertIn('trim=start=0:end=4.25', g)       # first scene is extended by the overlap
        self.assertIn('trim=start=4:end=9', g)
        g2, _ = layouts.build_graph(segs, self.ctx(), transition='none')
        self.assertIn('concat=n=2', g2)

    def test_720p_dimensions_everywhere(self):
        g, _ = layouts.build_graph([{'start': 0, 'end': 5, 'type': 'gameplay'}], self.ctx(W=720, H=1280))
        self.assertIn('720:1280', g)
        self.assertNotIn('1080:1920', g)

    def test_crop_never_stretches(self):
        cw, ch = layouts.crop_dims(1920, 1080, 1080, 1920)
        self.assertLessEqual(abs(cw / ch - 9 / 16), 0.01)
        cw, ch = layouts.crop_dims(1080, 1080, 1080, 1920)     # square source
        self.assertLessEqual(abs(cw / ch - 9 / 16), 0.01)


class Metadata(SimpleTestCase):
    TEXT = ('Почему игроки проигрывают этот раунд? Я расскажу вам секрет про экономику раунда. Если покупать оружие '
            'каждый раунд, то экономика ломается. Поэтому запомните главное правило экономики.')

    def test_titles_come_from_the_text(self):
        m = metadata.extractive(self.TEXT)
        self.assertTrue(3 <= len(m['titles']) <= 5)
        vocab = set(tx.tokens(self.TEXT)) | {'про', 'и'}
        for t in m['titles']:
            self.assertTrue(set(tx.tokens(t)) <= vocab, f'invented words in: {t}')
        self.assertIn('#shorts', m['hashtags'])
        self.assertNotIn('#gaming', m['hashtags'])     # no gameplay seen -> no gaming tag
        self.assertIn('#gaming', metadata.extractive(self.TEXT, scene_types=['gameplay'])['hashtags'])

    def test_ollama_output_is_validated(self):
        import os
        from unittest import mock
        os.environ['USE_OLLAMA'] = '1'
        try:
            bad = mock.Mock()
            bad.json.return_value = {'response': json.dumps({'titles_ru': ['x'], 'titles_en': []})}
            bad.raise_for_status = lambda: None
            with mock.patch('requests.post', return_value=bad):
                self.assertEqual(metadata.generate(self.TEXT)['source'], 'extractive')
            good = mock.Mock()
            good.json.return_value = {'response': json.dumps({'titles_ru': ['Секрет экономики раунда', 'Почему проигрывают раунд'],
                                                              'titles_en': ['The round economy secret', 'Why players lose rounds'],
                                                              'description_ru': 'Про экономику.', 'description_en': 'About economy.'})}
            good.raise_for_status = lambda: None
            with mock.patch('requests.post', return_value=good):
                r = metadata.generate(self.TEXT)
            self.assertTrue(r['bilingual'])
            self.assertEqual(r['source'], 'ollama')
        finally:
            os.environ.pop('USE_OLLAMA', None)


class QualityParsing(SimpleTestCase):
    def test_parse_detectors(self):
        err = ('[blackdetect @ 0x1] black_start:2.0 black_end:3.5 black_duration:1.5\n'
               '[silencedetect @ 0x2] silence_start: 5.0\n[silencedetect @ 0x2] silence_end: 8.2 | silence_duration: 3.2\n'
               '[silencedetect @ 0x2] silence_start: 12.0\n')
        black, silence = quality.parse_detect(err, 14.0)
        self.assertEqual(black, [(2.0, 3.5)])
        self.assertEqual(silence, [(5.0, 8.2), (12.0, 14.0)])
