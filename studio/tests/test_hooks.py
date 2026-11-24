"""2A Hook Engine: variants, Hook Score with reasons, coherence rules, teaser cut on word gaps, Timeline IR teaser,
pipeline 'auto' mode, manual choice through the queue (real render) and the API."""
import shutil
import subprocess
import tempfile
from unittest import mock, skipUnless

import numpy as np
from django.core.files import File
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import exports, hooks, jobs, pipeline
from studio import timeline as T
from studio.models import Clip, Job, Project
from studio.timeline import adapters
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for


def seg(start, text, step=0.35):
    words, t = [], start
    for w in text.split():
        words.append({'start': round(t, 2), 'end': round(t + step - 0.05, 2), 'text': w})
        t += step
    return {'start': start, 'end': words[-1]['end'], 'text': text, 'words': words}


class FakeAudio:
    """Just what the scorer reads from audio_analysis.AudioProfile."""

    def __init__(self, duration, events=(), hot=()):
        self.hop = 0.05
        self.arousal = np.full(int(duration / self.hop), 0.2, np.float32)
        for a, b in hot:
            self.arousal[int(a / self.hop):int(b / self.hop)] = 0.8
        self.events = sorted(events, key=lambda e: e['start'])

    def idx(self, t):
        return int(max(0, min(len(self.arousal), round(t / self.hop))))

    def events_in(self, a, b):
        return [e for e in self.events if e['end'] > a and e['start'] < b]


class FakeScreen:
    def __init__(self, events=()):
        self.events = list(events)


TRANSCRIPT = [
    seg(10.0, 'ну короче так вот значит мы тут сидим.'),                       # weak, slow opening
    seg(13.0, 'Почему никто не знает этот секрет?'),                           # strong hook line
    seg(16.0, 'Сейчас я покажу как это работает в игре.'),
    seg(20.0, 'И тут меня убили! Что?! Ничего себе!'),                        # payoff
    seg(24.0, 'Вот такая история получилась у нас сегодня.'),
    seg(28.0, 'Спасибо что смотрите до конца.'),
]


class Variants(SimpleTestCase):
    def test_weak_opening_gets_a_stronger_sentence_start(self):
        h = hooks.analyze(TRANSCRIPT, 10.0, 30.0, clip_min=12, clip_max=40)
        rec = hooks.variant(h, h['recommended'])
        self.assertEqual(rec['kind'], 'start')
        self.assertEqual(rec['start'], 13.0)
        self.assertEqual(rec['end'], 30.0)                                  # the end and the payoff stay
        self.assertGreaterEqual(rec['score'] - h['original'], hooks.MARGIN)
        orig = hooks.variant(h, 'orig')
        self.assertTrue(any('продолжение' in r['text'] for r in orig['reasons']))
        self.assertTrue(any('вопрос' in r['text'] for r in rec['reasons']))
        for v in h['variants']:
            self.assertTrue(0 <= v['score'] <= 100)
            self.assertAlmostEqual(sum(v['parts'].values()), v['score'], delta=0.2)

    def test_payoff_and_length_limits_are_kept(self):
        h = hooks.analyze(TRANSCRIPT, 10.0, 30.0, clip_min=12, clip_max=40)
        for v in h['variants']:
            if v['kind'] == 'start':
                self.assertLessEqual(v['start'], h['payoff'] - 0.5)          # never after the main moment
                self.assertGreaterEqual(v['end'] - v['start'], 12)
        tight = hooks.analyze(TRANSCRIPT, 10.0, 30.0, clip_min=19, clip_max=40)
        self.assertEqual([v['kind'] for v in tight['variants'] if v['kind'] == 'start'], [])

    def test_strong_line_just_before_the_clip_is_pulled_in(self):
        tr = [seg(7.0, 'Вы никогда не поверите что сейчас будет!')] + TRANSCRIPT[2:]
        h = hooks.analyze(tr, 16.0, 30.0, clip_min=10, clip_max=40)
        starts = [v['start'] for v in h['variants'] if v['kind'] == 'start']
        self.assertNotIn(7.0, starts)                                    # 9 s before: too far, not pulled in
        tr2 = [seg(13.0, 'Вы никогда не поверите что сейчас будет!')] + TRANSCRIPT[2:]
        h2 = hooks.analyze(tr2, 16.0, 30.0, clip_min=10, clip_max=40)
        early = hooks.variant(h2, 'start-13000')
        self.assertIsNotNone(early)                                       # 3 s before: pulled in
        self.assertGreater(early['score'], h2['original'])
        self.assertEqual(early['end'], 30.0)

    def test_original_is_kept_when_nothing_is_clearly_better(self):
        tr = [seg(10.0, 'Почему никто не знает этот секрет?'), seg(13.0, 'Сейчас покажу как это работает.'),
              seg(20.0, 'Вот и всё на сегодня друзья.')]
        h = hooks.analyze(tr, 10.0, 25.0, clip_min=10, clip_max=40)
        self.assertEqual(h['recommended'], 'orig')

    def test_teaser_from_a_peak_cut_on_word_gaps(self):
        audio = FakeAudio(40, events=[{'start': 20.4, 'end': 21.6, 'kind': 'laughter', 'strength': 1.0}],
                          hot=[(20.0, 22.5)])
        screen = FakeScreen([{'t': 20.6, 'kind': 'fail_screen'}])
        h = hooks.analyze(TRANSCRIPT, 10.0, 30.0, clip_min=12, clip_max=40, audio=audio, screen=screen)
        teasers = [v for v in h['variants'] if v['kind'] == 'teaser']
        self.assertTrue(teasers)
        t = teasers[0]
        a, b = t['teaser']
        self.assertTrue(1.0 <= b - a <= 3.0)
        words = hooks.Signals(TRANSCRIPT).words
        self.assertFalse(any(w['start'] + 0.01 < a < w['end'] - 0.01 for w in words))   # not inside a word
        self.assertEqual((t['start'], t['end']), (10.0, 30.0))                          # the whole clip stays
        self.assertTrue(any('звук' in r['text'] for r in t['reasons']))
        self.assertTrue(any('на экране' in r['text'] for r in t['reasons']))
        self.assertGreater(t['score'], h['original'])


# =====================================================================================================
PLAN = {'segments': [{'start': 0.0, 'end': 6.0, 'type': 'center', 'path': [(0.0, 100.0, 0.0)]},
                     {'start': 6.0, 'end': 12.0, 'type': 'gameplay'}], 'cam': None}


class TeaserTimeline(SimpleTestCase):
    def build(self):
        opts = exports.resolve({'preset': '720p', 'transition': 'fade'})
        vinfo = {'width': 1280, 'height': 720, 'fps': 30.0, 'duration': 600.0, 'has_audio': True}
        info = {'fps': 30, 'transition': 'fade', 'loudness': {'target': -14.0, 'mode': 'two-pass'}}
        return adapters.clip_timeline(project_id=1, clip_id=2, source_name='sources/x.mp4', fingerprint='',
                                      clip_start=100.0, clip_end=112.0, plan=PLAN, opts=opts, vinfo=vinfo,
                                      render_info=info, transcript=[seg(100.2, 'раз два три'),
                                                                    seg(107.0, 'Что?! Ничего себе!')])

    def test_teaser_is_prepended_and_everything_shifts(self):
        tl = self.build()
        out = hooks.prepend_teaser(tl, [107.0, 109.0], 100.0)
        self.assertEqual(T.validate(out), [])
        self.assertEqual(out.duration, tl.duration + 2.0)
        v0 = out.tracks[0].items[0]
        self.assertEqual((v0.start, v0.duration, v0.src_in), (0.0, 2.0, 107.0))
        self.assertEqual(v0.layout.type, 'gameplay')                       # layout of the scene it comes from
        self.assertEqual(out.scenes[0].name, 'hook')
        self.assertEqual([round(s.start, 3) for s in out.scenes[1:]], [2.0, 8.0])
        audio = out.tracks[1].items
        self.assertEqual((audio[0].start, audio[0].src_in, audio[1].start), (0.0, 107.0, 2.0))
        self.assertEqual(audio[0].ext, audio[1].ext)                       # same loudness treatment
        caps = out.tracks[2].items
        self.assertEqual([w.text for w in caps[0].words], ['Что?!', 'Ничего', 'себе!'])
        self.assertAlmostEqual(caps[0].words[0].t0, 0.0)
        self.assertAlmostEqual(caps[1].words[0].t0, tl.tracks[2].items[0].words[0].t0 + 2.0)
        self.assertEqual(out.meta['hook']['teaser'], [107.0, 109.0])
        self.assertEqual(T.validate(tl), [])                                # the original is untouched


# =====================================================================================================
def probe_duration(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                 str(path)], capture_output=True, text=True).stdout)


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class PipelineAndApi(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def run_full(self, **opts):
        p = Project.objects.create(title='t', clip_count=1, clip_min=6, clip_max=12, edit_mode='center',
                                   options=exports.resolve(dict({'preset': '720p', 'quality': 'small'}, **opts)))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        return p

    def test_suggest_mode_stores_variants_without_changing_the_clip(self):
        p = self.run_full()
        c = p.clips.get()
        h = c.meta['hooks']
        self.assertEqual(h['clip'], [round(c.start, 2), round(c.end, 2)])
        self.assertIsNone(h['selected'])
        self.assertTrue(h['variants'])
        self.assertNotIn('hook', c.render_info)

    def test_off_mode(self):
        c = self.run_full(hook_engine='off').clips.get()
        self.assertNotIn('hooks', c.meta)

    def test_choose_teaser_renders_it_in_front_through_timeline_ir(self):
        p = self.run_full()
        c = p.clips.get()
        L = c.end - c.start
        h = c.meta['hooks']
        a = round(c.start + 2.0, 2)
        h['variants'].append({'id': 'teaser-test', 'kind': 'teaser', 'start': c.start, 'end': c.end,
                              'teaser': [a, round(a + 1.5, 2)], 'window': [a, a + 1.5], 'score': 90.0,
                              'parts': {}, 'reasons': [], 'text': ''})
        c.meta = dict(c.meta, hooks=h)
        c.save()
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/hooks/', {'variant': 'nope'},
                                          content_type='application/json').status_code, 400)
        r = self.client.post(f'/api/clips/{c.pk}/hooks/', {'variant': 'teaser-test'}, content_type='application/json')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        c.refresh_from_db()
        self.assertEqual(c.status, 'ready', c.error)
        self.assertNotIn('error', c.render_info['hook'])
        self.assertEqual(c.render_info['hook']['teaser'], [a, round(a + 1.5, 2)])
        self.assertAlmostEqual(probe_duration(c.video.path), L + 1.5, delta=0.15)
        self.assertEqual(T.from_dict(c.render_info['timeline']).scenes[0].name, 'hook')
        self.assertEqual(c.meta['hooks']['selected'], 'teaser-test')
        # back to the original opening: plain render again
        self.client.post(f'/api/clips/{c.pk}/hooks/', {'variant': 'orig'}, content_type='application/json')
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        c.refresh_from_db()
        self.assertNotIn('hook', c.render_info)
        self.assertAlmostEqual(probe_duration(c.video.path), L, delta=0.15)

    def test_api_computes_missing_hooks_and_refuses_while_busy(self):
        p = self.run_full(hook_engine='off')
        c = p.clips.get()
        d = self.client.get(f'/api/clips/{c.pk}/hooks/').json()
        self.assertTrue(d['hooks']['variants'])
        c.refresh_from_db()
        self.assertIn('hooks', c.meta)
        Job.objects.create(kind='render', project=p, status=Job.PAUSED)
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/hooks/', {'variant': 'orig'},
                                          content_type='application/json').status_code, 409)

    def test_auto_mode_applies_a_start_variant(self):
        def analyze(sig, a0, b0, *args, **kw):
            v = {'id': 'start-x', 'kind': 'start', 'start': round(a0 + 1.0, 2), 'end': b0, 'teaser': None,
                 'window': [a0 + 1, a0 + 4], 'score': 80.0, 'parts': {}, 'reasons': [], 'text': ''}
            return {'version': 1, 'variants': [v], 'recommended': 'start-x', 'selected': None, 'original': 40.0,
                    'clip': [a0, b0]}
        with mock.patch.object(hooks, 'analyze', side_effect=analyze):
            p = self.run_full(hook_engine='auto')
        c = p.clips.get()
        self.assertEqual(c.meta['hooks']['selected'], 'start-x')
        self.assertAlmostEqual(c.start, c.meta['hooks']['clip'][0] + 1.0, delta=0.01)
        self.assertAlmostEqual(probe_duration(c.video.path), c.end - c.start, delta=0.15)
