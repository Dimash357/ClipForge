"""2C Viral Director: score parts and explanations, combination choice without rendering, duplicate filter,
ranking, automatic selection, pipeline modes and the API."""
import shutil
import tempfile
from unittest import mock, skipUnless

from django.core.files import File
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import exports, hooks, jobs, pipeline, smartedit, viral
from studio.models import Project
from .helpers import HAVE_FFMPEG, sample_video
from .test_hooks import TRANSCRIPT, FakeAudio, seg
from .test_jobs_views import wait_for
from .test_smartedit import TR, paused_transcript


def sig(segments=TR, audio=None):
    return hooks.Signals(segments, audio)


class Score(SimpleTestCase):
    def test_parts_reasons_and_bounds(self):
        h = hooks.analyze(TR, 0.0, 23.0, clip_min=10, clip_max=40)
        r = viral.evaluate(sig(), (0.0, 23.0), hooks.variant(h, 'orig'), None, {'context': 60, 'completeness': 70})
        self.assertEqual(set(r['parts']), set(viral.WEIGHTS))
        for k, w in viral.WEIGHTS.items():
            self.assertTrue(0 <= r['parts'][k] <= w, (k, r['parts'][k]))
        self.assertAlmostEqual(sum(r['parts'].values()), r['score'], delta=0.2)
        self.assertEqual(len(r['reasons']), 6)

    def test_tips_point_at_real_problems(self):
        long_tr = [seg(t, 'ну вот мы тут сидим и что-то делаем дальше потом') for t in range(0, 70, 7)]
        r = viral.evaluate(sig(long_tr), (0.0, 70.0), None, None)
        self.assertTrue(any('длинноват' in t for t in r['tips']))
        self.assertTrue(any('Мало реакций' in t for t in r['tips']))
        paused = viral.evaluate(sig(paused_transcript()), (0.0, 14.0), None, None)
        self.assertTrue(any('Smart Editing' in t for t in paused['tips']))

    def test_teaser_and_edit_change_the_score(self):
        audio = FakeAudio(40, events=[{'start': 20.4, 'end': 21.6, 'kind': 'laughter', 'strength': 1.0}])
        s = sig(TRANSCRIPT, audio)
        h = hooks.analyze(s, 10.0, 30.0, clip_min=12, clip_max=40)
        plain = viral.evaluate(s, (10.0, 30.0), hooks.variant(h, 'orig'), None)
        teaser = next(v for v in h['variants'] if v['kind'] == 'teaser')
        with_teaser = viral.evaluate(s, (10.0, 30.0), teaser, None)
        self.assertGreater(with_teaser['parts']['hook'], plain['parts']['hook'])
        self.assertGreater(with_teaser['parts']['retention'], plain['parts']['retention'] - 0.01)
        edit = smartedit.plan(s, 10.0, 30.0, 'dynamic', 12)
        edited = viral.evaluate(s, (10.0, 30.0), hooks.variant(h, 'orig'), edit)
        self.assertLess(edited['length'], plain['length'])


class Direct(SimpleTestCase):
    def test_best_combination_is_found_without_rendering(self):
        h = hooks.analyze(TR, 0.0, 23.0, clip_min=10, clip_max=40)
        with mock.patch('studio.timeline.renderer.render', side_effect=AssertionError('no renders')), \
                mock.patch.object(smartedit, 'plan', wraps=smartedit.plan) as plans:
            r = viral.direct(sig(), (0.0, 23.0), h, {'context': 60, 'completeness': 70}, clip_min=10)
        self.assertTrue(plans.called)
        best = max(r['combos'], key=lambda c: c['score'])
        self.assertGreaterEqual(r['score'], best['score'] - viral.PREFER_SIMPLE)
        self.assertIn(r['combo']['edit'], viral.EDIT_MODES)
        self.assertLessEqual(len(r['combos']), 6)
        self.assertGreaterEqual(r['gain'], 0)
        self.assertIn('не прогноз просмотров', r['disclaimer'])

    def test_simpler_combination_wins_a_tie(self):
        h = {'variants': [{'id': 'orig', 'kind': 'original', 'start': 0.0, 'end': 23.0, 'teaser': None, 'score': 60,
                           'label': 'как есть', 'reasons': []}], 'payoff': None}
        with mock.patch.object(viral, 'evaluate', return_value={'score': 70.0, 'parts': {}, 'reasons': [], 'tips': [],
                                                                  'length': 20.0}):
            r = viral.direct(sig(), (0.0, 23.0), h)
        self.assertEqual(r['combo'], {'hook': 'orig', 'hook_kind': 'original', 'edit': 'off'})
        self.assertEqual(r['gain'], 0.0)


class Rank(SimpleTestCase):
    def test_duplicates_ranking_and_selection(self):
        items = [{'id': 1, 'range': (0, 30), 'text': 'про дракона и меч битва', 'score': 60},
                 {'id': 2, 'range': (20, 50), 'text': 'совсем другая история про рыбалку', 'score': 70},
                 {'id': 3, 'range': (100, 130), 'text': 'битва дракона и меч про', 'score': 55},
                 {'id': 4, 'range': (200, 230), 'text': 'гонки на машинах финиш', 'score': 50}]
        ranked = viral.rank(items)
        by = {it['id']: it for it in ranked}
        self.assertEqual(by[2]['rank'], 1)
        self.assertEqual(by[1]['duplicate_of'], 2)                       # overlaps the stronger clip in time
        self.assertEqual(by[3]['duplicate_of'], None)                    # 1 is not kept, so 3 is not its duplicate
        self.assertEqual([it['id'] for it in ranked if it['rank']], [2, 3, 4])
        self.assertEqual(viral.select(ranked, 2), [2, 3])
        items[0]['range'] = (300, 330)
        self.assertEqual(viral.rank(items)[1]['id'], 1)
        same_words = [{'id': 'a', 'range': (0, 10), 'text': 'один два три четыре пять', 'score': 50},
                      {'id': 'b', 'range': (50, 60), 'text': 'один два три четыре пять шесть', 'score': 40}]
        self.assertEqual(viral.rank(same_words)[1]['duplicate_of'], 'a')


# =====================================================================================================
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

    def run_full(self, count=1, **opts):
        p = Project.objects.create(title='t', clip_count=count, clip_min=6, clip_max=10, edit_mode='center',
                                   options=exports.resolve(dict({'preset': '720p', 'quality': 'small'}, **opts)))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=paused_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        return p

    def test_rank_mode_scores_without_changing_clips(self):
        p = self.run_full()
        c = p.clips.get()
        v = c.meta['viral']
        self.assertEqual((v['rank'], v['applied']), (1, False))
        self.assertTrue(0 <= v['score'] <= 100)
        self.assertNotIn('edit', c.render_info)
        d = self.client.get(f'/api/projects/{p.pk}/viral/').json()
        self.assertEqual(d['ranking'][0]['id'], c.pk)
        self.assertIn('не прогноз', d['disclaimer'])
        self.assertEqual(self.client.post(f'/api/projects/{p.pk}/viral/').status_code, 200)

    def test_auto_mode_picks_from_a_bigger_pool_and_applies_the_combo(self):
        calls = {}
        real = pipeline.pick_highlights

        def spy(*a, **kw):
            calls['count'] = kw.get('count')
            return real(*a, **kw)
        with mock.patch.object(pipeline, 'pick_highlights', side_effect=spy):
            p = self.run_full(count=1, viral_director='auto')
        self.assertEqual(calls['count'], 2)                             # 2x the clips asked for
        self.assertEqual(p.clips.count(), 1)
        c = p.clips.get()
        v = c.meta['viral']
        self.assertTrue(v['applied'])
        self.assertEqual(c.meta['edit']['mode'], v['combo']['edit'])
        self.assertEqual(c.status, 'ready')

    def test_apply_endpoint_re_renders_with_the_best_combo(self):
        p = self.run_full()
        c = p.clips.get()
        v = dict(c.meta['viral'], combo={'hook': 'orig', 'hook_kind': 'original', 'edit': 'dynamic'})
        c.meta = dict(c.meta, viral=v)
        c.save()
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/viral/').status_code, 200)
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        c.refresh_from_db()
        self.assertEqual(c.status, 'ready', c.error)
        self.assertEqual(c.meta['edit']['mode'], 'dynamic')
        self.assertTrue(c.meta['viral']['applied'])
        self.assertEqual(c.render_info.get('edit', {}).get('mode'), 'dynamic')
