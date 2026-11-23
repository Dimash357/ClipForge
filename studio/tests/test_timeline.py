"""Timeline IR v1: model, strict reading, validation rules, versions, fingerprints, schema, adapter, pipeline."""
import copy
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock, skipUnless

from django.core.files import File
from django.core.management import call_command
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import exports, jobs, pipeline
from studio import timeline as T
from studio.timeline import adapters, examples, ids
from studio.timeline import model as m
from studio.timeline import serialize as ser
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'


def problems(data):
    """Issues of a timeline dict (structure errors come back as one issue)."""
    try:
        return T.validate(T.from_dict(data))
    except T.TimelineError as exc:
        return exc.issues or [{'path': '', 'message': str(exc)}]


def has(issues, path_part, msg_part=''):
    return any(path_part in i['path'] and msg_part in i['message'] for i in issues)


# =====================================================================================================
class Ids(SimpleTestCase):
    def test_stable_ids_are_deterministic(self):
        self.assertEqual(ids.stable_id('sc', 'clip', 7, 0), ids.stable_id('sc', 'clip', 7, 0))
        self.assertNotEqual(ids.stable_id('sc', 'clip', 7, 0), ids.stable_id('sc', 'clip', 7, 1))
        self.assertNotEqual(ids.stable_id('sc', 'a', 'bc'), ids.stable_id('sc', 'ab', 'c'))     # separator matters
        self.assertTrue(ids.is_valid(ids.stable_id('sc', 1)))

    def test_new_ids_unique_and_valid(self):
        made = {ids.new_id('it') for _ in range(500)}
        self.assertEqual(len(made), 500)
        self.assertTrue(all(ids.is_valid(i) for i in made))

    def test_bad_ids(self):
        for bad in ('', 'x', 'sc-123456', 'SC_123456', 'sc_12', 'sc_ABCDEF', 'scene_with_long_prefix_123', 7):
            self.assertFalse(ids.is_valid(bad), bad)


# =====================================================================================================
class RoundTrip(SimpleTestCase):
    def test_examples_are_valid_and_round_trip_byte_for_byte(self):
        for build in (examples.cartoon, examples.faceless):
            tl = build()
            self.assertEqual(T.validate(tl), [])
            text = T.to_json(tl)
            back = T.from_json(text)
            self.assertEqual(back, tl)
            self.assertEqual(T.to_json(back), text)

    def test_canonical_json_ignores_key_order_and_int_floats(self):
        d = T.to_dict(examples.cartoon())
        shuffled = json.loads(json.dumps(d, sort_keys=False))
        shuffled = dict(reversed(list(shuffled.items())))
        shuffled['canvas']['fps'] = 30                  # an int for a float field
        self.assertEqual(T.to_json(T.from_dict(shuffled)), T.to_json(examples.cartoon()))

    def test_times_rounded_to_milliseconds(self):
        d = T.to_dict(examples.faceless())
        d['tracks'][0]['items'][0]['duration'] = 5.9999999
        self.assertEqual(T.to_dict(T.from_dict(d))['tracks'][0]['items'][0]['duration'], 6.0)

    def test_item_union_uses_the_type_field(self):
        tl = T.from_dict(T.to_dict(examples.cartoon()))
        kinds = {type(i).__name__ for _, i in tl.items()}
        self.assertEqual(kinds, {'ImageItem', 'TransitionItem', 'AudioItem', 'CaptionItem'})


class StrictReading(SimpleTestCase):
    def setUp(self):
        self.d = T.to_dict(examples.faceless())

    def test_unknown_field_is_an_error_but_ext_is_free(self):
        d = copy.deepcopy(self.d)
        d['tracks'][0]['items'][0]['volum'] = 1
        self.assertTrue(has(problems(d), 'tracks[0].items[0]', "['volum']"))
        d = copy.deepcopy(self.d)
        d['tracks'][0]['items'][0]['ext'] = {'my.plugin': {'anything': [1, 2]}}
        self.assertEqual(problems(d), [])

    def test_types_enums_and_required_fields(self):
        cases = [
            (lambda d: d['canvas'].__setitem__('fps', 'fast'), 'canvas.fps', 'number'),
            (lambda d: d.__setitem__('kind', 'movie'), 'kind', 'not one of'),
            (lambda d: d['tracks'][0]['items'][0].__setitem__('type', 'hologram'), 'tracks[0].items[0]', 'unknown item'),
            (lambda d: d['tracks'][0]['items'][0].pop('asset'), 'tracks[0].items[0].asset', 'required'),
            (lambda d: d['scenes'][0].__setitem__('status', 'done'), 'scenes[0].status', 'not one of'),
            (lambda d: d['tracks'][0]['items'][0].__setitem__('speed', True), 'speed', 'number'),
        ]
        for mutate, path, msg in cases:
            d = copy.deepcopy(self.d)
            mutate(d)
            self.assertTrue(has(problems(d), path, msg), (path, problems(d)))

    def test_not_json(self):
        with self.assertRaises(T.TimelineError):
            T.from_json('{not json')
        with self.assertRaises(T.TimelineError):
            T.from_dict([1, 2])


# =====================================================================================================
class Rules(SimpleTestCase):
    """Each rule on its own: break one thing in a valid timeline, expect exactly that problem."""

    def cartoon(self):
        return T.to_dict(examples.cartoon())

    def faceless(self):
        return T.to_dict(examples.faceless())

    def check(self, d, path, msg=''):
        issues = problems(d)
        self.assertTrue(has(issues, path, msg), (path, msg, issues))

    def test_duplicate_and_bad_ids(self):
        d = self.cartoon()
        d['tracks'][1]['items'][1]['id'] = d['tracks'][1]['items'][0]['id']
        self.check(d, 'tracks[1].items[1].id', 'duplicate')
        d = self.cartoon()
        d['scenes'][0]['id'] = 'scene 1'
        self.check(d, 'scenes[0].id', 'bad id')

    def test_references(self):
        d = self.cartoon()
        d['tracks'][0]['items'][0]['asset'] = 'as_000000000000'
        self.check(d, 'tracks[0].items[0].asset', 'unknown asset')
        d = self.cartoon()
        d['tracks'][0]['items'][0]['scene'] = 'sc_000000000000'
        self.check(d, 'tracks[0].items[0].scene', 'unknown scene')
        d = self.cartoon()
        d['tracks'][3]['items'][0]['style'] = 'st_000000000000'
        self.check(d, 'tracks[3].items[0].style', 'unknown style')
        d = self.cartoon()
        d['scenes'][0]['characters'] = ['ch_000000000000']
        self.check(d, 'scenes[0].characters[0]', 'unknown character')
        d = self.cartoon()
        d['characters'][0]['refs'] = [d['assets'][4]['id']]                # an audio asset
        self.check(d, 'characters[0].refs[0]', 'image')

    def test_item_and_asset_types(self):
        d = self.cartoon()
        d['tracks'][0]['items'][0]['asset'] = d['assets'][4]['id']         # image item -> audio asset
        self.check(d, 'tracks[0].items[0].asset', 'cannot use')
        d = self.cartoon()
        d['tracks'][1]['items'].append(dict(d['tracks'][3]['items'][0], id='it_aaaaaaaaaaaa'))   # caption on audio track
        self.check(d, 'tracks[1].items[2].type', 'cannot hold')
        d = self.faceless()
        d['tracks'][2]['items'][0]['asset'] = d['assets'][0]['id']         # audio from a silent video
        self.check(d, 'tracks[2].items[0].asset', 'no audio')

    def test_time_ranges(self):
        d = self.cartoon()
        d['tracks'][2]['items'][0]['duration'] = 9.0
        self.check(d, 'tracks[2].items[0].duration', 'after the timeline')
        d = self.cartoon()
        d['tracks'][0]['items'][0]['start'] = -1
        self.check(d, 'tracks[0].items[0].start', '>= 0')
        d = self.cartoon()
        d['tracks'][1]['items'][0]['start'] = 5.0                            # voice of scene 1 placed in scene 2
        self.check(d, 'tracks[1].items[0].scene', 'outside its scene')
        d = self.cartoon()
        d['scenes'][1]['start'] = 3.0
        d['scenes'][1]['duration'] = 5.0
        self.check(d, 'scenes', 'overlap')
        d = self.faceless()
        d['tracks'][0]['items'][1]['src_in'] = 5.0                            # 5 + 6 s > 9 s asset
        self.check(d, 'tracks[0].items[1].duration', 'past the end')

    def test_video_track_overlaps_need_a_transition(self):
        d = self.cartoon()
        d['tracks'][0]['items'] = d['tracks'][0]['items'][:2]                # drop the fade
        self.check(d, 'tracks[0]', 'without a transition')
        d = self.cartoon()
        d['tracks'][0]['items'][2]['duration'] = 0.8                         # fade length != overlap
        self.check(d, 'tracks[0].items[2]', 'overlap by')
        d = self.cartoon()
        d['tracks'][0]['items'][2]['to_item'] = 'it_000000000000'
        self.check(d, 'tracks[0].items[2]', 'same track')

    def test_overlays_and_audio_may_stack(self):
        d = self.faceless()
        d['tracks'][1]['items'].append(dict(d['tracks'][1]['items'][0], id='it_bbbbbbbbbbbb', text='ЕЩЁ'))
        self.assertEqual(problems(d), [])

    def test_keyframes_and_animation_values(self):
        d = self.cartoon()
        kf = d['tracks'][0]['items'][0]['animations'][0]['keyframes']
        kf.reverse()
        self.check(d, 'keyframes[1].t', 'time order')
        d = self.cartoon()
        d['tracks'][0]['items'][0]['animations'][0]['keyframes'][1]['t'] = 9.0
        self.check(d, 'keyframes[1].t', 'within the item')
        d = self.faceless()
        d['tracks'][1]['items'][0]['animations'][0]['keyframes'][0]['value'] = 0
        self.check(d, 'keyframes[0].value', '> 0')

    def test_caption_words(self):
        d = self.cartoon()
        d['tracks'][3]['items'][0]['words'][0]['t1'] = 0.1
        self.check(d, 'words[0]', 't1 must be after t0')
        d = self.cartoon()
        d['tracks'][3]['items'][0]['duration'] = 3.0
        self.check(d, 'words[2]', 'outside its caption')
        d = self.cartoon()
        d['tracks'][3]['items'][0]['words'].reverse()
        self.check(d, 'words[1]', 'time order')

    def test_layouts(self):
        d = self.faceless()
        d['tracks'][0]['items'][1]['layout']['params']['path'] = [[1.0, 0, 0], [0.5, 0, 0]]
        self.check(d, 'layout.params.path', 'time order')
        d = self.faceless()
        d['tracks'][0]['items'][1]['layout'] = {'type': 'game_cam', 'params': {}}
        self.check(d, 'layout.params.cam', 'webcam')

    def test_canvas_and_colors(self):
        d = self.cartoon()
        d['canvas']['width'] = 1081
        self.check(d, 'canvas', 'even')
        d = self.cartoon()
        d['styles'][0]['color'] = 'white'
        self.check(d, 'styles[0].color', '#RRGGBB')

    def test_ensure_valid_raises_with_all_issues(self):
        d = self.cartoon()
        d['tracks'][0]['items'][0]['asset'] = 'as_000000000000'
        d['canvas']['fps'] = 500
        with self.assertRaises(T.TimelineError) as cm:
            T.ensure_valid(T.from_dict(d))
        self.assertEqual(len(cm.exception.issues), 2)


# =====================================================================================================
class Versions(SimpleTestCase):
    def test_newer_version_is_refused(self):
        d = T.to_dict(examples.cartoon())
        d['version'] = '2.0'
        with self.assertRaisesRegex(T.TimelineError, 'newer'):
            T.from_dict(d)
        d['version'] = '1.7'
        with self.assertRaisesRegex(T.TimelineError, 'newer'):
            T.from_dict(d)

    def test_wrong_schema_and_unknown_old_version(self):
        d = T.to_dict(examples.cartoon())
        with self.assertRaisesRegex(T.TimelineError, 'not a ClipForge'):
            T.from_dict(dict(d, schema='other.tool'))
        with self.assertRaisesRegex(T.TimelineError, 'no migration'):
            T.from_dict(dict(d, version='0.3'))

    def test_migrations_run_in_order(self):
        old = T.to_dict(examples.faceless())
        old['version'] = '0.8'
        old['title'] = old['meta'].pop('title')                  # pretend 0.8 kept the title at the top level

        @T.register_migration('0.8', '0.9')
        def _a(d):
            d['meta'] = dict(d.get('meta') or {}, title=d.pop('title'))
            return d

        @T.register_migration('0.9', '1.0')
        def _b(d):
            return d
        try:
            tl = T.from_dict(old)
        finally:
            ser._MIGRATIONS.pop('0.8', None)
            ser._MIGRATIONS.pop('0.9', None)
        self.assertEqual(tl.version, T.VERSION)
        self.assertEqual(tl.meta['title'], 'Два факта')
        self.assertEqual(T.validate(tl), [])


class Fingerprints(SimpleTestCase):
    def test_only_the_changed_scene_changes(self):
        tl = examples.cartoon()
        s1, s2 = tl.scenes[0].id, tl.scenes[1].id
        before = (T.scene_fingerprint(tl, s1), T.scene_fingerprint(tl, s2))
        d = T.to_dict(tl)
        d['tracks'][1]['items'][1]['volume'] = 0.8                       # voice of scene 2
        changed = T.from_dict(d)
        self.assertEqual(T.scene_fingerprint(changed, s1), before[0])
        self.assertNotEqual(T.scene_fingerprint(changed, s2), before[1])

    def test_asset_regeneration_changes_its_scene(self):
        tl = examples.cartoon()
        s1 = tl.scenes[0].id
        before = T.scene_fingerprint(tl, s1)
        tl.asset(tl.tracks[0].items[0].asset).generation.seed = 99
        self.assertNotEqual(T.scene_fingerprint(tl, s1), before)

    def test_meta_does_not_change_content(self):
        tl = examples.faceless()
        h = T.content_hash(tl)
        tl.meta['title'] = 'Другое название'
        self.assertEqual(T.content_hash(tl), h)
        with self.assertRaises(T.TimelineError):
            T.scene_fingerprint(tl, 'sc_000000000000')


# =====================================================================================================
def _conforms(value, schema, root, path='$'):
    """Tiny checker for the subset of JSON Schema json_schema() emits (no external dependency)."""
    if '$ref' in schema:
        return _conforms(value, root['$defs'][schema['$ref'].split('/')[-1]], root, path)
    if 'anyOf' in schema:
        errs = [_conforms(value, s, root, path) for s in schema['anyOf']]
        return [] if any(not e for e in errs) else errs[0]
    if 'oneOf' in schema:
        ok = [s for s in schema['oneOf'] if not _conforms(value, s, root, path)]
        return [] if len(ok) == 1 else [f'{path}: matches {len(ok)} of oneOf']
    if 'enum' in schema:
        return [] if value in schema['enum'] else [f'{path}: {value!r} not in enum']
    t = schema.get('type')
    if t == 'object':
        if not isinstance(value, dict):
            return [f'{path}: not an object']
        errs = [f'{path}: missing {r}' for r in schema.get('required', []) if r not in value]
        props = schema.get('properties')
        if props is not None:
            if schema.get('additionalProperties') is False:
                errs += [f'{path}: extra {k}' for k in value if k not in props]
            for k, v in value.items():
                if k in props:
                    errs += _conforms(v, props[k], root, f'{path}.{k}')
        return errs
    if t == 'array':
        if not isinstance(value, list):
            return [f'{path}: not an array']
        return [e for i, v in enumerate(value) for e in _conforms(v, schema['items'], root, f'{path}[{i}]')]
    checks = {'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
              'boolean': lambda v: isinstance(v, bool), 'string': lambda v: isinstance(v, str),
              'null': lambda v: v is None}
    if t in checks and not checks[t](value):
        return [f'{path}: expected {t}']
    return []


class Schema(SimpleTestCase):
    def test_schema_covers_the_model(self):
        sch = T.json_schema()
        json.dumps(sch)
        defs = sch['$defs']
        for cls in (m.Timeline, m.Track, m.Scene, m.Asset, m.Style, m.Character, *m.ITEM_TYPES.values()):
            d = defs[cls.__name__]
            self.assertFalse(d['additionalProperties'])
            self.assertEqual(set(d['properties']), {f.name for f in __import__('dataclasses').fields(cls)})
        self.assertEqual(set(defs['Timeline']['required']), {'id', 'duration'})
        items = defs['Track']['properties']['items']['items']
        self.assertEqual(items['discriminator'], {'propertyName': 'type'})
        self.assertEqual(len(items['oneOf']), len(m.ITEM_TYPES))

    def test_examples_conform_to_the_schema(self):
        sch = T.json_schema()
        for build in (examples.cartoon, examples.faceless):
            self.assertEqual(_conforms(T.to_dict(build()), sch, sch), [])
        bad = T.to_dict(examples.faceless())
        bad['tracks'][0]['items'][0]['colour'] = 1
        self.assertTrue(_conforms(bad, sch, sch))


# =====================================================================================================
def plan_two_scenes():
    cam = {'x': .78, 'y': 0.0, 'w': .22, 'h': .29, 'corner': 'top_right', 'confidence': .9, 'face_rel': (0.5, 0.45),
           'face_h': 0.5, 'eye_rel': 0.4, 'source': 'detected', 'snapped': True, 'frac': 0.9}
    return {'segments': [{'start': 0.0, 'end': 4.0, 'type': 'game_cam', 'face_rel': (0.5, 0.45), 'confidence': 0.9,
                          'fallback': False, 'detected': 'game_cam'},
                         {'start': 4.0, 'end': 9.0, 'type': 'action', 'path': [(0.0, 100.0, 0.0), (5.0, 640.0, 0.0)],
                          'confidence': 1.0, 'fallback': False, 'reason': 'no_webcam'}],
            'cam': cam, 'mode': 'auto'}


TRANSCRIPT = [{'start': 10.2, 'end': 13.0, 'text': 'Бля, меня убили! Ну вот.', 'words': [
    {'start': 10.2, 'end': 10.5, 'text': 'Бля,'}, {'start': 10.6, 'end': 10.9, 'text': 'меня'},
    {'start': 11.0, 'end': 11.4, 'text': 'убили!'}, {'start': 11.6, 'end': 11.9, 'text': 'Ну'},
    {'start': 12.0, 'end': 12.3, 'text': 'вот.'}]},
    {'start': 25.0, 'end': 26.0, 'text': 'после клипа', 'words': [{'start': 25.0, 'end': 25.5, 'text': 'после'}]}]


class Adapter(SimpleTestCase):
    def build(self, **opts):
        o = exports.resolve(dict({'preset': '1080p', 'transition': 'fade', 'caption_censor': True}, **opts))
        info = {'fps': 30, 'transition': o['transition'] if o['transition'] == 'fade' else 'none', 'mode': 'auto',
                'loudness': {'target': -14.0, 'mode': 'two-pass'}, 'encoder': 'libx264', 'size': '1080x1920',
                'auto': {'mode': 'director', 'reason': 'r'}, 'notes': ['n']}
        vinfo = {'width': 1920, 'height': 1080, 'fps': 60.0, 'duration': 3600.0, 'has_audio': True}
        return adapters.clip_timeline(project_id=3, clip_id=11, source_name='sources/abc.mp4', fingerprint='f' * 40,
                                      clip_start=10.0, clip_end=19.0, plan=plan_two_scenes(), opts=o, vinfo=vinfo,
                                      render_info=info, transcript=TRANSCRIPT, title='ЧТО?!')

    def test_describes_the_rendered_clip(self):
        tl = self.build()
        self.assertEqual(T.validate(tl), [])
        self.assertEqual((tl.kind, tl.duration, tl.canvas.width, tl.canvas.height, tl.canvas.fps),
                         ('clip', 9.0, 1080, 1920, 30.0))
        video = tl.tracks[0]
        v1, v2, fade = video.items
        self.assertEqual((v1.layout.type, v2.layout.type, fade.type), ('game_cam', 'action', 'transition'))
        self.assertEqual(v1.src_in, 10.0)
        self.assertEqual(v2.src_in, 14.0)
        self.assertAlmostEqual(v1.duration, 4.25)                 # extended by the fade, like build_graph does
        self.assertEqual(v1.layout.params['cam']['corner'], 'top_right')
        self.assertEqual(v2.layout.params['path'], [[0.0, 100.0, 0.0], [5.0, 640.0, 0.0]])
        audio = tl.tracks[1].items[0]
        self.assertEqual((audio.role, audio.ext['clipforge.loudness']['target']), ('source', -14.0))
        self.assertEqual(tl.meta['clip_range'], [10.0, 19.0])

    def test_captions_relative_censored_with_emphasis(self):
        words = self.build().tracks[2].items[0].words
        self.assertEqual([w.text for w in words], ['Б**,', 'меня', 'убили!', 'Ну', 'вот.'])
        self.assertAlmostEqual(words[0].t0, 0.2)
        self.assertTrue(next(w for w in words if w.text == 'убили!').emphasis)
        self.assertFalse(next(w for w in words if w.text == 'Ну').emphasis)

    def test_ids_are_stable_across_rebuilds(self):
        a, b = self.build(), self.build()
        self.assertEqual(T.to_json(a), T.to_json(b))
        self.assertEqual([s.id for s in a.scenes], [s.id for s in b.scenes])

    def test_without_transition_or_captions(self):
        tl = self.build(transition='none', captions=False)
        self.assertEqual(T.validate(tl), [])
        self.assertEqual([i.type for i in tl.tracks[0].items], ['video', 'video'])
        self.assertEqual(len(tl.tracks), 2)
        self.assertEqual(tl.tracks[0].items[0].duration, 4.0)

    def test_style_follows_options(self):
        tl = self.build(caption_font='Montserrat', subtitle_style='stream')
        st = tl.styles[0]
        self.assertEqual((st.font, st.max_words, st.uppercase), ('Montserrat Black', 1, True))
        classic = self.build(subtitle_style='classic').styles[0]
        self.assertEqual(classic.font, 'Arial')


# =====================================================================================================
class Command(SimpleTestCase):
    def run_cmd(self, *args):
        out = io.StringIO()
        call_command('timeline', *args, stdout=out)
        return out.getvalue()

    def test_example_schema_and_validate(self):
        tl = T.from_json(self.run_cmd('--example', 'cartoon'))
        self.assertEqual(T.validate(tl), [])
        self.assertIn('$defs', json.loads(self.run_cmd('--schema', '--compact')))
        td = Path(tempfile.mkdtemp())
        good = td / 'good.json'
        good.write_text(T.to_json(examples.faceless()), encoding='utf-8')
        self.assertIn('OK: faceless', self.run_cmd('--validate', str(good)))
        bad = T.to_dict(examples.faceless())
        bad['tracks'][0]['items'][0]['asset'] = 'as_000000000000'
        (td / 'bad.json').write_text(json.dumps(bad), encoding='utf-8')
        out = io.StringIO()
        with self.assertRaises(SystemExit):
            call_command('timeline', '--validate', str(td / 'bad.json'), stdout=out)
        self.assertIn('unknown asset', out.getvalue())


# =====================================================================================================
@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class PipelineSnapshot(TransactionTestCase):
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

    def project(self):
        p = pipeline.Project(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode='director',
                             options=exports.resolve({'preset': '720p', 'quality': 'small', 'transition': 'fade'}))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        return p

    def test_every_rendered_clip_carries_a_valid_timeline(self):
        p = self.project()
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()):
            self.run_job('full', p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        for c in p.clips.all():
            tl = T.from_dict(c.render_info['timeline'])
            self.assertEqual(T.validate(tl), [])
            self.assertEqual(len(tl.scenes), len(c.render_info['segments']))
            self.assertAlmostEqual(tl.duration, c.end - c.start, delta=0.01)
            self.assertEqual(tl.meta['clip_id'], c.pk)

    def test_a_broken_snapshot_never_breaks_the_render(self):
        p = self.project()
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()), \
                mock.patch.object(adapters, 'clip_timeline', side_effect=RuntimeError('boom')):
            self.run_job('full', p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        for c in p.clips.all():
            self.assertEqual(c.status, 'ready')
            self.assertIsNone(c.render_info['timeline'])
