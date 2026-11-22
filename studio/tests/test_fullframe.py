import subprocess
import tempfile
from pathlib import Path
from unittest import skipUnless
import os

import cv2
import numpy as np
from django.test import Client, SimpleTestCase, TestCase

from studio import editor, exports, scenes
from . import synth
from .helpers import HAVE_FFMPEG, film_video

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'
TR = [{'start': 0, 'end': 6, 'text': 'x', 'words': [{'start': 1 + i * .4, 'end': 1.3 + i * .4, 'text': w}
                                                    for i, w in enumerate('это пример текста для проверки'.split())]}]


class TextDetection(SimpleTestCase):
    def small(self, f):
        return cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)

    def test_finds_white_and_yellow_captions(self):
        boxes = scenes.find_text_boxes(self.small(synth.film_frame(0.0)))
        self.assertGreaterEqual(len(boxes), 2)
        xs0, xs1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        self.assertAlmostEqual(xs0, 0.34, delta=0.05)
        self.assertAlmostEqual(xs1, 0.66, delta=0.05)

    def test_no_false_positives_on_normal_frames(self):
        for f in (synth.game_frame(1.0), synth.cam_overlay(synth.game_frame(2.0)), synth.talking_frame(1.0),
                  synth.screen_frame(1.0), synth.film_frame(0.0, subs=False)):
            self.assertEqual(scenes.find_text_boxes(self.small(f)), [])

    def test_cut_fraction(self):
        samples = [{'t': i / 2, 'text': [(0.36, 0.64, 0.5, 0.55)]} for i in range(10)]
        centred = [(0.0, 656.0, 0.0)]
        self.assertEqual(scenes.crop_cut_fraction(samples, centred, 1920, 608), 0.0)      # text fits
        shifted = [(0.0, 100.0, 0.0), (5.0, 100.0, 0.0)]
        self.assertEqual(scenes.crop_cut_fraction(samples, shifted, 1920, 608), 1.0)     # crop slid left, slices text
        self.assertEqual(scenes.crop_cut_fraction([{'t': 0, 'text': []}], centred, 1920, 608), 0.0)

    def test_summary_needs_persistent_text(self):
        few = [{'t': i, 'text': [(.3, .7, .5, .55)] if i == 0 else []} for i in range(10)]
        self.assertFalse(scenes.hardsub_summary(few)['present'])


class Mode(TestCase):
    def test_mode_registered_and_selectable(self):
        self.assertIn('fullframe', exports.MODE_KEYS)
        html = Client().get('/').content.decode()
        self.assertIn('value="fullframe"', html)
        self.assertIn('name="text_guard"', html)
        self.assertTrue(exports.resolve({})['text_guard'])
        self.assertFalse(exports.resolve({'text_guard': 'off'})['text_guard'])


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class FullFrameRender(TestCase):
    def render(self, mode, **opt):
        src = str(film_video(True))
        vinfo = editor.probe_video(src)
        opts = exports.resolve({'preset': '720p', 'quality': 'small', **opt})
        clip = {'start': 0.0, 'end': 6.0}
        plan = editor.prepare(src, clip, mode, 'auto', opts, vinfo)
        td = Path(tempfile.mkdtemp())
        dest = td / 'o.mp4'
        info = editor.render(src, clip, TR, dest, td, opts, plan, vinfo, mode)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2.5', '-i', str(dest), '-frames:v', '1', str(td / 'f.png')], check=True)
        return cv2.imread(str(td / 'f.png')), info, plan

    def test_whole_frame_is_visible_nothing_cropped(self):
        im, info, _ = self.render('fullframe')
        self.assertEqual(im.shape[:2], (1280, 720))
        row = im[640]                                   # middle row passes through the fitted picture
        left, right = row[3], row[-4]                   # BGR; red marker on source's left edge, green on its right edge
        self.assertTrue(left[2] > 180 and left[0] < 90 and left[1] < 90, f'left edge marker lost: {left}')
        self.assertTrue(right[1] > 180 and right[0] < 90 and right[2] < 90, f'right edge marker lost: {right}')
        self.assertEqual([s['type'] for s in info['segments']], ['fullframe'])
        self.assertTrue(any('вшитые' in n for n in info['notes']), 'should tell the user the source has its own captions')

    def test_smart_switches_to_full_frame_instead_of_slicing_text(self):
        _, info, plan = self.render('smart')
        self.assertEqual([s['type'] for s in info['segments']], ['fullframe'])
        self.assertEqual(info['segments'][0]['reason'], 'hardsub')
        self.assertTrue(any('Без обрезки' in n for n in info['notes']))

    def test_guard_can_be_turned_off(self):
        _, info, _ = self.render('smart', text_guard=False)
        self.assertEqual([s['type'] for s in info['segments']], ['smart'])

    def test_plain_footage_stays_smart(self):
        src = str(film_video(False))
        vinfo = editor.probe_video(src)
        plan = editor.prepare(src, {'start': 0, 'end': 6}, 'smart', 'auto', exports.resolve({}), vinfo)
        self.assertEqual(plan['segments'][0]['type'], 'smart')
        self.assertEqual(plan['notes'], [])
