"""Long transcriptions never look frozen: phase + elapsed time until the first segment; quiet console."""
import logging
import time
from unittest import mock

from django.test import SimpleTestCase

from studio import pipeline
from clipforge.logfilters import QuietPolling


class Seg:
    def __init__(self, a, b, text):
        self.start, self.end, self.text, self.words = a, b, text, []


class Rep:
    def __init__(self):
        self.notes, self.fracs = [], []

    def note(self, m):
        self.notes.append(m)

    def frac(self, v, m=None):
        self.fracs.append((v, m))

    def check(self):
        pass


class Heartbeat(SimpleTestCase):
    def setUp(self):
        pipeline._model_cache.clear()

    def _run(self, cached, first_delay=0.35):
        class Model:
            def transcribe(self, *a, **k):
                def gen():
                    time.sleep(first_delay)          # VAD over a long file / slow first batch
                    yield Seg(0, 1, 'раз')
                return gen(), mock.Mock(language='ru')
        rep = Rep()
        with mock.patch('studio.diagnostics.whisper_settings', return_value=('cpu', 'int8', 'n')), \
                mock.patch.object(pipeline, '_get_model', return_value=Model()), \
                mock.patch.object(pipeline, '_whisper_cached', return_value=cached), \
                mock.patch.object(pipeline._Heartbeat, '__init__', lambda s, r, every=0.1: _init(s, r, 0.1)):
            out = pipeline.transcribe('a.wav', 'auto', rep=rep, duration=100)
        return out, rep

    def test_first_run_says_the_model_is_downloading(self):
        out, rep = self._run(cached=False)
        self.assertEqual(out[0]['text'], 'раз')
        self.assertTrue(any('Скачиваем модель Whisper' in n for n in rep.notes), rep.notes)

    def test_elapsed_time_is_shown_until_the_first_segment(self):
        _, rep = self._run(cached=True)
        ticks = [n for n in rep.notes if n.startswith('Распознаём речь (CPU)')]
        self.assertGreaterEqual(len(ticks), 2, rep.notes)
        self.assertIn('прошло 0:0', ticks[-1])
        self.assertTrue(rep.fracs and 'Распознаём речь: 1%' in rep.fracs[-1][1])
        self.assertIn('seconds', pipeline.transcribe.last_info)

    def test_cache_check_never_raises(self):
        with mock.patch('huggingface_hub.try_to_load_from_cache', side_effect=RuntimeError('x')):
            self.assertTrue(pipeline._whisper_cached('small'))


_real_init = pipeline._Heartbeat.__init__


def _init(self, rep, every):
    _real_init(self, rep, every)


class ConsoleFilter(SimpleTestCase):
    def _rec(self, msg):
        return logging.LogRecord('django.server', logging.INFO, __file__, 1, msg, None, None)

    def test_status_polls_hidden_errors_kept(self):
        f = QuietPolling()
        self.assertFalse(f.filter(self._rec('"GET /api/projects/1/status/ HTTP/1.1" 200 1803')))
        self.assertTrue(f.filter(self._rec('"GET /api/projects/1/status/ HTTP/1.1" 500 120')))
        self.assertTrue(f.filter(self._rec('"POST /api/projects/1/render/ HTTP/1.1" 200 11')))
        self.assertTrue(f.filter(self._rec('"GET /projects/1/ HTTP/1.1" 200 5000')))
