"""Windows install/start helpers: tools/preflight.py and the .bat scripts."""
import importlib.util
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.test import SimpleTestCase

ROOT = Path(settings.BASE_DIR)
_spec = importlib.util.spec_from_file_location('preflight', ROOT / 'tools' / 'preflight.py')
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


class Venv(SimpleTestCase):
    def test_system_python_is_refused(self):
        with mock.patch.object(sys, 'prefix', '/usr'), mock.patch.object(sys, 'base_prefix', '/usr'):
            self.assertFalse(pf.in_venv())
            self.assertEqual(pf.check_venv(), 1)

    def test_venv_accepted(self):
        with mock.patch.object(sys, 'prefix', '/p/env'), mock.patch.object(sys, 'base_prefix', '/usr'):
            self.assertTrue(pf.in_venv())
            self.assertEqual(pf.check_venv(), 0)

    def test_old_python_refused(self):
        with mock.patch.object(sys, 'version_info', (3, 9, 0)):
            self.assertEqual(pf.check_venv(), 1)


class Ffmpeg(SimpleTestCase):
    def _run(self, filters):
        def run(cmd, timeout=30):
            if '-version' in cmd:
                return 0, 'ffmpeg version 7.1-full_build-www.gyan.dev Copyright'
            return 0, 'Filters:\n' + '\n'.join(f' ... {f}  V->V  x' for f in filters)
        return run

    def test_missing(self):
        rep = pf.ffmpeg_report(which=lambda n: None, run=self._run([]))
        self.assertEqual(len(rep['problems']), 2)

    def test_no_libass(self):
        rep = pf.ffmpeg_report(which=lambda n: f'C:/ff/bin/{n}.exe', run=self._run(['scale', 'loudnorm']))
        self.assertFalse(rep['ass'])
        self.assertTrue(any('libass' in p for p in rep['problems']))

    def test_full_build(self):
        rep = pf.ffmpeg_report(which=lambda n: f'C:/ff/bin/{n}.exe', run=self._run(['ass', 'loudnorm', 'scale']))
        self.assertEqual(rep['problems'], [])
        self.assertEqual(rep['version'], '7.1-full_build-www.gyan.dev')

    def test_mixed_builds_warned(self):
        rep = pf.ffmpeg_report(which=lambda n: f'C:/{n}/{n}.exe', run=self._run(['ass', 'loudnorm']))
        self.assertTrue(rep['warnings'])

    def test_real_ffmpeg_here(self):
        if not pf.shutil.which('ffmpeg'):
            self.skipTest('no ffmpeg')
        self.assertEqual(pf.check_ffmpeg(), 0)


class Constraints(SimpleTestCase):
    def test_keeps_installed_numpy(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'c.txt'
            with mock.patch.dict(os.environ, {'CLIPFORGE_ALLOW_NUMPY_UPGRADE': ''}):
                pf.write_constraints(out)
            import numpy
            self.assertEqual(out.read_text().strip(), f'numpy=={numpy.__version__}')

    def test_upgrade_can_be_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'c.txt'
            with mock.patch.dict(os.environ, {'CLIPFORGE_ALLOW_NUMPY_UPGRADE': '1'}):
                pf.write_constraints(out)
            self.assertEqual(out.read_text().strip(), '')

    def test_deps_import(self):
        self.assertEqual(pf.check_deps(), 0)


class DatabaseBackup(SimpleTestCase):
    def test_backup_contains_uncheckpointed_wal_data(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'db.sqlite3'
            con = sqlite3.connect(db)
            con.execute('PRAGMA journal_mode=WAL')
            con.execute('PRAGMA wal_autocheckpoint=0')
            con.execute('create table t(x)')
            con.executemany('insert into t values (?)', [(i,) for i in range(100)])
            con.commit()                                  # rows live in db.sqlite3-wal, not in the main file yet
            dest = pf.backup_db(db, Path(td) / 'backups', pending=['studio.0004_source_analysis'])
            con.close()
            self.assertTrue(dest.is_file())
            self.assertEqual(sqlite3.connect(dest).execute('select count(*) from t').fetchone()[0], 100)
            self.assertTrue(db.is_file(), 'the original database is never touched')

    def test_no_backup_when_nothing_to_migrate_or_no_db(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(pf.backup_db(Path(td) / 'missing.sqlite3', Path(td) / 'b'))
            db = Path(td) / 'db.sqlite3'
            sqlite3.connect(db).close()
            self.assertIsNone(pf.backup_db(db, Path(td) / 'b', pending=[]))
            self.assertFalse((Path(td) / 'b').exists())


class BatchScripts(SimpleTestCase):
    NAMES = ('setup.bat', 'start.bat', 'run_tests.bat', 'diagnose.bat', 'install_gpu.bat')

    def test_ascii_crlf(self):
        for n in self.NAMES:
            raw = (ROOT / n).read_bytes()
            raw.decode('ascii')
            self.assertNotIn(b'\n', raw.replace(b'\r\n', b''), n)

    def test_only_the_env_python_installs_or_runs(self):
        for n in self.NAMES:
            text = (ROOT / n).read_text()
            self.assertIn(r'set "ENVDIR=%~dp0env"', text, n)
            self.assertNotIn('.venv', text, n)
            self.assertNotIn('activate', text, n)
            for line in text.splitlines():
                s = line.strip()
                if s.lower().startswith('rem') or s.lower().startswith('echo'):
                    continue
                if re.search(r'(?<![\w-])pip(?![\w-])', s):
                    self.assertTrue(s.startswith('"%ENVPY%" -m pip'), (n, s))
                if 'manage.py' in s or 'preflight.py' in s:
                    self.assertTrue(s.startswith('"%ENVPY%"'), (n, s))

    def test_setup_never_deletes(self):
        text = (ROOT / 'setup.bat').read_text().lower()
        for bad in ('rmdir', ' rd ', 'del ', 'erase '):
            self.assertNotIn(bad, text)

    def test_every_goto_has_a_label(self):
        for n in self.NAMES:
            text = (ROOT / n).read_text()
            labels = set(re.findall(r'^:(\w+)', text, re.M))
            for target in re.findall(r'goto :?(\w+)', text):
                self.assertIn(target, labels, (n, target))
            for target in re.findall(r'call :(\w+)', text):
                self.assertIn(target, labels, (n, target))


class JsRuntime(SimpleTestCase):
    def test_messages(self):
        self.assertEqual(pf.check_js_runtime(which=lambda n: None), 0)          # only a warning
        self.assertEqual(pf.check_js_runtime(which=lambda n: '/x/deno' if n == 'deno' else None), 0)
