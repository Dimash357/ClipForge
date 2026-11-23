import os
import sys
from pathlib import Path

from django.apps import AppConfig
from django.db.backends.signals import connection_created


def _sqlite_pragmas(sender, connection, **kwargs):
    # WAL lets the status endpoint read while a worker thread writes progress.
    if connection.vendor == 'sqlite':
        cur = connection.cursor()
        cur.execute('PRAGMA journal_mode=WAL;')
        cur.execute('PRAGMA busy_timeout=30000;')


def is_serving_process():
    """True only in the process that actually serves requests.

    `runserver` with autoreload starts a parent (watcher) and a child
    (RUN_MAIN=true) that serves. Only the child may start workers or recover jobs.
    """
    argv = sys.argv
    if 'runserver' in argv:
        return os.environ.get('RUN_MAIN') == 'true' or '--noreload' in argv
    return False


_backed_up = []


def _backup_before_migrate(sender, plan=None, **kwargs):
    """`manage.py migrate` run by hand: copy db.sqlite3 to backups/ first (start.bat/setup.bat already do it)."""
    if _backed_up or not plan or 'test' in sys.argv:
        return
    _backed_up.append(True)
    try:
        import importlib.util
        import time
        from django.conf import settings
        dest = Path(settings.BASE_DIR) / 'backups'
        if dest.is_dir() and any(time.time() - f.stat().st_mtime < 600 for f in dest.glob('db-*.sqlite3')):
            return                     # start.bat / setup.bat made one a moment ago
        spec = importlib.util.spec_from_file_location('cf_preflight', Path(settings.BASE_DIR) / 'tools' / 'preflight.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        db = settings.DATABASES['default']['NAME']
        mod.backup_db(db=db, dest_dir=Path(settings.BASE_DIR) / 'backups', pending=[f'{m.app_label}.{m.name}' for m, _ in plan])
    except Exception as exc:          # never block a migration because the copy failed; say it loudly
        print(f'[!]  database backup before migrate failed: {exc}')


class StudioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'studio'

    def ready(self):
        connection_created.connect(_sqlite_pragmas)
        from django.db.models.signals import pre_migrate
        pre_migrate.connect(_backup_before_migrate, sender=self)
        if is_serving_process():
            from . import jobs
            jobs.recover_once()
