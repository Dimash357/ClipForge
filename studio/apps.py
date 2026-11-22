import os
import sys

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


class StudioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'studio'

    def ready(self):
        connection_created.connect(_sqlite_pragmas)
        if is_serving_process():
            from . import jobs
            jobs.recover_once()
