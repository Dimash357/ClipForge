"""Timeline IR tools.

    python manage.py timeline --clip 12              print the stored timeline of clip 12 (JSON)
    python manage.py timeline --validate file.json   check a timeline file, list every problem
    python manage.py timeline --schema               print the JSON Schema
    python manage.py timeline --example cartoon      print a reference timeline (cartoon | faceless)
"""
import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from studio import timeline as T


class Command(BaseCommand):
    help = 'Timeline IR: show a clip timeline, validate a file, print the schema or an example'

    def add_arguments(self, parser):
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument('--clip', type=int)
        g.add_argument('--validate', metavar='FILE')
        g.add_argument('--schema', action='store_true')
        g.add_argument('--example', choices=['cartoon', 'faceless'])
        parser.add_argument('--compact', action='store_true', help='one-line JSON')

    def _out(self, data, compact):
        text = json.dumps(data, ensure_ascii=False, indent=None if compact else 2, sort_keys=True)
        self.stdout.write(text)

    def handle(self, *args, **o):
        compact = o['compact']
        if o['schema']:
            return self._out(T.json_schema(), compact)
        if o['example']:
            from studio.timeline import examples
            return self._out(T.to_dict(getattr(examples, o['example'])()), compact)
        if o['clip'] is not None:
            from studio.models import Clip
            c = Clip.objects.filter(pk=o['clip']).first()
            if c is None:
                raise CommandError(f'clip {o["clip"]} not found')
            data = (c.render_info or {}).get('timeline')
            if not data:
                raise CommandError('this clip has no timeline yet: it was rendered before Timeline IR existed '
                                   '(re-render it to create one)')
            return self._out(data, compact)
        path = Path(o['validate'])
        if not path.is_file():
            raise CommandError(f'{path} not found')
        try:
            tl = T.from_json(path.read_text(encoding='utf-8'))
        except T.TimelineError as exc:
            raise CommandError(f'cannot read: {exc}')
        issues = T.validate(tl)
        if not issues:
            self.stdout.write(self.style.SUCCESS(f'OK: {tl.kind} timeline {tl.id}, {tl.duration:.2f}s, '
                                                 f'{len(tl.scenes)} scene(s), {len(tl.tracks)} track(s)'))
            return
        for i in issues:
            self.stdout.write(f'[{i["level"]}] {i["path"]}: {i["message"]}')
        sys.exit(1)
