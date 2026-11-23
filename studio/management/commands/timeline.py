"""Timeline IR tools.

    python manage.py timeline --clip 12              print the stored timeline of clip 12 (JSON)
    python manage.py timeline --validate file.json   check a timeline file, list every problem
    python manage.py timeline --schema               print the JSON Schema
    python manage.py timeline --example cartoon      print a reference timeline (cartoon | faceless)
    python manage.py timeline --render file.json --out out.mp4 [--base DIR] [--rerender SCENE_ID ...] [--cpu]
    python manage.py timeline --render-example cartoon --out out.mp4   (local placeholder assets, no downloads)
    python manage.py timeline --render-clip 12 --out out.mp4          (a Stream Clipper clip via the new renderer)
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
        g.add_argument('--render', metavar='FILE')
        g.add_argument('--render-example', choices=['cartoon', 'faceless'])
        g.add_argument('--render-clip', type=int, metavar='CLIP_ID')
        parser.add_argument('--compact', action='store_true', help='one-line JSON')
        parser.add_argument('--out', metavar='MP4')
        parser.add_argument('--base', metavar='DIR', help='folder of file:/gen: assets (default MEDIA_ROOT)')
        parser.add_argument('--rerender', nargs='*', default=[], metavar='SCENE_ID', help='ignore the cache for these')
        parser.add_argument('--cpu', action='store_true', help='never use NVENC')

    def _out(self, data, compact):
        text = json.dumps(data, ensure_ascii=False, indent=None if compact else 2, sort_keys=True)
        self.stdout.write(text)

    def _render(self, tl, o, base=None):
        from studio.timeline import renderer
        if not o['out']:
            raise CommandError('--out is required')
        st = renderer.RenderSettings(base_dir=base, encoder='cpu' if o['cpu'] else 'auto')
        try:
            r = renderer.render(tl, o['out'], st, rerender=o['rerender'],
                                on_progress=lambda f: self.stderr.write(f'\r{int(f * 100):3d}%', ending=''))
        except T.TimelineError as exc:
            self.stderr.write('')
            for i in exc.issues or [{'path': '', 'message': str(exc)}]:
                self.stdout.write(f'[error] {i["path"]}: {i["message"]}')
            sys.exit(1)
        self.stderr.write('')
        for s in r['scenes']:
            self.stdout.write(f'scene {s["id"]}: {"cache" if s["cached"] else "rendered %ss" % s.get("seconds")}')
        self.stdout.write(self.style.SUCCESS(f'OK: {o["out"]} {r["size"]} {r["duration"]:.2f}s, {r["encoder_note"]}, '
                                             f'{r["seconds"]}s'))

    def handle(self, *args, **o):
        compact = o['compact']
        if o['render_example']:
            from django.conf import settings
            from pathlib import Path as P
            from studio.timeline import examples
            base = P(o['base'] or P(settings.MEDIA_ROOT) / 'timeline_examples')
            tl = getattr(examples, o['render_example'])()
            examples.placeholder_assets(tl, base)
            return self._render(tl, o, base)
        if o['render']:
            path = Path(o['render'])
            if not path.is_file():
                raise CommandError(f'{path} not found')
            try:
                tl = T.from_json(path.read_text(encoding='utf-8'))
            except T.TimelineError as exc:
                raise CommandError(f'cannot read: {exc}')
            return self._render(tl, o, o['base'])
        if o['render_clip'] is not None:
            from studio.models import Clip
            c = Clip.objects.filter(pk=o['render_clip']).first()
            data = c and (c.render_info or {}).get('timeline')
            if not data:
                raise CommandError('clip not found or it has no timeline (re-render it first)')
            return self._render(T.from_dict(data), o)
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
