import json

from django.core.management.base import BaseCommand

from studio import diagnostics


class Command(BaseCommand):
    help = 'Print FFmpeg / CUDA / NVENC / library diagnostics (same data as the /system/ page).'

    def handle(self, *args, **opts):
        self.stdout.write(json.dumps(diagnostics.full_report(), indent=2, ensure_ascii=False, default=str))
