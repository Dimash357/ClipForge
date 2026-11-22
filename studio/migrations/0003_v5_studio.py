from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [('studio', '0002_edit_modes')]
    operations = [
        migrations.AddField('project', 'stage', models.CharField(max_length=24, blank=True, default='')),
        migrations.AddField('project', 'stage_progress', models.FloatField(null=True, blank=True)),
        migrations.AddField('project', 'cancel_requested', models.BooleanField(default=False)),
        migrations.AddField('project', 'options', models.JSONField(default=dict, blank=True)),
        migrations.AddField('project', 'transcript_meta', models.JSONField(default=dict, blank=True)),
        migrations.AddField('project', 'source_fingerprint', models.CharField(max_length=64, blank=True, default='')),
        migrations.AddField('project', 'source_duration', models.FloatField(default=0)),
        migrations.AddField('project', 'updated_at', models.DateTimeField(auto_now=True, default=django.utils.timezone.now), preserve_default=False),
        migrations.AddField('clip', 'status', models.CharField(max_length=16, default='ready')),
        migrations.AddField('clip', 'error', models.TextField(blank=True)),
        migrations.AddField('clip', 'text', models.TextField(blank=True)),
        migrations.AddField('clip', 'reason', models.TextField(blank=True)),
        migrations.AddField('clip', 'scores', models.JSONField(default=dict, blank=True)),
        migrations.AddField('clip', 'quality', models.JSONField(default=dict, blank=True)),
        migrations.AddField('clip', 'meta', models.JSONField(default=dict, blank=True)),
        migrations.AddField('clip', 'render_info', models.JSONField(default=dict, blank=True)),
        migrations.AddField('clip', 'poster', models.FileField(upload_to='posters/', blank=True)),
        migrations.AddField('clip', 'position', models.PositiveIntegerField(default=0)),
        migrations.AddField('clip', 'updated_at', models.DateTimeField(auto_now=True, default=django.utils.timezone.now), preserve_default=False),
    ]
