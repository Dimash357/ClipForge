from django.db import migrations,models
import django.db.models.deletion
class Migration(migrations.Migration):
 initial=True
 dependencies=[]
 operations=[
 migrations.CreateModel(name='Project',fields=[('id',models.BigAutoField(auto_created=True,primary_key=True,serialize=False,verbose_name='ID')),('created_at',models.DateTimeField(auto_now_add=True)),('title',models.CharField(blank=True,max_length=250)),('youtube_url',models.URLField(blank=True)),('source',models.FileField(blank=True,upload_to='sources/')),('status',models.CharField(default='queued',max_length=30)),('progress',models.PositiveIntegerField(default=0)),('message',models.TextField(blank=True)),('language',models.CharField(default='auto',max_length=8)),('clip_count',models.PositiveIntegerField(default=3)),('clip_min',models.PositiveIntegerField(default=20)),('clip_max',models.PositiveIntegerField(default=55)),('transcript',models.JSONField(default=list)),('error',models.TextField(blank=True))]),
 migrations.CreateModel(name='Clip',fields=[('id',models.BigAutoField(auto_created=True,primary_key=True,serialize=False,verbose_name='ID')),('start',models.FloatField()),('end',models.FloatField()),('score',models.FloatField(default=0)),('title',models.CharField(blank=True,max_length=250)),('video',models.FileField(blank=True,upload_to='renders/')),('created_at',models.DateTimeField(auto_now_add=True)),('project',models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,related_name='clips',to='studio.project'))])]
