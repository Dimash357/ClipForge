from django.db import models


class Project(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=250, blank=True)
    youtube_url = models.URLField(blank=True)
    source = models.FileField(upload_to='sources/', blank=True)
    status = models.CharField(max_length=30, default='queued')
    progress = models.PositiveIntegerField(default=0)
    message = models.TextField(blank=True)
    language = models.CharField(max_length=8, default='auto')
    clip_count = models.PositiveIntegerField(default=3)
    clip_min = models.PositiveIntegerField(default=20)
    clip_max = models.PositiveIntegerField(default=55)
    edit_mode = models.CharField(max_length=24, default='smart')
    facecam_corner = models.CharField(max_length=16, default='auto')
    transcript = models.JSONField(default=list)
    error = models.TextField(blank=True)
    # --- v5 additions (all with defaults, existing projects stay valid) ---
    stage = models.CharField(max_length=24, blank=True, default='')
    stage_progress = models.FloatField(null=True, blank=True)  # None = unknown
    cancel_requested = models.BooleanField(default=False)
    options = models.JSONField(default=dict, blank=True)  # export/subtitle/scene settings
    transcript_meta = models.JSONField(default=dict, blank=True)
    source_fingerprint = models.CharField(max_length=64, blank=True, default='')
    source_duration = models.FloatField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or f'Project {self.pk}'


class Clip(models.Model):
    project = models.ForeignKey(Project, related_name='clips', on_delete=models.CASCADE)
    start = models.FloatField()
    end = models.FloatField()
    score = models.FloatField(default=0)
    title = models.CharField(max_length=250, blank=True)
    video = models.FileField(upload_to='renders/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # --- v5 additions ---
    status = models.CharField(max_length=16, default='ready')  # candidate/queued/rendering/ready/error/cancelled
    error = models.TextField(blank=True)
    text = models.TextField(blank=True)
    reason = models.TextField(blank=True)
    scores = models.JSONField(default=dict, blank=True)
    quality = models.JSONField(default=dict, blank=True)
    meta = models.JSONField(default=dict, blank=True)  # titles / description / hashtags
    render_info = models.JSONField(default=dict, blank=True)  # scene plan, encoder, settings used
    poster = models.FileField(upload_to='posters/', blank=True)
    position = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
