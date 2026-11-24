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
    # --- v5.2: saved source analysis (probe, webcam, language, audio/screen summaries, candidate pool) ---
    analysis = models.JSONField(default=dict, blank=True)

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


class Job(models.Model):
    """One unit of background work in the persistent queue (see studio/jobs.py).

    A project has at most one live job (queued / running / paused): enforced by a partial unique index, so even two
    ClipForge processes cannot queue or run the same project twice. Checkpoints record finished steps; a resumed
    job (after a pause, a crash or a restart) skips them."""
    QUEUED, RUNNING, PAUSED = 'queued', 'running', 'paused'
    COMPLETED, FAILED, CANCELLED, INTERRUPTED = 'completed', 'failed', 'cancelled', 'interrupted'
    LIVE = (QUEUED, RUNNING, PAUSED)
    STATUSES = [(QUEUED, 'В очереди'), (RUNNING, 'Выполняется'), (PAUSED, 'Пауза'), (COMPLETED, 'Готово'),
                (FAILED, 'Ошибка'), (CANCELLED, 'Отменено'), (INTERRUPTED, 'Прервано')]

    kind = models.CharField(max_length=32)
    project = models.ForeignKey(Project, related_name='jobs', null=True, blank=True, on_delete=models.CASCADE)
    title = models.CharField(max_length=250, blank=True)
    payload = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=16, choices=STATUSES, default=QUEUED, db_index=True)
    priority = models.IntegerField(default=0)                 # higher runs first; equal -> oldest first
    resource = models.CharField(max_length=16, default='heavy')   # 'heavy' (GPU/CPU bound) | 'light'
    stage = models.CharField(max_length=24, blank=True, default='')
    stage_progress = models.FloatField(null=True, blank=True)
    progress = models.PositiveIntegerField(default=0)
    message = models.TextField(blank=True)
    error = models.TextField(blank=True)
    checkpoints = models.JSONField(default=dict, blank=True)   # name -> {'at': iso time, ...data}
    attempts = models.PositiveIntegerField(default=0)          # starts (manual retries reset it)
    max_attempts = models.PositiveIntegerField(default=3)      # automatic restarts after crashes stop here
    cancel_requested = models.BooleanField(default=False)
    pause_requested = models.BooleanField(default=False)
    worker = models.CharField(max_length=64, blank=True, default='')   # owner token of the running process/thread
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)   # start of the current run
    finished_at = models.DateTimeField(null=True, blank=True)
    run_seconds = models.FloatField(default=0)                 # processing time of finished runs
    activity = models.JSONField(default=dict, blank=True)     # live operations, ETA, stage timings (UI)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-priority', 'created_at']
        constraints = [models.UniqueConstraint(fields=['project'], condition=models.Q(status__in=('queued', 'running',
                                                                                                   'paused')),
                                               name='one_live_job_per_project')]
        indexes = [models.Index(fields=['status', '-priority', 'created_at'], name='job_pick_idx')]

    def __str__(self):
        return f'{self.kind} #{self.pk} ({self.status})'


class Script(models.Model):
    """AI Script Generator (3A): a script document (see studio/scripts.py) with its generation settings."""
    FORMATS = [('faceless', 'Faceless Shorts'), ('story', 'История'), ('education', 'Обучающее видео'),
               ('cartoon', 'Мультфильм')]
    STATUSES = [('draft', 'Черновик'), ('generating', 'Генерируется'), ('ready', 'Готов'), ('error', 'Ошибка')]

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=250, blank=True)
    idea = models.TextField()
    format = models.CharField(max_length=16, choices=FORMATS, default='faceless')
    language = models.CharField(max_length=8, default='ru')
    duration = models.PositiveIntegerField(default=45)            # target seconds
    genre = models.CharField(max_length=60, blank=True)
    style = models.CharField(max_length=60, blank=True)
    audience = models.CharField(max_length=120, blank=True)
    status = models.CharField(max_length=16, choices=STATUSES, default='draft')
    data = models.JSONField(default=dict, blank=True)              # the script document
    error = models.TextField(blank=True)

    def __str__(self):
        return self.title or f'Script {self.pk}'
