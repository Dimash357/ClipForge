# Очередь задач (1C)

Код: `studio/jobs.py`, модель `Job` (`studio/models.py`, миграция `0005_job_queue`), страница `/queue/`.
Django + SQLite, без Redis/Celery и внешних сервисов.

## Статусы задачи
`queued` → `running` → `completed` | `failed` | `cancelled`; `paused` (пауза на безопасной точке);
`interrupted` (процесс умер во время работы).

## Как это работает
* Задачи хранятся в таблице `studio_job`: очередь переживает закрытие приложения, перезагрузку Windows и сбой.
* Рабочие потоки забирают задачу одним атомарным `UPDATE … WHERE status='queued'` — одну задачу не запустят дважды,
  даже два процесса ClipForge. В том же запросе проверяется лимит тяжёлых задач (`CLIPFORGE_MAX_JOBS`, по умолчанию 1).
* У проекта не больше одной живой задачи (queued/running/paused): частичный уникальный индекс в базе.
* Работающая задача пишет heartbeat каждые 5 с. Если он старше `JOB_STALE_SECONDS` (30 с) — задача
  `interrupted` и, пока `attempts < max_attempts` (3), снова ставится в очередь автоматически. Ручной «Повторить»
  сбрасывает счётчик — бесконечных перезапусков нет.
* Контрольные точки (`Reporter.checkpoint`) пишутся в задачу. Stream Clipper отмечает: `source`, `transcript`,
  `clips` (выбранные фрагменты), `rendered` (каждый готовый клип). Продолженная задача не скачивает, не
  распознаёт и не подбирает моменты заново и не перерендеривает готовые клипы.
* Пауза срабатывает только на безопасной точке (начало этапа, контрольная точка — например, между клипами).
  Отмена — сразу (FFmpeg останавливается). Запросы паузы/отмены из другого процесса подхватываются heartbeat'ом.
* Блокировки SQLite: WAL + `busy_timeout`, плюс повтор записи при `database is locked`.
* Перед миграциями база копируется в `backups/` (`start.bat`/`setup.bat`, а также при ручном `manage.py migrate`).

## Новые виды задач
```python
from studio import jobs
def run_cartoon(job, rep):           # rep: stage(), frac(), check(), checkpoint(), done(), get()
    if not rep.done('script'):
        ...; rep.checkpoint('script', path='...')
jobs.register('cartoon', run_cartoon, label='Мультфильм', resource='heavy', needs_project=False)
jobs.submit('cartoon', priority=5, title='Лис и сова', idea='...')
```
Уже зарегистрированы: `full`, `resume`, `reprocess`, `render` (Stream Clipper) и `timeline_render` (рендер
Timeline IR в `media/renders/timeline/`). Поле `resource` — задел для GPU-планировщика (1D).

## Настройки (переменные окружения / settings)
| | по умолчанию |
|---|---|
| `CLIPFORGE_MAX_JOBS` | 1 тяжёлая задача одновременно |
| `CLIPFORGE_JOB_STALE_SECONDS` | 30 |
| `CLIPFORGE_JOB_MAX_ATTEMPTS` | 3 |
| `CLIPFORGE_JOB_AUTO_RESUME` | 1 (продолжать прерванные задачи после перезапуска; 0 — выкл.) |
