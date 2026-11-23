# Timeline IR 1.0

Единое версионированное описание вертикального видео (клип со стрима, мультфильм, faceless-ролик).
Код: `studio/timeline/`. Рендерер пока не читает IR — после рендера каждого клипа Stream Clipper
сохраняет снимок в `Clip.render_info['timeline']` (постепенная интеграция).

## Структура
```
Timeline {schema:"clipforge.timeline", version:"1.0", id, kind, duration, canvas,
          assets[], scenes[], tracks[], styles[], characters[], meta{}, ext{}}
Track    {id, type: video|overlay|audio|caption, items[]}
Item     type = video | image | text | audio | caption | transition
           общие поля: id, start, duration, scene, animations[], ext
```
* Время — секунды на таймлайне; `Keyframe.t` — относительно начала элемента; `src_in` — позиция в ассете.
* Ассеты по ID, `uri`: `media:` (MEDIA_ROOT), `file:`, `gen:` (результат генерации, см. `Asset.generation`).
* Переход: элементы `from_item`/`to_item` на одной дорожке перекрываются ровно на `duration` (cut — встык).
* Сторонние данные — только в `ext` (`{"мой.плагин": {...}}`); неизвестные поля — ошибка.

## Стабильные ID и перегенерация сцен
* `stable_id(prefix, *parts)` → `sc_<sha1[:16]>`: одна и та же сцена клипа всегда получает тот же ID.
* `scene_fingerprint(tl, scene_id)` — хэш сцены, её элементов и используемых ассетов/стилей/персонажей.
  Отпечаток не изменился → рендер сцены можно переиспользовать; перегенерировать только изменённые.
* `content_hash(tl)` — хэш всего таймлайна без `meta`.

## API
```python
from studio import timeline as T
tl = T.from_json(text)          # строгое чтение + миграции старых версий
issues = T.validate(tl)         # [{'path','message','level'}], [] = ок
T.ensure_valid(tl)              # или TimelineError с .issues
T.to_json(tl)                   # канонический, побайтно стабильный JSON
T.json_schema()                 # JSON Schema 2020-12 (редакторы, structured output LLM)
```
Новая версия формата: поднять `model.VERSION` и добавить `@T.register_migration('1.0', '1.1')`.
Документ более новой версии, чем знает ClipForge, отклоняется.

## Команда
```
python manage.py timeline --clip 12
python manage.py timeline --validate file.json
python manage.py timeline --schema
python manage.py timeline --example cartoon|faceless
```
