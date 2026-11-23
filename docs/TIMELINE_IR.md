# Timeline IR 1.0

Единое версионированное описание вертикального видео (клип со стрима, мультфильм, faceless-ролик).
Код: `studio/timeline/`. Основной рендерер Stream Clipper IR не читает — после рендера каждого клипа он
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

# Timeline Renderer (экспериментальный, с 1B)

Код: `studio/timeline/renderer.py` (запуск, кэш, NVENC→CPU), `compile.py` (IR → filtergraph FFmpeg),
`subtitles.py` (текст и субтитры → ASS), `media.py` (безопасные пути и ffprobe).

```python
from studio.timeline import renderer
st = renderer.RenderSettings(base_dir='папка_проекта', quality='balanced', encoder='auto')
rep = renderer.render(tl, 'out.mp4', st)                       # rep['scenes'][i]['cached'] — взята ли сцена из кэша
rep = renderer.render(tl, 'out.mp4', st, rerender={'sc_…'})    # принудительно перерендерить одну сцену
path = renderer.render_scene(tl, 'sc_…', st)                   # только картинка одной сцены (превью)
```

## Как рендерится
1. `ensure_valid` → `check_supported` → проверка путей и ffprobe каждого файла (файл существует, есть нужный
   поток, элемент не читает дальше конца файла). Ошибка — до запуска FFmpeg, с путём к полю IR.
2. **Проход сцены**: видео/оверлей-элементы сцены (раскладки `layouts.py`, zoom/pan, переходы внутри сцены) на
   фоне холста → промежуточный клип без звука в `MEDIA_ROOT/render_cache/scenes/<ключ>.mp4`.
3. **Финальный проход**: клипы сцен на таймлайне (переход между сценами — альфа-фейд следующей сцены поверх
   «хвоста» предыдущей), оверлеи без сцены, один ASS (текст + субтитры), микс всех аудио → mp4 (H.264 + AAC 48 кГц).

Все элементы ставятся по времени таймлайна, поэтому картинка, звук и субтитры синхронны (тест: вспышка и бип
совпадают с точностью до кадра на холстах 640×360 и 540×960).

## Кэш сцен
Ключ = `scene_fingerprint` + переходы, касающиеся сцены + размер и mtime каждого читаемого файла + холст
(размер, fps, фон) + параметры промежуточного кодирования + `RENDERER_VERSION`. Изменили одну сцену или
перезаписали её файл — перерендерится только она. Старые клипы кэша удаляются (LRU), когда кэш больше
`CLIPFORGE_TL_CACHE_GB` (по умолчанию 5 ГБ); удаляются только файлы `<32 hex>.mp4` из `render_cache/scenes`.

## Поддерживается
| Возможность | Детали |
|---|---|
| Холст | любой чётный размер (1080×1920 по умолчанию), fps 1–120, цвет фона |
| Видео | `src_in`, `speed` 0.25–4 (со своим звуком 0.5–2), `volume` (свой звук), все раскладки `LayoutType` |
| Изображения | `fit` cover / contain / fill, PNG с прозрачностью на оверлей-дорожке |
| Камера | анимации `zoom` (≥ 1), `pan_x`, `pan_y` (доля кадра) для видео и изображений, все easing |
| Переходы | `cut`, `fade`, `dissolve` (= кросс-фейд) внутри сцены и между сценами |
| Аудио | роли, `src_in`, `volume` + анимация `volume`, `fade_in/out`, `duck` (sidechain под voice/source), громкость по `ext.clipforge.loudness` (двухпроходный loudnorm, как у классического рендерера) |
| Текст | `x/y`, `align`, анимации `x/y` (одно прямое движение, linear), `scale`, `rotation`, `opacity` (easing linear / ease_in / ease_out / hold) |
| Субтитры | стили Stream Clipper — тем же `captions.build_ass` (идентичные события); прочие стили — из полей `Style`, `emphasis` = `highlight_color` |
| Кодирование | NVENC, если тест-кодирование проходит; при ошибке NVENC — повтор на CPU (libx264) |

## Пока не поддерживается (явная ошибка `UnsupportedFeature`)
Переходы `wipe`, `zoom`, `whoosh`; анимации `x/y/scale/rotation/opacity` у видео и изображений; `zoom` < 1;
`volume`-анимация не у аудио; анимации субтитров; движение текста с > 2 ключами или не linear; easing
`ease_in_out` у текста; неизвестные параметры раскладки; запятые/фигурные скобки в имени шрифта; элементы
видеодорожки без сцены в таймлайне со сценами. `Word.speaker`, `Generation`, `Character` — метаданные, на
картинку не влияют.

## Флаг для Stream Clipper
`CLIPFORGE_TIMELINE_RENDERER=1` (по умолчанию выключен): клип → Timeline IR (`adapters.clip_timeline`) →
новый рендерер. Любая ошибка — лог и рендер классическим `editor.render`. В `render_info`:
`renderer: "timeline"`, `timeline`, `timeline_render` (сцены, кэш, время).

## Команды
```
python manage.py timeline --render-example cartoon --out cartoon.mp4   # локальные заглушки, без скачиваний
python manage.py timeline --render file.json --out out.mp4 --base DIR [--rerender sc_…] [--cpu]
python manage.py timeline --render-clip 12 --out out.mp4               # клип Stream Clipper новым рендерером
```
