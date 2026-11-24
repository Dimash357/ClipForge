# 3C — AI Visual & Voice Engine

Локальная генерация изображений и озвучки по раскадровке 3B. Платных API нет; без работающего
провайдера генерация завершается понятной ошибкой — заглушки вместо результата не подставляются.

## Архитектура

| Модуль | Что делает |
|---|---|
| `studio/comfy.py` | HTTP-клиент ComfyUI: проверка (`/system_stats`, `/object_info`), сборка workflow SDXL / SD 1.5, IP-Adapter, `/prompt` → `/history` → `/view`, `/free`, `/interrupt`, распознавание CUDA OOM |
| `studio/tts.py` | Piper (piper-tts 1.2/1.3 API или CLI), локальный OpenAI-совместимый TTS-сервер, импорт своих записей, тайминги слов (Whisper или по энергии сигнала) |
| `studio/genmedia.py` | Настройки (`clipforge_settings.json` → `media`), статус провайдеров, генерация кадров/листов персонажей и реплик, кэш, OOM-ретрай, `fit_timings`, ручная замена |
| `studio/jobs.py` | Задача очереди `media` (resource `heavy`) — прогресс, отмена, сохранение после каждого элемента |
| `studio/storyboard.py` | `apply_to_timeline` — реальные `media:` URI, размеры, длительности озвучки и точные слова субтитров |
| UI | `/generation/` — настройки, проверка, рекомендуемые модели; в раскадровке — кнопки генерации, превью, замена, перегенерация |

Файлы: `MEDIA_ROOT/generated/scripts/<id>/` (`frames/`, `characters/`, `voice/`). Сценарий хранит
метаданные в `data.storyboard` (`frames[].image`, `characters[].sheet`, `voice[scene][line]`).

## Изображения

* SDXL — 768×1344 (≈6.5 ГБ VRAM), SD 1.5 — 512×912 (≈3.5 ГБ). Шаги/CFG/sampler/scheduler в настройках.
* Постоянство персонажей: текст Character Bible в промпте + фиксированный seed кадра + лист персонажа
  (генерируется первым или загружается вручную). При установленном ComfyUI_IPAdapter_plus лист подаётся
  в `IPAdapterUnifiedLoader` + `IPAdapter` (сила делится на √N при нескольких персонажах в кадре).
  Без нод IP-Adapter работает только текст + seed — статус это показывает.
* Последовательность VRAM: выгрузка Ollama (`keep_alive: 0`) → аренда GPU у планировщика (1D) →
  генерация → `/free` в ComfyUI. OOM → `/free` + `resources.on_oom` → один повтор в меньшем размере,
  затем ошибка элемента (остальные продолжаются).

## Озвучка

* Piper: голоса `<имя>.onnx` + `.onnx.json` в папке `models/piper`. Голос персонажа выбирается в
  карточке персонажа, иначе голос по умолчанию для языка сценария.
* OpenAI-совместимый локальный сервер (Kokoro-FastAPI, openedai-speech, AllTalk): URL, модель, голоса.
* После синтеза: длительность из WAV, слова — Whisper (faster-whisper из Stream Clipper) или
  пропорционально внутри речевого интервала. `fit_timings` растягивает сцены под реплики
  (пауза 0.15 с, вход 0.2 с, хвост 0.35 с) и пересчитывает кадры.

## Кэш и возобновление

Ключ элемента = хэш (промпт, модель, размер, seed, референсы / текст, голос, скорость). Совпадает ключ и
файл существует — элемент пропускается. Каждый готовый элемент сразу сохраняется в БД, поэтому
прерванная задача при повторном запуске продолжает с места остановки. Ручные замены (`manual`) никогда
не перезаписываются, пока не нажата перегенерация этого элемента.

## Рекомендуемый набор (RTX 4060 Laptop 8 GB)

| Назначение | Модель | Куда |
|---|---|---|
| SDXL (основной) | Juggernaut XL v9 (~6.5 ГБ) или SDXL base 1.0 | `ComfyUI\models\checkpoints` |
| SD 1.5 (быстрый/экономный) | DreamShaper 8 (~2 ГБ) | `ComfyUI\models\checkpoints` |
| IP-Adapter | `ip-adapter-plus_sdxl_vit-h.safetensors` (SD1.5: `ip-adapter-plus_sd15.safetensors`) | `ComfyUI\models\ipadapter` |
| CLIP Vision | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `ComfyUI\models\clip_vision` |
| Голоса RU | `ru_RU-dmitri-medium`, `ru_RU-irina-medium`, `ru_RU-denis-medium`, `ru_RU-ruslan-medium` | `models\piper` |
| Голоса EN | `en_US-ryan-high`, `en_US-amy-medium` | `models\piper` |

Ничего не скачивается автоматически.

## Установка (Windows)

1. ComfyUI portable (NVIDIA) с github.com/comfyanonymous/ComfyUI → распаковать, запустить
   `run_nvidia_gpu.bat` (по умолчанию http://127.0.0.1:8188). Для 8 ГБ можно добавить `--lowvram`.
2. Положить checkpoint в `ComfyUI\models\checkpoints`.
3. IP-Adapter (опционально): `ComfyUI\custom_nodes` → `git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus`,
   веса в `models\ipadapter` и `models\clip_vision`, перезапустить ComfyUI.
4. Piper в venv проекта:
   ```bat
   env\Scripts\python -m pip install piper-tts
   ```
5. Голоса с huggingface.co/rhasspy/piper-voices (оба файла `.onnx` и `.onnx.json`) → `models\piper`.
6. ClipForge → «Генерация» → «Проверить снова»: должны быть зелёные пункты ComfyUI, модель,
   IP-Adapter, озвучка. Выбрать checkpoint и голоса → «Сохранить».

## Проверка

```bat
env\Scripts\python manage.py test studio.tests.test_genmedia
```
Тесты используют локальный тестовый двойник ComfyUI / TTS (реальные PNG/WAV), проверяют workflow,
кэш, возобновление, IP-Adapter, аренду VRAM, OOM-ретрай, тайминги, отказ без провайдера, API, задачу
очереди и `renderer.preflight` таймлайна. Реальное качество изображений и голосов проверяется только
с установленными ComfyUI/Piper.

## Ограничения

* Время генерации SDXL на 4060 Laptop — порядка 10–25 с на кадр (28 шагов); SD 1.5 быстрее.
* Выравнивание без Whisper приблизительное (пропорционально длине слов).
* Один персонаж на IP-Adapter-ноду; сложные мульти-персонажные сцены лучше править вручную.
* Видео-генерация (img2video) не входит в 3C.
