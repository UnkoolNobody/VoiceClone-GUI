```markdown
# VoiceClone-GUI — запуск из исходного кода / Running from source

Это руководство предназначено для запуска программы **VoiceClone-GUI** непосредственно из исходного Python-кода на другом компьютере. Программа выполняет клонирование голоса (XTTS v2) и распознавание речи (Whisper) полностью локально.

This guide explains how to run **VoiceClone-GUI** directly from the Python source code on another computer. The program performs voice cloning (XTTS v2) and speech recognition (Whisper) entirely locally.

---

## Системные требования / System Requirements

- **ОС / OS**: Windows 10/11 (64-bit) — рекомендуется; возможна работа на Linux/macOS с незначительными изменениями.
- **Python**: версия 3.12.6 (или совместимая 3.10–3.12)
- **Процессор / CPU**: Intel Core i5 или аналогичный (многоядерный рекомендуется) / Intel Core i5 or equivalent (multi-core recommended)
- **Оперативная память / RAM**: минимум 8 ГБ (рекомендуется 16 ГБ) / minimum 8 GB (16 GB recommended)
- **Диск / Storage**: ~10 ГБ свободного места для моделей и кэша / ~10 GB free space for models and cache
- **Дополнительно / Additional**: **FFmpeg** (полная shared-сборка / full shared build) – необходима для работы аудио-обработки / required for audio processing.

---

## 1. Установка Python / Installing Python

Скачайте и установите Python 3.12.6 с официального сайта: [python.org](https://www.python.org/downloads/release/python-3126/). При установке обязательно отметьте галочку **«Add Python to PATH»**.

Download and install Python 3.12.6 from the official website: [python.org](https://www.python.org/downloads/release/python-3126/). During installation, make sure to check **"Add Python to PATH"**.

Проверьте установку, открыв терминал (cmd) и выполнив:
```bash
python --version
```
Должно отобразиться `Python 3.12.6`.

Verify the installation by opening a terminal (cmd) and running:
```bash
python --version
```
It should display `Python 3.12.6`.

---

## 2. Получение кода / Getting the Code

Скопируйте все файлы проекта (включая `main.py`) в отдельную папку, например `C:\VoiceClone-GUI_source`. Убедитесь, что у вас есть файл `main.py` (основной скрипт) и, возможно, другие вспомогательные файлы (иконка и т.д.).

Copy all project files (including `main.py`) into a separate folder, e.g., `C:\VoiceClone-GUI_source`. Ensure you have the `main.py` file (the main script) and possibly other auxiliary files (icon, etc.).

---

## 3. Создание виртуального окружения (рекомендуется) / Creating a Virtual Environment (recommended)

Откройте терминал в папке проекта и выполните:

Open a terminal in the project folder and run:

```bash
python -m venv venv
```

Активируйте окружение / Activate the environment:

- **Windows (cmd)**:
  ```bash
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```bash
  venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

После активации в начале строки терминала появится `(venv)`.

After activation, you should see `(venv)` at the beginning of the terminal prompt.

---

## 4. Установка зависимостей / Installing Dependencies

Убедитесь, что pip обновлён / Ensure pip is up to date:

```bash
python -m pip install --upgrade pip
```

Установите необходимые библиотеки (список может быть предоставлен в виде `requirements.txt`). Если файла `requirements.txt` нет, выполните установку вручную:

Install the required libraries (the list may be provided as a `requirements.txt` file). If there is no `requirements.txt`, install them manually:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install whisper
pip install TTS
pip install sounddevice soundfile pygame webrtcvad
pip install numpy scipy matplotlib
pip install transformers huggingface_hub tokenizers safetensors
pip install gruut ko_speech_tools
pip install inflect typeguard
pip install fsspec
pip install torchcodec   # для аудио-обработки (требует FFmpeg)
pip install pysbd        # для сегментации текста
pip install librosa      # для аудио-анализа
pip install scikit-learn # может потребоваться для некоторых функций TTS
```

Обратите внимание: `torchcodec` требует наличия FFmpeg (shared) в системе (см. следующий раздел).

Note: `torchcodec` requires FFmpeg (shared) to be available on the system (see next section).

Если вы хотите точное воспроизведение версий, используйте файл `requirements.txt` со следующим содержимым (пример):

If you need exact version reproduction, use a `requirements.txt` file with the following content (example):

```
torch==2.9.1+cpu
torchaudio==2.9.1+cpu
whisper==1.1.10
TTS==0.22.0
sounddevice==0.5.1
soundfile==0.13.1
pygame==2.6.1
webrtcvad==2.0.10
numpy==2.1.1
scipy==1.17.0
matplotlib==3.10.1
transformers==4.57.0
huggingface_hub==0.36.2
tokenizers==0.22.2
safetensors==0.7.0
gruut==2.3.2
ko_speech_tools==0.2.0
inflect==7.5.0
typeguard==4.4.2
fsspec==2025.3.0
torchcodec==0.1.0
pysbd==0.3.4
librosa==0.11.0
scikit-learn==1.6.1
```

Затем выполните:

Then run:

```bash
pip install -r requirements.txt
```

---

## 5. Установка FFmpeg / Installing FFmpeg

Программа требует наличия полной shared-сборки FFmpeg (с DLL) для работы `torchcodec` и других аудио-операций.

The program requires the full shared build of FFmpeg (with DLLs) for `torchcodec` and other audio operations to work.

### Windows

1. Скачайте **полную shared-сборку FFmpeg** с официального сайта:  
   [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)  
   Выберите **ffmpeg-release-full-shared.7z**.
2. Распакуйте архив. Из папки `bin` скопируйте **все файлы** (в том числе `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swresample-*.dll` и др.) в папку `ffmpeg`, созданную **рядом с вашим скриптом `main.py`**. Если папки `ffmpeg` нет – создайте её вручную.
3. Убедитесь, что в папке `ffmpeg` есть исполняемый файл `ffmpeg.exe` и все необходимые DLL.

Альтернативно, можно добавить путь к `bin` FFmpeg в системную переменную `PATH`, но программа сама добавит локальную папку `ffmpeg` в `PATH` при запуске (как указано в коде).

### Linux / macOS

Установите FFmpeg через пакетный менеджер (например, `sudo apt install ffmpeg` для Ubuntu, `brew install ffmpeg` для macOS). Убедитесь, что команда `ffmpeg` доступна в терминале.

---

## 6. Запуск программы / Running the Program

Убедитесь, что виртуальное окружение активировано. Затем выполните:

Make sure the virtual environment is activated. Then run:

```bash
python main.py
```

При первом запуске программа создаст рядом со скриптом папки:
- `cache` – для хранения моделей и кэша.
- `input` – для входных файлов.
- `output` – для результатов.

При первом использовании клонирования голоса будет автоматически загружена модель XTTS v2 (~1.8 ГБ). При первом распознавании – модель Whisper (размер зависит от выбора). Для загрузки требуется интернет.

On first run, the program will create the following folders next to the script:
- `cache` – for models and cache.
- `input` – for input files.
- `output` – for results.

The first time you use voice cloning, the XTTS v2 model (~1.8 GB) will be downloaded automatically. The first time you perform recognition, the Whisper model (size depends on selection) will be downloaded. Internet connection is required for downloads.

---

## 7. Возможные проблемы и их решение / Troubleshooting

### 7.1. Ошибка «Could not load libtorchcodec» или «FFmpeg not found»
- Убедитесь, что в папке `ffmpeg` рядом со скриптом находятся все DLL из полной shared-сборки FFmpeg.
- Проверьте, что в этой папке есть файлы `avcodec-*.dll`, `avformat-*.dll` и т.д.
- Если вы не хотите использовать локальную папку, добавьте путь к FFmpeg в системную переменную `PATH` и перезапустите терминал.

### 7.2. Ошибка «No module named '...'»
- Убедитесь, что виртуальное окружение активировано.
- Проверьте, что все зависимости установлены (запустите `pip list` и сравните с требуемыми).
- Попробуйте переустановить проблемный пакет: `pip install --upgrade <package>`.

### 7.3. Ошибка «[Errno 2] No such file or directory: '.../mel_filters.npz'»
- Эта ошибка возникает, если whisper не может найти свои файлы данных. Обычно они загружаются автоматически при первом импорте. Попробуйте удалить папку `cache` и запустить программу заново – файлы должны скачаться.

### 7.4. Предупреждения от `inflect` и `typeguard`
- Предупреждения вида `InstrumentationWarning` безопасны и не влияют на работу. Они связаны с декораторами проверки типов. Чтобы их скрыть, можно установить переменную окружения `TYPEGUARD_DISABLE=1` (код уже делает это для собранного EXE, но для исходного кода можно добавить вручную или игнорировать).

### 7.5. Долгая загрузка при первом запуске
- Нормально – модели скачиваются из интернета. При повторных запусках они будут использоваться локально.

---

## Лицензии и благодарности / Licenses and Acknowledgements

Программа использует:
- [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS v2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyTorch](https://pytorch.org/)
- [FFmpeg](https://ffmpeg.org/)

Все компоненты распространяются под своими лицензиями (MIT, Apache 2.0, GPL и др.). Данный код предназначен для запуска в исследовательских и личных целях.

The software uses:
- [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS v2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyTorch](https://pytorch.org/)
- [FFmpeg](https://ffmpeg.org/)

All components are distributed under their respective licenses (MIT, Apache 2.0, GPL, etc.). This code is intended for research and personal use.

---
