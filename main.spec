# -*- mode: python ; coding: utf-8 -*-

import sys
import os
import glob
import tempfile
import importlib.util
sys.setrecursionlimit(10000)

from PyInstaller.utils.hooks import collect_submodules, collect_all, copy_metadata, collect_data_files

def find_dist_info(package_name):
    """Возвращает путь к папке package_name-*.dist-info, если она существует."""
    site_packages = [p for p in sys.path if p.endswith('site-packages') and os.path.isdir(p)]
    for sp in site_packages:
        pattern = os.path.join(sp, f"{package_name}-*.dist-info")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def add_dist_info(package_name, a):
    """Добавляет все файлы из dist-info пакета в a.datas."""
    dist_path = find_dist_info(package_name)
    if dist_path and os.path.isdir(dist_path):
        count = 0
        for root, dirs, files in os.walk(dist_path):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, os.path.dirname(dist_path))
                a.datas.append((rel_path, src, 'DATA'))
                count += 1
        print(f"Метаданные для {package_name} добавлены вручную ({count} файлов) из {dist_path}")
        return True
    else:
        print(f"Предупреждение: не найдена папка dist-info для {package_name}")
        return False

def add_package_subdir(package, subdir, a):
    """Добавляет все файлы из подкаталога subdir пакета package в a.datas."""
    spec = importlib.util.find_spec(package)
    if spec and spec.submodule_search_locations:
        base_path = spec.submodule_search_locations[0]
        target_dir = os.path.join(base_path, subdir)
        if os.path.isdir(target_dir):
            count = 0
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    src = os.path.join(root, file)
                    # Целевой путь внутри сборки: package/subdir/...
                    rel = os.path.relpath(src, os.path.dirname(base_path))
                    a.datas.append((rel, src, 'DATA'))
                    count += 1
            print(f"Добавлены файлы из {target_dir} ({count} файлов)")
        else:
            print(f"Предупреждение: {target_dir} не существует")
    else:
        print(f"Пакет {package} не найден")

def safe_collect_all(package_name, a):
    """Безопасно добавляет все данные из collect_all, нормализуя записи (нормализация будет позже)."""
    try:
        datas, binaries, hiddenimports = collect_all(package_name)
        a.datas.extend(datas)
        a.binaries.extend(binaries)
        a.hiddenimports.extend(hiddenimports)
        print(f"collect_all для {package_name}: {len(datas)} datas, {len(binaries)} binaries, {len(hiddenimports)} hiddenimports")
    except Exception as e:
        print(f"Ошибка при collect_all для {package_name}: {e}")

def normalize_toc(toc_list):
    """Преобразует все двухэлементные кортежи в трёхэлементные (dest, src, typecode)."""
    normalized = []
    for item in toc_list:
        if len(item) == 3:
            normalized.append(item)
        elif len(item) == 2:
            normalized.append((item[0], item[1], 'DATA'))
            print(f"  Нормализована запись: {item}")
        else:
            print(f"Предупреждение: некорректная запись длины {len(item)}: {item}")
    return normalized

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],  # стандартные хуки, без кастомных
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'gevent', 'gevent.*', 'zope.interface', 'zope.event',
        'IPython', 'jedi', 'parso', 'zmq',
        'pandas', 'tensorflow', 'tensorflow.*', 'keras', 'jax', 'flax',
        'notebook', 'jupyter_client', 'jupyter_core', 'ipykernel',
        'PyQt5', 'PySide2', 'PyQt6', 'PySide6',
        'ipywidgets', 'qtpy', 'qtconsole',
        'torchvision', 'torchvision.*',
    ],
    noarchive=False,
)

# ----- Сбор подмодулей основных пакетов -----
a.hiddenimports += collect_submodules('TTS')
a.hiddenimports += collect_submodules('whisper')
a.hiddenimports += collect_submodules('scipy')
a.hiddenimports += collect_submodules('matplotlib')
a.hiddenimports += collect_submodules('torchcodec')

# ----- Обработка gruut (фонематизатор) -----
a.hiddenimports += collect_submodules('gruut')
a.datas += collect_data_files('gruut')
safe_collect_all('gruut', a)
add_package_subdir('gruut', '', a)

# ----- Обработка ko_speech_tools -----
a.hiddenimports += collect_submodules('ko_speech_tools')
a.hiddenimports += collect_submodules('ko_speech_tools.data')
add_package_subdir('ko_speech_tools', 'data', a)
a.datas += collect_data_files('ko_speech_tools')
safe_collect_all('ko_speech_tools', a)

# ----- Обработка whisper (добавляем файлы данных) -----
a.hiddenimports += collect_submodules('whisper')
a.datas += collect_data_files('whisper')
safe_collect_all('whisper', a)
add_package_subdir('whisper', 'assets', a)  # гарантирует включение mel_filters.npz

# ----- Дополнительные подмодули TTS, которые могут не попасть автоматически -----
a.hiddenimports += collect_submodules('TTS.vocoder.datasets')
a.hiddenimports += collect_submodules('TTS.vocoder.layers')
a.hiddenimports += collect_submodules('TTS.vocoder.models')
a.hiddenimports += collect_submodules('TTS.vocoder.configs')
a.hiddenimports += collect_submodules('TTS.vocoder.utils')
a.hiddenimports += collect_submodules('TTS.tts.datasets')
a.hiddenimports += collect_submodules('TTS.tts.layers')
a.hiddenimports += collect_submodules('TTS.tts.models')
a.hiddenimports += collect_submodules('TTS.tts.utils')

# ----- Ручное добавление всех файлов из критических подкаталогов TTS -----
add_package_subdir('TTS', 'vocoder/configs', a)
add_package_subdir('TTS', 'vocoder/models', a)
add_package_subdir('TTS', 'vocoder/datasets', a)
add_package_subdir('TTS', 'vocoder/layers', a)
add_package_subdir('TTS', 'vocoder/utils', a)
add_package_subdir('TTS', 'tts/configs', a)
add_package_subdir('TTS', 'tts/models', a)
add_package_subdir('TTS', 'tts/layers', a)
add_package_subdir('TTS', 'tts/utils', a)

# ----- Добавляем все data-файлы для TTS (включая конфиги) -----
a.datas += collect_data_files('TTS')
print("Data-файлы TTS добавлены через collect_data_files")

# ----- Добавляем .models.json (или фиктивный) -----
models_json_added = False
for sp in [p for p in sys.path if p.endswith('site-packages') and os.path.isdir(p)]:
    candidate = os.path.join(sp, 'TTS', '.models.json')
    if os.path.isfile(candidate):
        a.datas.append(('TTS/.models.json', candidate, 'DATA'))
        print(f"Найден и добавлен реальный .models.json: {candidate}")
        models_json_added = True
        break
if not models_json_added:
    fake_json_dir = os.path.join(tempfile.gettempdir(), 'TTS_fake')
    os.makedirs(fake_json_dir, exist_ok=True)
    fake_json = os.path.join(fake_json_dir, '.models.json')
    with open(fake_json, 'w', encoding='utf-8') as f:
        f.write('{}')
    a.datas.append(('TTS/.models.json', fake_json, 'DATA'))
    print("Предупреждение: реальный .models.json не найден, добавлен фиктивный.")

# ----- Добавляем torchcodec полностью -----
safe_collect_all('torchcodec', a)
add_dist_info('torchcodec', a)

# ----- Дополнительно: добавляем все DLL из папки torchcodec с сохранением подпути -----
try:
    torchcodec_spec = importlib.util.find_spec('torchcodec')
    if torchcodec_spec and torchcodec_spec.origin:
        torchcodec_dir = os.path.dirname(torchcodec_spec.origin)
        for root, dirs, files in os.walk(torchcodec_dir):
            for file in files:
                if file.endswith('.dll') or file.endswith('.pyd'):
                    src = os.path.join(root, file)
                    # Относительный путь от site-packages
                    rel = os.path.relpath(src, os.path.dirname(torchcodec_spec.origin))
                    # Целевой путь внутри сборки: torchcodec/...
                    dest = os.path.join('torchcodec', rel).replace('\\', '/')
                    a.binaries.append((dest, src, 'BINARY'))
                    print(f"Добавлена библиотека torchcodec: {dest}")
except Exception as e:
    print(f"Предупреждение: не удалось добавить нативные библиотеки torchcodec: {e}")

# ----- Добавляем метаданные dist-info для всех критических пакетов вручную -----
critical_packages = [
    'tqdm',
    'regex',
    'transformers',
    'torch',
    'huggingface_hub',
    'safetensors',
    'tokenizers',
    'packaging',
    'requests',
    'filelock',
    'pyyaml',
    'numpy',
    'scipy',
    'torchcodec',
]

for pkg in critical_packages:
    add_dist_info(pkg, a)

# ----- Полный сбор данных для ключевых пакетов через collect_all -----
safe_collect_all('transformers', a)
safe_collect_all('TTS', a)
safe_collect_all('whisper', a)
safe_collect_all('scipy', a)
safe_collect_all('torch', a)
safe_collect_all('torchaudio', a)
safe_collect_all('librosa', a)
safe_collect_all('soundfile', a)
safe_collect_all('matplotlib', a)

# ----- Явные скрытые импорты (дополнительная подстраховка) -----
a.hiddenimports += [
    'scipy._cyutility',
    'scipy._lib._ccallback_c',
    'array_api_compat',
    'array_api_compat.numpy',
    'array_api_compat.numpy.fft',
    'array_api_compat.numpy.linalg',
    'array_api_compat.torch',
    'pkg_resources',
    'importlib.metadata',
    'importlib_metadata',
    'webrtcvad',
    'sounddevice',
    'pygame',
    'soundfile',
    'torchaudio',
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.utils',
    'torch.cuda',
    'torch.backends',
    'torchaudio.datasets',
    'torchaudio.functional',
    'torchaudio.transforms',
    'pygame.mixer',
    'pygame.midi',
    'pygame.base',
    'pygame.constants',
    'pygame.bufferproxy',
    'pygame.rect',
    'pygame.surface',
    'pygame.display',
    'pygame.event',
    'pygame.key',
    'pygame.mouse',
    'pygame.image',
    'pygame.font',
    'pygame.time',
    'pygame.music',
    'pygame.sndarray',
    'sounddevice',
    '_sounddevice_data',
    'numpy.core',
    'numpy.fft',
    'numpy.linalg',
    'numpy.random',
    'huggingface_hub.snapshot_download',
    'TTS.vocoder.layers',
    'TTS.vocoder.layers.wavegrad',
    'TTS.vocoder.utils',
    'TTS.vocoder.utils.distribution',
    'gruut',
    'gruut.lang',
    'matplotlib',
    'matplotlib.backends.backend_agg',
]

# Иконка
if os.path.exists('icon.ico'):
    a.datas.append(('icon.ico', 'icon.ico', 'DATA'))

# Нормализуем datas и binaries перед сборкой
a.datas = normalize_toc(a.datas)
a.binaries = normalize_toc(a.binaries)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VoiceClone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)