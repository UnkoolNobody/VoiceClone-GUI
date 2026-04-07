import sys
import os

# Автоматическое подтверждение лицензии Coqui TTS (иначе программа падает)
os.environ['COQUI_TOS_AGREED'] = '1'

# ----- Определяем базовую директорию (где лежит исполняемый файл) -----
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки для пользовательских файлов (только они создаются рядом с программой)
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REF_SAMPLES_DIR = os.path.join(INPUT_DIR, "reference_samples")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_SAMPLES_DIR, exist_ok=True)

# -------------------- ТЕПЕРЬ МОЖНО ИМПОРТИРОВАТЬ ОСТАЛЬНЫЕ БИБЛИОТЕКИ --------------------
import warnings
import subprocess
import traceback
import threading
import time
import tempfile
import numpy as np
import sounddevice as sd
import wave
import pygame
import torch
import torchaudio
import whisper
import webrtcvad
import soundfile as sf
import TTS.api
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------- Импорты после настройки путей --------------------
warnings.filterwarnings('ignore')
pygame.mixer.init()

# -------------------- Функция проверки FFmpeg --------------------
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "="*60)
        print("РЕШЕНИЕ ПРОБЛЕМЫ С FFMPEG:")
        print("="*60)
        print("1. Установите FFmpeg (full-shared версия) со всеми библиотеками глобально с официального сайта.")
        print("2. Убедитесь, что команда 'ffmpeg' доступна в терминале.")
        print("="*60 + "\n")
        raise RuntimeError("FFmpeg не найден")

# -------------------- Класс распознавания речи (Whisper + VAD + норм., ленивая загрузка) --------------------
class SpeechRecognizer:
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vad = webrtcvad.Vad(2)
        self.target_sr = 16000

    def load_model(self):
        if self.model is not None:
            return
        print(f"Загрузка модели Whisper ({self.model_size})...")
        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
            download_root=None
        )
        print("Модель загружена.")

    def _load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        audio = waveform.numpy().flatten().astype(np.float32)
        return audio, self.target_sr

    def _normalize_loudness(self, audio, target_dbfs=-20):
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        target_rms = 10 ** (target_dbfs / 20)
        gain = target_rms / rms
        audio = audio * gain
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        return audio

    def _vad_trim(self, audio, sr, frame_duration_ms=30, padding_ms=300):
        audio_int16 = (audio * 32767).astype(np.int16)
        frame_size = int(sr * frame_duration_ms / 1000)
        frames = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) == frame_size:
                frames.append(frame)
        speech_flags = []
        for frame in frames:
            is_speech = self.vad.is_speech(frame.tobytes(), sr)
            speech_flags.append(is_speech)
        speech_indices = [i for i, flag in enumerate(speech_flags) if flag]
        if not speech_indices:
            return audio
        start_frame = max(0, speech_indices[0] - padding_ms // frame_duration_ms)
        end_frame = min(len(frames), speech_indices[-1] + padding_ms // frame_duration_ms + 1)
        start_sample = start_frame * frame_size
        end_sample = min(len(audio_int16), end_frame * frame_size)
        trimmed_int16 = audio_int16[start_sample:end_sample]
        trimmed_audio = trimmed_int16.astype(np.float32) / 32767.0
        return trimmed_audio

    def recognize(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        if self.model is None:
            raise RuntimeError("Модель не загружена. Сначала вызовите load_model().")

        audio, sr = self._load_audio(audio_path)
        audio = self._vad_trim(audio, sr)
        if len(audio) < 0.5 * sr:
            print("VAD не обнаружил речи, используется оригинальная запись.")
            audio, sr = self._load_audio(audio_path)
        audio = self._normalize_loudness(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sr, subtype='PCM_16')

        try:
            result = self.model.transcribe(temp_path, language="ru", fp16=torch.cuda.is_available())
            text = result["text"].strip()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return text

# -------------------- Класс синтеза речи (XTTS v2) --------------------
class VoiceCloningSystem:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        check_ffmpeg()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
        self.default_reference = os.path.join(REF_SAMPLES_DIR, "my_voice.wav")
        print("Загрузка модели TTS...")
        try:
            self.tts = TTS.api.TTS(model_name=model_name, progress_bar=True).to(self.device)
            print("Модель успешно загружена!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

    def record_voice_sample(self, filename: str = None, duration: int = 10, sr: int = 16000) -> str:
        if filename is None:
            filename = self.default_reference
        print(f"Запись {duration} секунд...")
        try:
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            for i in range(duration):
                time.sleep(1)
                print(f"Записано: {i+1}/{duration} сек.")
            sd.wait()
            recording = recording.flatten()
            recording_int16 = np.int16(recording * 32767)
            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sr)
                wav_file.writeframes(recording_int16.tobytes())
            print(f"Запись сохранена в: {filename}")
            return self._simple_preprocess_wav(filename)
        except Exception as e:
            print(f"Ошибка при записи: {e}")
            return None

    def _simple_preprocess_wav(self, input_path: str) -> str:
        try:
            with wave.open(input_path, 'r') as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(params.nframes)
                audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            threshold = 0.01
            abs_audio = np.abs(audio_float)
            start = 0
            for i in range(0, len(audio_float), 160):
                if np.max(abs_audio[i:i+160]) > threshold:
                    start = max(0, i - 320)
                    break
            end = len(audio_float)
            for i in range(len(audio_float)-1, 0, -160):
                if np.max(abs_audio[i-160:i]) > threshold:
                    end = min(len(audio_float), i + 320)
                    break
            if end - start > 16000:
                audio_float = audio_float[start:end]
            max_samples = 15 * 16000
            if len(audio_float) > max_samples:
                audio_float = audio_float[:max_samples]
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float / max_val
            audio_int16 = np.int16(audio_float * 32767)
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_processed.wav"
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            return output_path
        except Exception as e:
            print(f"Не удалось обработать аудио: {e}")
            return input_path

    def clone_voice(self, text: str, reference_audio_path: str, output_path: str = "output.wav", language: str = "ru") -> str:
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Файл {reference_audio_path} не найден")
        print("Начинаю синтез...")
        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=language,
                file_path=output_path
            )
        except Exception as e:
            print(f"Ошибка при синтезе: {e}")
            raise
        if os.path.exists(output_path):
            return output_path
        else:
            raise RuntimeError("Файл не был создан")

# -------------------- Класс для записи с микрофона --------------------
class Recorder:
    def __init__(self, entry_widget, var, status_entry, sample_rate=16000, channels=1):
        self.entry = entry_widget
        self.var = var
        self.status_entry = status_entry
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.thread = None

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.status_entry.config(state='normal')
        self.status_entry.delete(0, tk.END)
        self.status_entry.insert(0, "Запись...")
        self.status_entry.config(state='readonly')
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        try:
            def callback(indata, frames, time, status):
                if self.recording:
                    self.audio_data.append(indata.copy())

            self.stream = sd.InputStream(samplerate=self.sample_rate,
                                         channels=self.channels,
                                         callback=callback)
            self.stream.start()
            while self.recording:
                sd.sleep(100)
        except Exception as e:
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, f"Ошибка: {e}")
            self.status_entry.config(state='readonly')
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def stop_recording(self, filename):
        self.recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, filename)
            self.status_entry.config(state='readonly')
            self.var.set(filename)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, filename)
        else:
            self.status_entry.config(state='normal')
            self.status_entry.delete(0, tk.END)
            self.status_entry.insert(0, "Нет данных")
            self.status_entry.config(state='readonly')

# -------------------- Основной GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Программа для распознавания и синтеза речи")
        self.root.geometry("1340x740")
        self.root.minsize(1340, 740)
        self.root.resizable(True, True)

        # Переменные
        self.stt_input_path = tk.StringVar()
        self.stt_output_path = tk.StringVar()
        self.tts_text_file = tk.StringVar()
        self.tts_reference_path = tk.StringVar()
        self.tts_output_path = tk.StringVar()
        self.last_synthesized = tk.StringVar()

        self.stt_status = tk.StringVar(value="Готов")
        self.tts_status = tk.StringVar(value="Готов")

        self.rec_mode = tk.StringVar(value="file")
        self.synth_mode = tk.StringVar(value="text")
        self.ref_mode = tk.StringVar(value="file")

        # Выбор модели Whisper
        self.stt_model_size = tk.StringVar(value="small")

        # Создаём объект STT сразу (без загрузки модели)
        self.stt_engine = SpeechRecognizer(model_size=self.stt_model_size.get())
        self.tts_engine = None

        # Флаги записи
        self.is_recording_stt = False
        self.is_recording_ref = False

        self.create_widgets()

        # Загружаем TTS
        self.loading_frame = ttk.Frame(self.root)
        self.loading_frame.pack(fill=tk.X, pady=5)
        self.loading_label = ttk.Label(self.loading_frame, text="Загрузка модели синтеза речи...")
        self.loading_label.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(self.loading_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress.start()

        threading.Thread(target=self.load_tts_model, daemon=True).start()

        # Отслеживание изменений
        self.stt_input_path.trace_add('write', lambda *a: self.update_buttons_state())
        self.tts_reference_path.trace_add('write', lambda *a: self.update_buttons_state())
        self.rec_mode.trace_add('write', lambda *a: self.update_rec_mode())
        self.synth_mode.trace_add('write', lambda *a: self.update_synth_mode())
        self.ref_mode.trace_add('write', lambda *a: self.update_ref_mode())
        self.stt_model_size.trace_add('write', lambda *a: self.on_model_size_change())

        self.loading_whisper_frame = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Левая колонка: Распознавание речи
        left_frame = ttk.LabelFrame(main_frame, text="Распознавание речи", padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        mode_frame = ttk.Frame(left_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(mode_frame, text="Выбрать файл", variable=self.rec_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Записать файл", variable=self.rec_mode,
                        value="record").pack(side=tk.LEFT, padx=5)

        ttk.Label(left_frame, text="Файл:").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_stt_input = ttk.Entry(left_frame, textvariable=self.stt_input_path, width=50)
        self.entry_stt_input.grid(row=1, column=1, padx=5, pady=2)
        self.stt_browse_btn = ttk.Button(left_frame, text="Обзор",
                                         command=lambda: self.select_file(self.stt_input_path, [("WAV файлы", "*.wav")]))
        self.stt_browse_btn.grid(row=1, column=2)

        ttk.Label(left_frame, text="Запись:").grid(row=2, column=0, sticky="w", pady=2)
        self.record_status_entry = ttk.Entry(left_frame, width=60, state='readonly')
        self.record_status_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.record_btn = ttk.Button(left_frame, text="Записать", command=self.toggle_record)
        self.record_btn.grid(row=2, column=2)

        ttk.Label(left_frame, text="Путь выходного файла:").grid(row=3, column=0, sticky="w", pady=2)
        entry_stt_output = ttk.Entry(left_frame, textvariable=self.stt_output_path, width=50)
        entry_stt_output.grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(left_frame, text="Обзор",
                   command=lambda: self.select_save_file(self.stt_output_path, [("Текстовые файлы", "*.txt")], ".txt")).grid(row=3, column=2)

        # Выбор модели Whisper
        model_frame = ttk.LabelFrame(left_frame, text="Модель Whisper", padding="5")
        model_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        sizes = [("tiny", "tiny"), ("base", "base"), ("small", "small"), ("medium", "medium"), ("large", "large")]
        for i, (label, val) in enumerate(sizes):
            rb = ttk.Radiobutton(model_frame, text=label, variable=self.stt_model_size, value=val)
            rb.grid(row=0, column=i, padx=5, sticky="w")

        self.recognize_btn = ttk.Button(left_frame, text="Распознать", command=self.recognize, state=tk.DISABLED)
        self.recognize_btn.grid(row=5, column=1, pady=10)
        ttk.Label(left_frame, textvariable=self.stt_status).grid(row=5, column=2, sticky="w")

        text_frame = ttk.LabelFrame(left_frame, text="Распознанный текст", padding="5")
        text_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=10)
        self.recognized_text = tk.Text(text_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.recognized_text.pack(fill=tk.BOTH, expand=True)

        left_frame.rowconfigure(6, weight=1)
        left_frame.columnconfigure(1, weight=1)

        # Правая колонка
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Синтез речи
        tts_frame = ttk.LabelFrame(right_frame, text="Синтез речи", padding="10")
        tts_frame.pack(fill=tk.X, pady=5)

        mode_synth_frame = ttk.Frame(tts_frame)
        mode_synth_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(mode_synth_frame, text="Выбрать файл", variable=self.synth_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_synth_frame, text="Ввести текст", variable=self.synth_mode,
                        value="text").pack(side=tk.LEFT, padx=5)

        ttk.Label(tts_frame, text="Файл с текстом:").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_tts_file = ttk.Entry(tts_frame, textvariable=self.tts_text_file, width=50)
        self.entry_tts_file.grid(row=1, column=1, padx=5, pady=2)
        self.tts_browse_btn = ttk.Button(tts_frame, text="Обзор",
                                         command=lambda: self.select_file(self.tts_text_file, [("Текстовые файлы", "*.txt")]))
        self.tts_browse_btn.grid(row=1, column=2)

        ttk.Label(tts_frame, text="Текст:").grid(row=2, column=0, sticky="nw", pady=2)
        self.tts_text_entry = tk.Text(tts_frame, height=18, width=62, state=tk.NORMAL)
        self.tts_text_entry.grid(row=2, column=1, columnspan=2, pady=2, sticky="we")

        # Образец голоса
        ref_frame = ttk.LabelFrame(right_frame, text="Образец голоса", padding="10")
        ref_frame.pack(fill=tk.X, pady=5)

        mode_ref_frame = ttk.Frame(ref_frame)
        mode_ref_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(mode_ref_frame, text="Выбрать файл", variable=self.ref_mode,
                        value="file").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_ref_frame, text="Записать файл", variable=self.ref_mode,
                        value="record").pack(side=tk.LEFT, padx=5)

        ttk.Label(ref_frame, text="Файл:").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_ref = ttk.Entry(ref_frame, textvariable=self.tts_reference_path, width=50)
        self.entry_ref.grid(row=1, column=1, padx=5, pady=2)
        self.ref_browse_btn = ttk.Button(ref_frame, text="Обзор",
                                         command=lambda: self.select_file(self.tts_reference_path, [("WAV файлы", "*.wav")]))
        self.ref_browse_btn.grid(row=1, column=2)

        ttk.Label(ref_frame, text="Запись:").grid(row=2, column=0, sticky="w", pady=2)
        self.ref_record_status_entry = ttk.Entry(ref_frame, width=60, state='readonly')
        self.ref_record_status_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.ref_record_btn = ttk.Button(ref_frame, text="Записать", command=self.toggle_ref_record)
        self.ref_record_btn.grid(row=2, column=2)

        ttk.Label(ref_frame, text="Путь выходного файла:").grid(row=3, column=0, sticky="w", pady=2)
        entry_tts_out = ttk.Entry(ref_frame, textvariable=self.tts_output_path, width=50)
        entry_tts_out.grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(ref_frame, text="Обзор",
                   command=lambda: self.select_save_file(self.tts_output_path, [("WAV файлы", "*.wav")], ".wav")).grid(row=3, column=2)

        self.synthesize_btn = ttk.Button(ref_frame, text="Озвучить", command=self.synthesize, state=tk.DISABLED)
        self.synthesize_btn.grid(row=4, column=1, pady=10)
        ttk.Label(ref_frame, textvariable=self.tts_status).grid(row=4, column=2, sticky="w")

        # Плеер
        player_frame = ttk.LabelFrame(right_frame, text="Пример озвученного файла", padding="5")
        player_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.player_file_label = ttk.Label(player_frame, text="Файл не выбран")
        self.player_file_label.pack(pady=5)

        btn_frame = ttk.Frame(player_frame)
        btn_frame.pack()

        self.play_btn = ttk.Button(btn_frame, text="Воспроизвести", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Остановить", command=self.stop_audio, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.recorder = None
        self.ref_recorder = None

        self.update_rec_mode()
        self.update_synth_mode()
        self.update_ref_mode()

    def update_rec_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.rec_mode.get()
        if mode == "file":
            self.entry_stt_input.config(state=tk.NORMAL)
            self.stt_browse_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL if not self.is_recording_stt else tk.DISABLED)
        else:
            self.entry_stt_input.config(state=tk.DISABLED)
            self.stt_browse_btn.config(state=tk.DISABLED)
            self.record_btn.config(state=tk.NORMAL if not self.is_recording_stt else tk.DISABLED)

    def update_synth_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.synth_mode.get()
        if mode == "file":
            self.entry_tts_file.config(state=tk.NORMAL)
            self.tts_browse_btn.config(state=tk.NORMAL)
            self.tts_text_entry.config(state=tk.DISABLED)
        else:
            self.entry_tts_file.config(state=tk.DISABLED)
            self.tts_browse_btn.config(state=tk.DISABLED)
            self.tts_text_entry.config(state=tk.NORMAL)

    def update_ref_mode(self):
        if self.is_recording_stt or self.is_recording_ref:
            return
        mode = self.ref_mode.get()
        if mode == "file":
            self.entry_ref.config(state=tk.NORMAL)
            self.ref_browse_btn.config(state=tk.NORMAL)
            self.ref_record_btn.config(state=tk.NORMAL if not self.is_recording_ref else tk.DISABLED)
        else:
            self.entry_ref.config(state=tk.DISABLED)
            self.ref_browse_btn.config(state=tk.DISABLED)
            self.ref_record_btn.config(state=tk.NORMAL if not self.is_recording_ref else tk.DISABLED)

    def update_buttons_state(self):
        # STT кнопка
        file_ok = self.stt_input_path.get() and os.path.exists(self.stt_input_path.get())
        if file_ok and not self.is_recording_stt and not self.is_recording_ref:
            self.recognize_btn.config(state=tk.NORMAL)
        else:
            self.recognize_btn.config(state=tk.DISABLED)

        # TTS кнопка
        ref_ok = self.tts_reference_path.get() and os.path.exists(self.tts_reference_path.get())
        text_ok = False
        if self.synth_mode.get() == "file":
            text_ok = self.tts_text_file.get() and os.path.exists(self.tts_text_file.get())
        else:
            text_ok = bool(self.tts_text_entry.get(1.0, tk.END).strip())

        if self.tts_engine and ref_ok and not self.is_recording_stt and not self.is_recording_ref:
            self.synthesize_btn.config(state=tk.NORMAL)
        else:
            self.synthesize_btn.config(state=tk.DISABLED)

    def select_file(self, var, filetypes):
        initial_dir = INPUT_DIR
        if not os.path.exists(initial_dir):
            initial_dir = BASE_DIR
        filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes)
        if filename:
            var.set(filename)
            self.update_buttons_state()

    def select_save_file(self, var, filetypes, defaultextension):
        initial_dir = OUTPUT_DIR
        if not os.path.exists(initial_dir):
            initial_dir = BASE_DIR
        timestamp = int(time.time())
        default_filename = f"output_{timestamp}{defaultextension}"
        filename = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=default_filename,
            filetypes=filetypes,
            defaultextension=defaultextension
        )
        if filename:
            var.set(filename)
            self.update_buttons_state()

    def set_ui_enabled(self, enable, except_rec_stt=False, except_rec_ref=False):
        state = tk.NORMAL if enable else tk.DISABLED

        self.stt_browse_btn.config(state=state)
        self.entry_stt_input.config(state=state)
        self.recognize_btn.config(state=state)
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ttk.LabelFrame):
                        for widget in subchild.winfo_children():
                            if isinstance(widget, ttk.Radiobutton):
                                widget.config(state=state)

        self.tts_browse_btn.config(state=state)
        self.entry_tts_file.config(state=state)
        self.tts_text_entry.config(state=tk.NORMAL if (enable and self.synth_mode.get()=="text") else tk.DISABLED)
        self.ref_browse_btn.config(state=state)
        self.entry_ref.config(state=state)
        self.synthesize_btn.config(state=state)
        self.play_btn.config(state=state)
        self.stop_btn.config(state=state)

        if except_rec_stt:
            self.record_btn.config(state=tk.NORMAL)
        else:
            self.record_btn.config(state=tk.DISABLED if not enable else tk.NORMAL)

        if except_rec_ref:
            self.ref_record_btn.config(state=tk.NORMAL)
        else:
            self.ref_record_btn.config(state=tk.DISABLED if not enable else tk.NORMAL)

    def toggle_record(self):
        if self.is_recording_stt:
            self.recorder.stop_recording(self.current_record_filename)
            self.record_btn.config(text="Записать")
            self.is_recording_stt = False
            self.recorder = None
            if not self.is_recording_ref:
                self.set_ui_enabled(True)
            else:
                self.set_ui_enabled(False, except_rec_ref=True)
            self.update_rec_mode()
            self.update_ref_mode()
            self.update_buttons_state()
        else:
            timestamp = int(time.time())
            self.current_record_filename = os.path.join(REF_SAMPLES_DIR, f"recorded_input_{timestamp}.wav")
            self.recorder = Recorder(self.entry_stt_input, self.stt_input_path, self.record_status_entry)
            self.recorder.start_recording()
            self.record_btn.config(text="Стоп")
            self.is_recording_stt = True
            self.set_ui_enabled(False, except_rec_stt=True)

    def toggle_ref_record(self):
        if self.is_recording_ref:
            self.ref_recorder.stop_recording(self.current_ref_filename)
            self.ref_record_btn.config(text="Записать")
            self.is_recording_ref = False
            self.ref_recorder = None
            if not self.is_recording_stt:
                self.set_ui_enabled(True)
            else:
                self.set_ui_enabled(False, except_rec_stt=True)
            self.update_rec_mode()
            self.update_ref_mode()
            self.update_buttons_state()
        else:
            timestamp = int(time.time())
            self.current_ref_filename = os.path.join(REF_SAMPLES_DIR, f"recorded_reference_{timestamp}.wav")
            self.ref_recorder = Recorder(self.entry_ref, self.tts_reference_path, self.ref_record_status_entry)
            self.ref_recorder.start_recording()
            self.ref_record_btn.config(text="Стоп")
            self.is_recording_ref = True
            self.set_ui_enabled(False, except_rec_ref=True)

    def load_tts_model(self):
        try:
            self.tts_engine = VoiceCloningSystem()
            self.root.after(0, self.tts_loaded)
        except Exception as e:
            traceback.print_exc()
            self.root.after(0, lambda e=e: self.show_error_and_exit(f"Ошибка загрузки TTS:\n{e}"))

    def tts_loaded(self):
        self.progress.stop()
        self.loading_frame.destroy()
        self.update_buttons_state()

    def on_model_size_change(self):
        new_size = self.stt_model_size.get()
        self.stt_engine = SpeechRecognizer(model_size=new_size)

    def show_error_and_exit(self, message):
        messagebox.showerror("Критическая ошибка", message)
        self.root.quit()

    def show_loading_whisper(self, message):
        if self.loading_whisper_frame:
            self.loading_whisper_frame.destroy()
        self.loading_whisper_frame = ttk.Frame(self.root)
        self.loading_whisper_frame.pack(fill=tk.X, pady=5)
        self.loading_whisper_label = ttk.Label(self.loading_whisper_frame, text=message)
        self.loading_whisper_label.pack(side=tk.LEFT, padx=5)
        self.loading_whisper_progress = ttk.Progressbar(self.loading_whisper_frame, mode='indeterminate')
        self.loading_whisper_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.loading_whisper_progress.start()

    def hide_loading_whisper(self):
        if self.loading_whisper_frame:
            self.loading_whisper_progress.stop()
            self.loading_whisper_frame.destroy()
            self.loading_whisper_frame = None

    def recognize(self):
        input_path = self.stt_input_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showwarning("Предупреждение", "Аудиофайл не найден.")
            return

        def task():
            try:
                if self.stt_engine.model is None:
                    self.root.after(0, lambda: self.show_loading_whisper(f"Загрузка модели {self.stt_engine.model_size}..."))
                    self.stt_engine.load_model()
                    self.root.after(0, self.hide_loading_whisper)

                self.stt_status.set("Распознавание...")
                text = self.stt_engine.recognize(input_path)
                self.root.after(0, lambda: self._set_recognized_text(text))
                self.stt_status.set("Готов")
                out_path = self.stt_output_path.get()
                if out_path:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(text)
            except Exception as e:
                self.stt_status.set("Ошибка")
                self.root.after(0, lambda e=e: messagebox.showerror("Ошибка распознавания", str(e)))
                self.root.after(0, self.hide_loading_whisper)

        threading.Thread(target=task, daemon=True).start()

    def _set_recognized_text(self, text):
        self.recognized_text.config(state=tk.NORMAL)
        self.recognized_text.delete(1.0, tk.END)
        self.recognized_text.insert(1.0, text)
        self.recognized_text.config(state=tk.DISABLED)

    def synthesize(self):
        text = ""
        if self.synth_mode.get() == "file":
            file_path = self.tts_text_file.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning("Предупреждение", "Файл с текстом не найден.")
                return
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")
                return
        else:
            text = self.tts_text_entry.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("Предупреждение", "Введите текст для озвучки.")
                return

        ref_path = self.tts_reference_path.get()
        if not ref_path or not os.path.exists(ref_path):
            messagebox.showwarning("Предупреждение", "Образец голоса не найден.")
            return

        out_path = self.tts_output_path.get()
        if not out_path:
            timestamp = int(time.time())
            out_path = os.path.join(OUTPUT_DIR, "output_"+str(timestamp)+".wav")
            self.tts_output_path.set(out_path)

        def task():
            self.tts_status.set("Синтез...")
            try:
                # Останавливаем воспроизведение, если оно идёт
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                # Пытаемся удалить существующий файл, чтобы избежать Permission denied
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except Exception as e_del:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Ошибка", f"Не удалось перезаписать файл {out_path}. Возможно, он открыт другой программой."
                        ))
                        self.tts_status.set("Ошибка")
                        return

                result = self.tts_engine.clone_voice(text, ref_path, out_path, language="ru")
                if result:
                    self.last_synthesized.set(result)
                    self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.player_file_label.config(text=os.path.basename(result)))
                    self.tts_status.set("Готов")
                else:
                    self.tts_status.set("Ошибка")
            except Exception as e:
                self.tts_status.set("Ошибка")
                self.root.after(0, lambda e=e: messagebox.showerror("Ошибка синтеза", str(e)))

        threading.Thread(target=task, daemon=True).start()

    def play_audio(self):
        file = self.last_synthesized.get()
        if not file or not os.path.exists(file):
            messagebox.showinfo("Информация", "Нет файла для воспроизведения.")
            return
        try:
            # Останавливаем текущее воспроизведение перед загрузкой нового
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось воспроизвести: {e}")

    def stop_audio(self):
        pygame.mixer.music.stop()

# -------------------- Запуск --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
