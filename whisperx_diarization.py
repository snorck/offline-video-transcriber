#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎙️ Распознавание речи с диаризацией через WhisperX (Docker) 🎙️ 

Этот Python-скрипт является оркестратором для пакетной обработки аудиофайлов
с использованием WhisperX. Он запускает транскрибацию и диаризацию (разделение
по спикерам) в изолированном Docker-контейнере, что решает проблемы
совместимости и обеспечивает стабильную работу на системах с GPU NVIDIA.

Напоминание: для работы диаризации требуется токен Hugging Face (HF_TOKEN)
и принятие лицензий для моделей pyannote.

Основные задачи:
- Изоляция зависимостей: Использует готовый Docker-образ, избавляя от ручной
  установки PyTorch, CUDA и других сложных компонентов.
- Поддержка GPU и CPU: Автоматически задействует GPU NVIDIA для максимального
  ускорения и может работать в режиме CPU.
- Пакетная обработка: Обрабатывает как отдельные файлы, так и все аудио
  в указанной директории (mp3, wav, m4a и др.).
- Централизованный кеш: Сохраняет скачанные модели в общей папке `~/.whisperx/`,
  экономя дисковое пространство и время при повторных запусках.
- Гибкая конфигурация: Управляет параметрами (модель, язык, токен) через
  внешний файл `config.env`.
- Информативный вывод: Отображает детальный прогресс и итоговую статистику,
  включая скорость обработки относительно реального времени.
- Встроенная проверка системы: Команда `--check` позволяет быстро убедиться,
  что Docker, GPU и права доступа настроены корректно.

Порядок использования:
1. Отредактируйте файл `config.env`, указав ваш HF_TOKEN.
2. Поместите аудиофайлы в папку `audio/`.
3. Запустите скрипт:
   python3 whisperx_diarization.py
4. Результаты (txt, srt, json) появятся в папке `results/`.

Автор: Михаил Шардин https://shardin.name/
Дата создания: 13.09.2025
Версия: 2.1

Актуальная версия скрипта всегда здесь: https://github.com/empenoso/offline-audio-transcriber
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime
import shutil
import itertools

# Определяем базовую директорию относительно местоположения скрипта
SCRIPT_DIR = Path(__file__).parent.resolve()

# Определяем глобальную директорию для кеша моделей в домашней папке пользователя
USER_CACHE_DIR = Path.home() / 'whisperx'

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'whisperx_diarization.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI цвета для консольного вывода"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'

class WhisperXDocker:
    """Класс для работы с WhisperX через Docker"""

    def __init__(self, config_path: str = "config.env"):
        self.work_dir = SCRIPT_DIR
        # ИЗМЕНЕНИЕ: Ссылка на глобальный кеш
        self.cache_dir = USER_CACHE_DIR
        self.config_path = self.work_dir / config_path
        self.config = self._load_config()
        self.image_name = "ghcr.io/jim60105/whisperx:latest"
        self.use_gpu = self.config.get('DEVICE') == 'cuda'
        self._ensure_directories()

    def _load_config(self) -> Dict[str, str]:
        """Загружает конфигурацию из файла .env"""
        config = {
            'HF_TOKEN': '', 'WHISPER_MODEL': 'large-v3', 'LANGUAGE': 'ru',
            'BATCH_SIZE': '16', 'DEVICE': 'cuda', 'ENABLE_DIARIZATION': 'true',
            'MIN_SPEAKERS': '', 'MAX_SPEAKERS': '', 'COMPUTE_TYPE': 'float16',
            'VAD_METHOD': 'pyannote', 'CHUNK_SIZE': '30'
        }
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip().strip('"\'')
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
                logger.info("Создание файла конфигурации по умолчанию...")
                self._create_default_config()
        else:
            logger.info("Файл конфигурации не найден. Создание по умолчанию...")
            self._create_default_config()
        return config

    def _create_default_config(self):
        """Создает файл конфигурации по умолчанию"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = """# Конфигурация WhisperX
# HuggingFace токен для диаризации (получите на https://huggingface.co/settings/tokens)
# ВАЖНО: Примите лицензии на:
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0
HF_TOKEN=your_token_here

# Модель Whisper (tiny, base, small, medium, large-v1, large-v2, large-v3)
WHISPER_MODEL=large-v3

# Язык аудио (ru, en, auto для автоопределения)
LANGUAGE=ru

# Размер батча (чем больше - тем быстрее, но больше памяти GPU)
BATCH_SIZE=16

# Устройство для вычислений (cuda или cpu)
DEVICE=cuda

# Включить диаризацию (разделение по спикерам)
ENABLE_DIARIZATION=true

# Минимальное количество спикеров (оставить пустым для автоопределения)
MIN_SPEAKERS=

# Максимальное количество спикеров (оставить пустым для автоопределения)
MAX_SPEAKERS=

# Тип вычислений (float16, float32, int8)
COMPUTE_TYPE=float16

# Метод VAD для обнаружения речи (pyannote, silero)
VAD_METHOD=pyannote

# Размер чанков в секундах
CHUNK_SIZE=30
"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(default_config)
        logger.info(f"Создан файл конфигурации: {self.config_path}")

    def _ensure_directories(self):
        """Создает необходимые рабочие директории"""
        # Создаем локальные папки audio и results
        for dir_name in ['audio', 'results']:
            p = self.work_dir / dir_name
            p.mkdir(parents=True, exist_ok=True)
        
        # Создаем глобальную папку для кеша моделей, если ее нет
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Кеш моделей будет сохраняться в: {self.cache_dir}")


    def _run_command(self, cmd: List[str], timeout: int = 45) -> Optional[subprocess.CompletedProcess]:
        """Унифицированная функция для запуска внешних команд"""
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True, encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"Команда '{cmd[0]}' не найдена.")
        except subprocess.TimeoutExpired:
            logger.error(f"Команда '{' '.join(cmd)}' заняла слишком много времени.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка выполнения команды. Код: {e.returncode}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr.strip()}")
        return None

    def _check_gpu(self) -> bool:
        """Проверяет доступность GPU через Docker"""
        logger.info("Проверка доступа к GPU из Docker...")
        cmd = [
            'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
            'nvidia/cuda:12.4.1-base-ubuntu22.04',
            'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
        ]
        result = self._run_command(cmd)
        if result and result.stdout.strip():
            logger.info(f"✅ GPU успешно обнаружен: {result.stdout.strip()}")
            return True
        return False

    def _format_time(self, seconds: float) -> str:
        """Форматирует секунды в читаемый вид (ч:м:с)"""
        if seconds < 0: return "0.0с"
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{int(hours)}ч {int(mins)}м {int(secs)}с"
        elif mins > 0:
            return f"{int(mins)}м {int(secs)}с"
        else:
            return f"{secs:.1f}с"

    def _get_audio_duration(self, file_path: Path) -> Optional[float]:
        """Получает длительность аудиофайла через ffprobe"""
        if not shutil.which('ffprobe'):
            return None
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)
        ]
        result = self._run_command(cmd, timeout=15)
        try:
            return float(result.stdout.strip()) if result and result.stdout.strip() else None
        except (ValueError, AttributeError):
            return None

    def list_audio_files(self, directory: Optional[Path] = None) -> List[Path]:
        """Находит все поддерживаемые аудиофайлы в директории"""
        directory = directory or self.work_dir / "audio"
        extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.mp4', '.mkv', '.avi']
        return sorted([p for p in directory.rglob('*') if p.suffix.lower() in extensions and p.is_file()])

    def process_file(self, audio_file: Path, output_dir: Optional[Path] = None) -> bool:
        """Обрабатывает один аудиофайл с помощью WhisperX в Docker"""
        output_dir = output_dir or self.work_dir / "results"
        file_output_dir = output_dir / audio_file.stem
        file_output_dir.mkdir(exist_ok=True)

        cmd = ['sudo', 'docker', 'run', '--rm', '--user', f"{os.getuid()}:{os.getgid()}"]

        if self.use_gpu:
            cmd.extend(['--gpus', 'all'])

        cmd.extend([
            '-v', f"{audio_file.parent.resolve()}:/audio:ro",
            '-v', f"{file_output_dir.resolve()}:/results",
            # ИЗМЕНЕНИЕ: Монтируем глобальную директорию кеша в /models внутри контейнера
            '-v', f"{self.cache_dir.resolve()}:/models",
            '--workdir', '/app',
            # ИЗМЕНЕНИЕ: Все пути кеша внутри контейнера теперь указывают на смонтированный том /models
            '-e', 'HOME=/models',
            '-e', 'HF_HOME=/models/.cache/huggingface',
            '-e', 'XDG_CACHE_HOME=/models/.cache',
            '-e', 'TORCH_HOME=/models/.cache/torch',
        ])

        hf_token = self.config.get('HF_TOKEN', '').strip()
        if hf_token and hf_token != 'your_token_here':
            cmd.extend(['-e', f"HF_TOKEN={hf_token}"])
            logger.info("✅ HF_TOKEN передан в контейнер")
        else:
            logger.warning(f"{Colors.YELLOW}⚠️ HF_TOKEN не настроен! Диализация может не работать.{Colors.NC}")

        cmd.extend([self.image_name, 'whisperx'])
        whisper_args = [
            '--output_dir', "/results",
            '--model', self.config.get('WHISPER_MODEL', 'large-v3'),
            '--language', self.config.get('LANGUAGE', 'ru'),
            '--batch_size', self.config.get('BATCH_SIZE', '16'),
            '--device', 'cuda' if self.use_gpu else 'cpu',
            '--compute_type', self.config.get('COMPUTE_TYPE', 'float16'),
            '--output_format', 'all',
            '--verbose', 'False'
        ]

        if (self.config.get('ENABLE_DIARIZATION', 'true').lower() == 'true' and hf_token and hf_token != 'your_token_here'):
            whisper_args.extend(['--diarize', '--hf_token', hf_token])
            for key, name in [('MIN_SPEAKERS', '--min_speakers'), ('MAX_SPEAKERS', '--max_speakers')]:
                value = self.config.get(key)
                if value and value.isdigit() and int(value) > 0:
                    whisper_args.extend([name, value])
        elif self.config.get('ENABLE_DIARIZATION', 'true').lower() == 'true':
            logger.warning("⚠️ Диаризация отключена - нет HF_TOKEN")

        whisper_args.append(f"/audio/{audio_file.name}")
        cmd.extend(whisper_args)

        duration = self._get_audio_duration(audio_file)
        logger.info(f"{Colors.CYAN}🎵 Обрабатываем: {audio_file.name}{Colors.NC}")
        logger.info(f"   📊 Размер: {audio_file.stat().st_size / (1024*1024):.1f} МБ")
        if duration:
            logger.info(f"   ⏱️  Длительность: {self._format_time(duration)}")
        logger.info(f"   📁 Результаты: {file_output_dir}")

        start_time = time.time()
        logger.info(f"{Colors.YELLOW}🚀 Запуск WhisperX...{Colors.NC}")

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding='utf-8', bufsize=1, universal_newlines=True)
            
            spinner = itertools.cycle(['⠇', '⠏', '⠋', '⠙', '⠸', '⠴', '⠦', '⠇'])
            stderr_lines = []
            
            current_status = "Инициализация..."
            sys.stdout.write(f"   [PROGRESS] {next(spinner)} {current_status}\r")

            while process.poll() is None:
                line = process.stderr.readline() 
                if line:
                    stderr_lines.append(line.strip())
                    if "Performing VAD" in line or "voice activity detection" in line:
                        current_status = "1/4 Обнаружение речи (VAD)..."
                    elif "Performing transcription" in line:
                        current_status = "2/4 Транскрибация текста..."
                    elif "Performing alignment" in line:
                        current_status = "3/4 Выравнивание временных меток..."
                    elif "Performing diarization" in line:
                        current_status = f"4/4 Диарізація (може зайняти багато часу)..."
                    
                sys.stdout.write(f"   [PROGRESS] {next(spinner)} {current_status}\r")
                sys.stdout.flush()
                time.sleep(0.1)

            sys.stdout.write(" " * (len(current_status) + 20) + "\r")
            sys.stdout.flush()

            process.wait()

            if process.returncode == 0:
                processing_time = time.time() - start_time
                logger.info(f"{Colors.GREEN}✅ Обработка завершена успешно!{Colors.NC}")
                logger.info(f"   ⏱️  Время обработки: {self._format_time(processing_time)}")
                if duration and processing_time > 0:
                    speed_factor = duration / processing_time
                    logger.info(f"   🚀 Скорость: {speed_factor:.1f}x от реального времени")
                
                result_files = list(file_output_dir.glob('*'))
                if result_files:
                    logger.info(f"   📄 Создано файлов: {len(result_files)}")
                    for rf in sorted(result_files):
                        logger.info(f"      • {rf.name}")
                return True
            else:
                logger.error(f"{Colors.RED}❌ Ошибка обработки файла {audio_file.name}{Colors.NC}")
                logger.error(f"   Код возврата Docker: {process.returncode}")
                if stderr_lines:
                    logger.error("   Последние сообщения из лога контейнера:")
                    for line in stderr_lines[-10:]:
                        if line.strip():
                            logger.error(f"   [Docker ERR]: {line}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при запуске Docker: {e}")
            return False

    def process_directory(self, input_dir: Optional[Path] = None):
        """Обрабатывает все аудиофайлы в директории"""
        audio_files = self.list_audio_files(input_dir)
        if not audio_files:
            logger.warning(f"В директории {input_dir or self.work_dir / 'audio'} не найдено аудиофайлов.")
            return

        logger.info(f"{Colors.CYAN}📁 Найдено {len(audio_files)} аудиофайлов{Colors.NC}")
        stats = {"total": len(audio_files), "success": 0, "failed": 0}
        start_time = time.time()

        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n{Colors.WHITE}═══ Файл {i}/{stats['total']} ═══{Colors.NC}")
            if self.process_file(audio_file):
                stats["success"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"\n{Colors.WHITE}{'═'*35}{Colors.NC}")
        logger.info(f"{Colors.GREEN}🎯 ИТОГИ ОБРАБОТКИ{Colors.NC}")
        logger.info(f"{Colors.WHITE}{'═'*35}{Colors.NC}")
        logger.info(f"📊 Всего файлов: {stats['total']}")
        logger.info(f"✅ Успешно: {stats['success']}")
        logger.info(f"❌ С ошибками: {stats['failed']}")
        total_time = time.time() - start_time
        logger.info(f"⏱️  Общее время: {self._format_time(total_time)}")
        if stats['total'] > 0:
            logger.info(f"📈 Среднее время на файл: {self._format_time(total_time / stats['total'])}")

    def check_system(self) -> bool:
        logger.info(f"{Colors.CYAN}🔍 Проверка системы...{Colors.NC}")
        
        if not self._run_command(['docker', '--version']):
            logger.error("❌ Docker не найден. Установите Docker."); return False
        logger.info("✅ Docker найден")

        if self.use_gpu:
            if not self._check_gpu():
                logger.warning("⚠️ GPU недоступен через Docker, переключаемся на CPU.")
                self.use_gpu = False; self.config['DEVICE'] = 'cpu'
            else: logger.info("✅ GPU-ускорение активно")
        else: logger.info("⚙️  Режим CPU активен (согласно config.env)")

        if not self._run_command(['sudo', 'docker', 'image', 'inspect', self.image_name]):
            logger.error(f"❌ Образ WhisperX не найден. Выполните: sudo docker pull {self.image_name}"); return False
        logger.info("✅ Образ WhisperX найден")
        
        hf_token = self.config.get('HF_TOKEN', '').strip()
        if self.config.get('ENABLE_DIARIZATION', 'true').lower() == 'true':
            if not hf_token or hf_token == 'your_token_here':
                logger.error(f"❌ HF_TOKEN не настроен в {self.config_path}")
                logger.info("💡 Получите токен на https://huggingface.co/settings/tokens и примите лицензии."); return False
            else: logger.info("✅ HF_TOKEN настроен")

        if not shutil.which('ffprobe'):
            logger.warning("⚠️ ffprobe не найден. Длительность аудио не будет отображаться. (sudo apt install ffmpeg)")

        #Проверяем права на запись в глобальный кеш, а не в локальную папку
        try:
            test_file = self.cache_dir / 'test_write.tmp'
            test_file.touch(); test_file.unlink()
            logger.info(f"✅ Есть права на запись в кеш {self.cache_dir}")
        except Exception as e:
            logger.error(f"❌ Нет прав на запись в директорию кеша {self.cache_dir}: {e}"); return False
            
        logger.info(f"{Colors.GREEN}✅ Система готова к работе{Colors.NC}")
        return True


def main():
    """Главная функция-обработчик CLI"""
    parser = argparse.ArgumentParser(description='🎙️ Распознавание речи с диаризацией через WhisperX (DOCKER)')
    parser.add_argument('-f', '--file', type=str, help='Путь к конкретному аудиофайлу для обработки')
    parser.add_argument('-d', '--directory', type=str, help='Путь к директории с аудиофайлами')
    parser.add_argument('--check', action='store_true', help='Проверить готовность системы к работе')
    parser.add_argument('--config', type=str, default="config.env", help='Путь к файлу конфигурации относительно скрипта')
    parser.add_argument('--debug', action='store_true', help='Включить отладочный режим')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"{Colors.CYAN}{'═'*70}\n🎙️  WHISPERX ДИАРИЗАЦИЯ РЕЧИ (DOCKER)\n{'═'*70}{Colors.NC}")
    print(f"Автор скрипта: Михаил Шардин | https://shardin.name/\nДата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")

    try:
        whisperx = WhisperXDocker(config_path=args.config)
        
        if args.check:
            whisperx.check_system(); return
        
        if not whisperx.check_system():
            logger.error("Система не готова. Исправьте ошибки и повторите."); sys.exit(1)

        if args.file:
            file_path = Path(args.file).expanduser().resolve()
            if not file_path.exists():
                logger.error(f"Файл не найден: {file_path}"); sys.exit(1)
            whisperx.process_file(file_path)
        else:
            input_dir = Path(args.directory).expanduser() if args.directory else None
            whisperx.process_directory(input_dir=input_dir)

    except KeyboardInterrupt:
        logger.info(f"\n{Colors.YELLOW}⏹️  Работа прервана пользователем{Colors.NC}"); sys.exit(130)
    except Exception as e:
        logger.error(f"Критическая непредвиденная ошибка: {e}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main()