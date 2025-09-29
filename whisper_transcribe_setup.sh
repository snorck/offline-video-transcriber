#!/bin/bash

# 🛠️ Скрипт для настройки окружения OpenAI Whisper на Ubuntu 🛠️
#
# Этот Shell-скрипт автоматизирует полную установку и настройку программного
# окружения, необходимого для работы системы распознавания речи OpenAI Whisper
# с использованием GPU от NVIDIA.
#
# Напоминание: Скрипт ориентирован на Ubuntu и может потребовать адаптации
# для других дистрибутивов Linux.
#
# Основные задачи:
# - Обновление системы и установка базовых утилит (python3-venv, ffmpeg).
# - Проверка и установка драйверов NVIDIA.
# - Проверка и установка CUDA Toolkit для вычислений на GPU.
# - Создание изолированного Python-окружения (.venv) для избежания конфликтов.
# - Установка PyTorch с учетом архитектуры GPU (стабильная или nightly версия).
# - Установка библиотеки openai-whisper и других зависимостей.
# - Запуск финального теста для проверки совместимости PyTorch и GPU.
#
# Порядок использования:
# 1. Сделайте скрипт исполняемым: chmod +x setup_whisper.sh
# 2. Запустите его: ./setup_whisper.sh
# 3. В случае установки драйверов NVIDIA может потребоваться перезагрузка.

#  Следить за состоянием GPU: $ watch -n 5 nvidia-smi
#
# Автор: Михаил Шардин https://shardin.name/
# Дата создания: 29.08.2025
# Версия: 1.1
#
# Актуальная версия скрипта всегда здесь: https://github.com/empenoso/offline-audio-transcriber
#

echo "🚀 Установка окружения для OpenAI Whisper"
echo "========================================="

# Проверка Ubuntu версии
echo "📋 Информация о системе:"
lsb_release -a
echo ""

# Обновление системы
echo "🔄 Обновление пакетов..."
sudo apt update && sudo apt upgrade -y

# Установка Python и pip
echo "🐍 Установка Python и зависимостей..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Установка системных зависимостей для аудио
echo "🎵 Установка библиотек для работы с аудио..."
sudo apt install -y ffmpeg libsndfile1 portaudio19-dev

# Проверка NVIDIA драйверов
echo "🎮 Проверка NVIDIA драйверов..."
if nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA драйверы установлены"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA драйверы не найдены. Установка..."
    sudo apt install -y nvidia-driver-575 nvidia-dkms-575
    echo "🔄 После установки драйверов требуется перезагрузка!"
    echo "Запустите: sudo reboot"
fi

# Установка CUDA toolkit (если нужно)
echo "🔧 Проверка CUDA..."
if nvcc --version &> /dev/null; then
    echo "✅ CUDA toolkit уже установлен"
    nvcc --version
else
    echo "📦 Установка CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt-get -y install cuda-toolkit-13-0
fi

# Создание виртуального окружения
echo "🏠 Создание виртуального окружения..."
python3 -m venv .venv
source .venv/bin/activate

# Обновление pip
echo "⬆️  Обновление pip..."
pip install --upgrade pip

# Определение архитектуры GPU для выбора совместимой версии PyTorch
echo "🔥 Установка PyTorch с поддержкой RTX 5060 Ti..."

# Проверяем архитектуру GPU
if nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
    echo "🎮 Обнаружен GPU: $GPU_INFO"
    
    # Для RTX 5060 Ti (Ada Lovelace) нужна nightly версия PyTorch
    if echo "$GPU_INFO" | grep -q "RTX 5060 Ti\|RTX 40\|RTX 50"; then
        echo "🚀 Установка PyTorch nightly для поддержки новых GPU..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    else
        echo "📦 Установка стабильной версии PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "📦 GPU не обнаружен, установка CPU версии PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Установка OpenAI Whisper
echo "🎙️  Установка OpenAI Whisper..."
pip install openai-whisper

# Дополнительные полезные библиотеки
echo "📚 Установка дополнительных библиотек..."
pip install numpy scipy librosa soundfile pydub

# Дополнительные библиотеки для работы web-интерфейса
echo "📚 Установка дополнительных библиотек..."
pip install flask flask-wtf werkzeug

# Тест установки с проверкой совместимости GPU
echo "🧪 Тестирование установки..."
python3 -c "
import torch
import whisper
print(f'PyTorch версия: {torch.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f'GPU: {gpu_name}')
        print(f'CUDA версия: {torch.version.cuda}')
        print(f'GPU устройств: {torch.cuda.device_count()}')
        
        # Тест совместимости
        test_tensor = torch.zeros(10, 10).cuda()
        result = test_tensor + 1
        print('✅ GPU совместим с PyTorch')
        
    except Exception as e:
        print(f'⚠️  GPU несовместим: {e}')
        print('🔄 Будет использоваться CPU режим')
else:
    print('💻 Будет использоваться CPU')

print('✅ Whisper импортирован успешно')
"

echo ""
echo "🎉 Установка завершена!"
echo "========================================="
echo "Для активации окружения используйте:"
echo "source .venv/bin/activate"
echo ""
echo "Для запуска скрипта:"
echo "python3 whisper_transcribe.py [директория] [модель] [выходная_папка]"
echo ""
echo "Примеры:"
echo "python3 whisper_transcribe.py ./audio"
echo "python3 whisper_transcribe.py ./audio large ./results"
echo ""
echo "Доступные модели (от быстрой к точной):"
echo "tiny, base, small, medium, large"
