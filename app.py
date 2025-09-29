import os
import sys
import glob
import json
import time
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import whisper
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Добавляем фильтр nl2br
@app.template_filter('nl2br')
def nl2br_filter(s):
    if not s:
        return ""
    return s.replace('\n', '<br>')

class UploadForm(FlaskForm):
    file = FileField('Аудио/видео файл', validators=[
        FileRequired(),
        FileAllowed([
            'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'wma',
            'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'mpeg', 'mpg', 'm4v', 'webm', '3gp'
        ], 'Разрешены только аудио и видео файлы!')
    ])
    model_size = SelectField('Модель', choices=[
        ('tiny', 'Tiny (самая быстрая, низкое качество)'),
        ('base', 'Base (быстрая, среднее качество)'),
        ('small', 'Small (умеренная скорость, хорошее качество)'),
        ('medium', 'Medium (медленная, высокое качество)'),
        ('large', 'Large (очень медленная, лучшее качество)')
    ], default='medium')
    language = SelectField('Язык', choices=[
        ('ru', 'Русский'),
        ('en', 'Английский'),
        ('auto', 'Автоопределение')
    ], default='ru')
    submit = SubmitField('Загрузить и распознать')

class BatchForm(FlaskForm):
    model_size = SelectField('Модель', choices=[
        ('tiny', 'Tiny (самая быстрая, низкое качество)'),
        ('base', 'Base (быстрая, среднее качество)'),
        ('small', 'Small (умеренная скорость, хорошее качество)'),
        ('medium', 'Medium (медленная, высокое качество)'),
        ('large', 'Large (очень медленная, лучшее качество)')
    ], default='medium')
    language = SelectField('Язык', choices=[
        ('ru', 'Русский'),
        ('en', 'Английский'),
        ('auto', 'Автоопределение')
    ], default='ru')
    submit = SubmitField('Запустить пакетную обработку')

# Глобальные переменные для хранения состояния
current_task = None
task_progress = {'current': 0, 'total': 0, 'status': 'idle', 'current_file': ''}
task_results = {}

def check_gpu():
    """Проверка доступности CUDA и GPU с тестированием совместимости"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Тест совместимости GPU
        test_tensor = torch.zeros(10, 10).cuda()
        _ = test_tensor + 1
        test_tensor = test_tensor.cpu()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False

def load_whisper_model(model_size="medium", use_gpu=True):
    """Загрузка модели Whisper с обработкой ошибок GPU"""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    try:
        model = whisper.load_model(model_size, device=device)
        return model, device
    except Exception as e:
        if device == "cuda":
            model = whisper.load_model(model_size, device="cpu")
            return model, "cpu"
        else:
            raise e

def format_timestamp(seconds):
    """Форматирование времени для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def process_single_file(file_path, model_size, language, task_id):
    """Обработка одного файла в отдельном потоке"""
    global task_progress, task_results
    
    try:
        task_progress['status'] = 'processing'
        task_progress['current_file'] = os.path.basename(file_path)
        
        # Загрузка модели
        use_gpu = check_gpu()
        model, device = load_whisper_model(model_size, use_gpu)
        
        # Определение языка
        whisper_language = None if language == 'auto' else language
        
        # Распознавание
        result = model.transcribe(
            file_path,
            language=whisper_language,
            verbose=False,
            fp16=(device == "cuda")
        )
        
        # Сохранение результатов
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Текстовый файл
        txt_path = os.path.join(result_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result["text"].strip())
        
        # SRT файл
        srt_path = None
        if result.get("segments"):
            srt_path = os.path.join(result_dir, f"{base_name}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = format_timestamp(segment['start'])
                    end = format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        
        # Сохраняем информацию о результате
        task_results[task_id] = {
            'file': file_path,
            'text': result["text"].strip(),
            'segments': result.get("segments", []),
            'language': result.get("language", language),
            'txt_path': txt_path,
            'srt_path': srt_path,
            'model_size': model_size,
            'device': device
        }
        
        task_progress['status'] = 'completed'
        task_progress['current'] = 1
        task_progress['total'] = 1
        
    except Exception as e:
        task_progress['status'] = 'error'
        task_results[task_id] = {'error': str(e)}

def process_batch_files(directory, model_size, language, task_id):
    """Пакетная обработка файлов в директории"""
    global task_progress, task_results
    
    try:
        # Поиск файлов
        extensions = [
            '*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac', '*.ogg', '*.wma',
            '*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.mpeg', '*.mpg', '*.m4v', '*.webm', '*.3gp'
        ]
        
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        
        if not files:
            task_progress['status'] = 'error'
            task_results[task_id] = {'error': 'Файлы не найдены'}
            return
        
        files = sorted(files)
        
        # Загрузка модели
        use_gpu = check_gpu()
        model, device = load_whisper_model(model_size, use_gpu)
        
        # Определение языка
        whisper_language = None if language == 'auto' else language
        
        task_progress['status'] = 'processing'
        task_progress['total'] = len(files)
        task_progress['current'] = 0
        
        results = []
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Обработка каждого файла
        for i, file_path in enumerate(files):
            task_progress['current'] = i + 1
            task_progress['current_file'] = os.path.basename(file_path)
            
            result = model.transcribe(
                file_path,
                language=whisper_language,
                verbose=False,
                fp16=(device == "cuda")
            )
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Сохранение индивидуальных файлов
            txt_path = os.path.join(result_dir, f"{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result["text"].strip())
            
            srt_path = None
            if result.get("segments"):
                srt_path = os.path.join(result_dir, f"{base_name}.srt")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for j, segment in enumerate(result["segments"], 1):
                        start = format_timestamp(segment['start'])
                        end = format_timestamp(segment['end'])
                        text = segment['text'].strip()
                        f.write(f"{j}\n{start} --> {end}\n{text}\n\n")
            
            # Сохранение в общий файл
            all_txt_path = os.path.join(result_dir, "all_transcripts.txt")
            with open(all_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"=== {os.path.basename(file_path)} ===\n")
                f.write(f"{result['text'].strip()}\n\n")
            
            results.append({
                'file': file_path,
                'text': result["text"].strip(),
                'segments': result.get("segments", []),
                'language': result.get("language", language),
                'txt_path': txt_path,
                'srt_path': srt_path
            })
        
        # Сохранение JSON с результатами
        json_path = os.path.join(result_dir, "transcripts_detailed.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        task_results[task_id] = {
            'results': results,
            'total_files': len(files),
            'model_size': model_size,
            'device': device,
            'json_path': json_path,
            'all_txt_path': os.path.join(result_dir, "all_transcripts.txt")
        }
        
        task_progress['status'] = 'completed'
        
    except Exception as e:
        task_progress['status'] = 'error'
        task_results[task_id] = {'error': str(e)}

@app.route('/', methods=['GET', 'POST'])
def index():
    single_form = UploadForm()
    batch_form = BatchForm()
    
    if single_form.validate_on_submit() and single_form.file.data:
        # Обработка одиночного файла
        filename = secure_filename(single_form.file.data.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        single_form.file.data.save(file_path)
        
        task_id = f"single_{int(time.time())}"
        thread = threading.Thread(
            target=process_single_file,
            args=(file_path, single_form.model_size.data, single_form.language.data, task_id)
        )
        thread.start()
        
        flash(f'Файл {filename} загружен и поставлен в очередь на обработку', 'success')
        return redirect(url_for('progress', task_id=task_id))
    
    # Получение списков уже обработанных файлов
    processed_files = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        for task_dir in os.listdir(app.config['RESULTS_FOLDER']):
            task_path = os.path.join(app.config['RESULTS_FOLDER'], task_dir)
            if os.path.isdir(task_path):
                processed_files.append({
                    'id': task_dir,
                    'files': [f for f in os.listdir(task_path) if f.endswith('.txt') and f != 'all_transcripts.txt']
                })
    
    return render_template('index.html', 
                         single_form=single_form, 
                         batch_form=batch_form,
                         processed_files=processed_files)

@app.route('/batch', methods=['POST'])
def batch_process():
    batch_form = BatchForm()
    
    if batch_form.validate_on_submit():
        task_id = f"batch_{int(time.time())}"
        thread = threading.Thread(
            target=process_batch_files,
            args=(app.config['UPLOAD_FOLDER'], batch_form.model_size.data, batch_form.language.data, task_id)
        )
        thread.start()
        
        flash('Пакетная обработка запущена', 'success')
        return redirect(url_for('progress', task_id=task_id))
    
    return redirect(url_for('index'))

@app.route('/progress/<task_id>')
def progress(task_id):
    return render_template('progress.html', task_id=task_id)

@app.route('/api/progress/<task_id>')
def api_progress(task_id):
    global task_progress, task_results
    
    if task_id in task_results:
        if task_progress['status'] == 'completed':
            return jsonify({
                'status': 'completed',
                'result': task_results[task_id]
            })
        elif task_progress['status'] == 'error':
            return jsonify({
                'status': 'error',
                'error': task_results[task_id].get('error', 'Unknown error')
            })
    
    return jsonify({
        'status': task_progress['status'],
        'current': task_progress['current'],
        'total': task_progress['total'],
        'current_file': task_progress['current_file']
    })

@app.route('/results/<task_id>')
def results(task_id):
    if task_id not in task_results:
        flash('Результаты не найдены', 'error')
        return redirect(url_for('index'))
    
    result = task_results[task_id]
    
    if 'error' in result:
        flash(f'Ошибка обработки: {result["error"]}', 'error')
        return redirect(url_for('index'))
    
    return render_template('results.html', result=result, task_id=task_id)

@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    file_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('Файл не найден', 'error')
        return redirect(url_for('index'))

@app.route('/system_info')
def system_info():
    gpu_available = check_gpu()
    gpu_info = ""
    
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"{gpu_name} ({memory_gb:.1f} GB)"
        except:
            gpu_info = "Доступен но не совместим"
    
    return jsonify({
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'pytorch_version': torch.__version__,
        'whisper_available': True
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
