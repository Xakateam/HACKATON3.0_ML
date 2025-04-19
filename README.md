Ок, вот сокращенная версия документации в формате GitHub Markdown (README.md):

# Детектор Людей HOG + SVM (person_detector_hog_svm)

Классический детектор людей на изображениях с использованием HOG (Гистограммы Ориентированных Градиентов) для признаков и Linear SVM (Линейный Метод Опорных Векторов) для классификации. Включает подготовку данных, обучение, детекцию и простое API на FastAPI.

**Цель:** Демонстрация пайплайна компьютерного зрения и машинного обучения. Может использоваться для базового подсчета людей.

## Основные Возможности

*   Подготовка данных (позитивные/негативные сэмплы) из YOLO-разметки.
*   Извлечение HOG-признаков.
*   Обучение Linear SVM.
*   Детекция людей (скользящее окно, пирамида масштабов, NMS).
*   Оценка качества подсчета (MAE).
*   FastAPI-интерфейс для детекции на загружаемых изображениях.

## Структура Проекта


person_detector_hog_svm/
├── data/ # Данные (YOLO формат, не в репо)
│ └── person-detection-10k/ # Пример структуры
│ ├── train/ (images/, labels/)
│ ├── val/ (images/, labels/)
│ └── test/ (images/, labels/)
├── output/ # Генерируемые файлы (создается авто)
│ ├── prepared_data/ # Нарезанные сэмплы (+/-)
│ ├── features/ # HOG-признаки (.npy)
│ ├── models/ # Модель SVM (.joblib), параметры HOG (.json)
│ └── detections/ # Результаты детекции (картинки)
├── scripts/ # Скрипты для запуска
│ ├── 1_prepare_data.py # Подготовка данных
│ ├── 2_train_model.py # Обучение (HOG + SVM)
│ └── 3_run_detection.py # Детекция на тесте + MAE
├── src/ # Исходный код (модули)
│ ├── config.py # Конфигурация (пути, параметры HOG/SVM/детекции)
│ ├── data_preparation.py # Нарезка сэмплов
│ ├── feature_extraction.py # Расчет HOG
│ ├── training.py # Обучение SVM
│ ├── detection.py # Логика детекции
│ └── utils.py # Вспомогательные функции
├── api_fastapi.py # FastAPI приложение
└── README.md # Этот файл

## Требования

*   Python 3.8+
*   Основные библиотеки: `opencv-python`, `numpy`, `scikit-learn`, `joblib`, `tqdm`, `fastapi`, `uvicorn[standard]`, `python-multipart`
*   (Опционально) `matplotlib` (для графика), `imutils` (если стандартный NMS не работает)

**Установка:**

```bash
# (Рекомендуется) Создать и активировать venv
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows: venv\Scripts\activate

# Установить зависимости (создайте requirements.txt или установите по одному)
pip install opencv-python numpy scikit-learn joblib tqdm fastapi "uvicorn[standard]" python-multipart matplotlib
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Датасет

Формат: YOLO (images/*.jpg, labels/*.txt).

Структура: train/, val/, test/.

Разметка (.txt): <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> (класс 0 для людей в config.py).

Поместите в data/ или укажите путь в src/config.py.

Конфигурация

Основные параметры (пути, HOG, SVM, детекция) настраиваются в src/config.py. Важно: Параметры HOG, детекции (DETECTION_THRESHOLD, NMS_THRESHOLD) требуют подбора под ваши данные.

Использование (Пайплайн)

Запускайте скрипты из корневой папки проекта:

Подготовка данных: Нарезает сэмплы из train.

python scripts/1_prepare_data.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Обучение модели: Извлекает HOG-признаки и обучает SVM.

python scripts/2_train_model.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Детекция и Оценка: Запускает детектор на test, считает MAE, сохраняет результаты.

python scripts/3_run_detection.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Анализируйте MAE и картинки в output/detections/. При необходимости измените параметры в config.py и перезапустите.

Использование API (FastAPI)

Запуск сервера: (убедитесь, что модель обучена: output/models/svm_model.joblib существует)

uvicorn api_fastapi:app --reload --host 0.0.0.0 --port 8000
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Отправка запроса (curl):

curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:8000/predict
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Отправка запроса (Python requests):

import requests
api_url = "http://localhost:8000/predict"
image_path = "/path/to/your/image.jpg"
with open(image_path, 'rb') as f:
    files = {'image': (image_path, f, 'image/jpeg')}
    response = requests.post(api_url, files=files)
print(response.json())
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

Ответ (JSON):

# Успех
{ "success": true, "error": null, "person_count": 5 }
# Ошибка
{ "success": false, "error": "Описание ошибки", "person_count": 0 }
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

Проверка состояния: GET http://localhost:8000/health

Оценка и Ограничения

Метрика: Mean Absolute Error (MAE) на тестовом наборе (чем ниже, тем лучше).

Ограничения:

Устаревший подход по сравнению с Deep Learning (YOLO, SSD и т.д.).

Ниже точность, особенно в сложных условиях (перекрытия, освещение, позы).

Медленная детекция из-за скользящего окна.

Возможны ложные срабатывания.

Возможные Улучшения

Hard Negative Mining.

Подбор гиперпараметров (HOG, SVM, детекция).

Использование других классификаторов (Random Forest, Boosting).

Переход на нейросетевые детекторы (YOLOv5/v8, Faster R-CNN).

Лицензия

(Укажите здесь вашу лицензию, например: MIT License)

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
