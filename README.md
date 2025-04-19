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


![изображение](https://github.com/user-attachments/assets/c533b409-2b18-47f9-bf56-f2fa69e57c23)

## Требования

*   Python 3.8+
*   Основные библиотеки: `opencv-python`, `numpy`, `scikit-learn`, `joblib`, `tqdm`, `fastapi`, `uvicorn[standard]`, `python-multipart`
*   (Опционально) `matplotlib` (для графика), `imutils` (если стандартный NMS не работает)

**Установка:**


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
pip install opencv-python numpy scikit-learn joblib tqdm fastapi "uvicorn[standard]" python-multipart matplotlib```
```

# Датасет
    Формат: YOLO (images/*.jpg, labels/*.txt).
    Структура: train/, val/, test/.
    Разметка (.txt): <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> (класс 0 для людей в config.py).
    Поместите в data/ или укажите путь в src/config.py.

# Конфигурация
Основные параметры (пути, HOG, SVM, детекция) настраиваются в src/config.py. Важно: Параметры HOG, детекции (DETECTION_THRESHOLD, NMS_THRESHOLD) требуют подбора под ваши данные.

#Использование (Пайплайн)
Запускайте скрипты из корневой папки проекта:
    Подготовка данных: Нарезает сэмплы из train.

```python
python scripts/1_prepare_data.py
```

Обучение модели: Извлекает HOG-признаки и обучает SVM.

```python
python scripts/2_train_model.py
```

Детекция и Оценка: Запускает детектор на test, считает MAE, сохраняет результаты.

```python
python scripts/3_run_detection.py
```

Анализируйте MAE и картинки в output/detections/. При необходимости измените параметры в config.py и перезапустите.

# Использование API (FastAPI)

Запуск сервера: (убедитесь, что модель обучена: output/models/svm_model.joblib существует)

```python
uvicorn api_fastapi:app --reload --host 0.0.0.0 --port 8000
```

Отправка запроса (curl):

```python
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:8000/predict
```

Отправка запроса (Python requests):
```python
import requests
api_url = "http://localhost:8000/predict"
image_path = "/path/to/your/image.jpg"
with open(image_path, 'rb') as f:
    files = {'image': (image_path, f, 'image/jpeg')}
    response = requests.post(api_url, files=files)
print(response.json())
```

Ответ (JSON):

```python
# Успех
{ "success": true, "error": null, "person_count": 5 }
# Ошибка
{ "success": false, "error": "Описание ошибки", "person_count": 0 }
```

Проверка состояния: GET http://localhost:8000/health
