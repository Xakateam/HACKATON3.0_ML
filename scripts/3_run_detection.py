# person_detector_hog_svm/scripts/3_run_detection.py

import sys
import os
import cv2 # OpenCV для работы с изображениями
from tqdm import tqdm # Рисует красивый прогресс-бар для циклов
import numpy as np # NumPy для вычислений, особенно для MAE
import time # Для замера времени выполнения

# Снова добавляем корень проекта в пути, чтобы импорты работали
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Наши модули: сам детектор, конфиг и утилиты (для аннотаций)
from src import detection
from src import config
from src import utils # Понадобится для загрузки правильных ответов (аннотаций)

def calculate_mae(true_counts, pred_counts):
    """
    Простая функция для расчета Mean Absolute Error (Средняя Абсолютная Ошибка).
    Показывает, насколько в среднем мы ошибаемся в подсчете людей.
    """
    # Превращаем списки в numpy массивы для удобства вычислений
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)
    # Считаем разницу по модулю для каждой картинки и усредняем
    mae = np.mean(np.abs(true_counts - pred_counts))
    return mae

def run_evaluation(detector, test_image_dir, test_label_dir, detections_save_dir=None):
    """
    Основная функция этого скрипта: прогоняет детектор по всем тестовым картинкам,
    сравнивает результат с правильными ответами и считает метрику MAE.

    Args:
        detector (cv2.HOGDescriptor): Уже готовый, загруженный HOG+SVM детектор.
        test_image_dir (str): Папка с картинками для теста.
        test_label_dir (str): Папка с файлами аннотаций (правильными ответами) для этих картинок.
        detections_save_dir (str, optional): Если указать папку, сюда будут сохраняться
                                              картинки с нарисованными рамками детекции.
                                              Полезно для визуального анализа ошибок. По умолчанию None (не сохранять).
    """
    print("\n--- Запускаем оценку модели на тестовом наборе ---")

    # Проверяем, существуют ли папки с тестовыми данными
    if not os.path.isdir(test_image_dir):
        print(f"Ошибка: Не найдена папка с тестовыми изображениями: {test_image_dir}")
        return # Дальше работать бессмысленно
    if not os.path.isdir(test_label_dir):
        print(f"Ошибка: Не найдена папка с тестовыми аннотациями: {test_label_dir}")
        return

    # Собираем список всех картинок в тестовой папке (jpg, png и т.д.)
    # Сортируем для порядка
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"В папке {test_image_dir} не найдено изображений.")
        return

    # Списки для хранения результатов по каждой картинке
    true_counts = [] # Сколько людей на самом деле (из аннотаций)
    predicted_counts = [] # Сколько людей нашла наша модель
    processing_times = [] # Сколько времени ушло на обработку каждой картинки

    # Используем tqdm для красивого прогресс-бара в консоли
    print(f"Обрабатываем {len(image_files)} тестовых изображений...")
    for filename in tqdm(image_files, desc="Оценка тестового набора"):
        image_path = os.path.join(test_image_dir, filename)
        # Имя файла аннотации должно совпадать с именем картинки, но с расширением .txt
        label_path = os.path.join(test_label_dir, os.path.splitext(filename)[0] + '.txt')

        # Грузим картинку
        image = cv2.imread(image_path)
        if image is None:
            # Если картинка битая или не читается, пропускаем ее
            print(f"Предупреждение: Не удалось прочитать изображение {image_path}. Пропускаем.")
            continue

        # Загружаем "правильные ответы" - аннотации в формате YOLO
        annotations = utils.load_annotations(label_path)
        # Считаем, сколько там объектов с ID класса "человек" (из нашего конфига)
        true_count = sum(1 for ann in annotations if ann['class_id'] == config.PERSON_CLASS_ID)
        true_counts.append(true_count) # Запоминаем правильное число

        # ----- Самый главный момент: ЗАПУСК ДЕТЕКЦИИ -----
        start_time = time.time() # Время пошло
        # Вызываем функцию детекции из модуля detection.py
        # Передаем ей картинку, наш загруженный детектор и параметры из конфига
        final_boxes, pred_count, output_img = detection.detect_people(
            image,
            detector,
            # Эти параметры влияют на скорость и качество детекции
            win_stride=tuple(config.HOG_PARAMS['blockStride']), # Шаг окна, берем из HOG параметров
            padding=(16, 16),                    # Отступы по краям окна
            scale=config.DETECTION_PYRAMID_SCALE,# Масштаб пирамиды изображений
            hit_threshold=config.DETECTION_THRESHOLD, # Порог уверенности SVM
            use_nms=True,                        # Включаем Non-Maximum Suppression (убирает дубликаты)
            nms_threshold=config.NMS_THRESHOLD   # Порог для NMS
        )
        end_time = time.time() # Время стоп
        # ----- Детекция завершена -----

        processing_times.append(end_time - start_time) # Запоминаем время обработки
        predicted_counts.append(pred_count) # Запоминаем, сколько насчитала модель

        # Если указана папка для сохранения результатов - сохраняем картинку с рамками
        if detections_save_dir:
            # Создаем папку, если её еще нет
            os.makedirs(detections_save_dir, exist_ok=True)
            # Формируем путь для сохранения
            save_path = os.path.join(detections_save_dir, f"detected_{filename}")
            try:
                # Пытаемся сохранить картинку
                cv2.imwrite(save_path, output_img)
            except Exception as e:
                # Если не получилось сохранить (например, нет прав или места) - пишем предупреждение
                print(f"Предупреждение: Не удалось сохранить картинку с детекцией {save_path}. Ошибка: {e}")


    # --- Подведение итогов ---
    # Если по какой-то причине списки результатов пустые, считать нечего
    if not true_counts or not predicted_counts:
        print("Нет результатов для оценки.")
        return

    # Считаем финальную метрику MAE
    mae = calculate_mae(true_counts, predicted_counts)
    # Считаем среднее время обработки одного изображения
    avg_time = np.mean(processing_times) if processing_times else 0

    # Красиво выводим результаты в консоль
    print("-" * 30)
    print("Результаты оценки:")
    print(f"Обработано изображений: {len(image_files)}")
    print(f"Средняя Абсолютная Ошибка (MAE): {mae:.4f}") # Чем ниже, тем лучше
    print(f"Среднее время обработки изображения: {avg_time:.4f} сек")

    # --- Закомментированный код для построения графика ---
    # Иногда полезно посмотреть на график рассеяния: что предсказали vs что было на самом деле.
    # Если раскомментировать, понадобится matplotlib (`pip install matplotlib`)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 8))
    # plt.scatter(true_counts, predicted_counts, alpha=0.5) # Точки предсказаний
    # plt.xlabel("Истинное количество людей")
    # plt.ylabel("Предсказанное количество людей")
    # plt.title(f"Предсказания vs Истина (MAE: {mae:.2f})")
    # # Рисуем красную пунктирную линию идеального предсказания (y=x)
    # min_val = min(min(true_counts), min(predicted_counts))
    # max_val = max(max(true_counts), max(predicted_counts))
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    # plt.grid(True)
    # plt.show() # Показать график
    # ------------------------------------------------------
    print("-" * 30)


# Основной блок выполнения скрипта
if __name__ == "__main__":
    print("--- Запускаем скрипт детекции и оценки ---")
    overall_start_time = time.time() # Общее время старта

    # Шаг 1: Загружаем наш обученный детектор
    # Функция load_detector берет сохраненную модель SVM и параметры HOG
    print("Загружаем детектор...")
    hog_detector = detection.load_detector(config.SVM_MODEL_PATH, config.HOG_PARAMS_JSON)

    # Если детектор успешно загрузился...
    if hog_detector:
        print("Детектор успешно загружен.")
        # Шаг 2: Запускаем оценку на тестовых данных
        run_evaluation(
            detector=hog_detector,               # Передаем загруженный детектор
            test_image_dir=config.TEST_IMAGE_DIR, # Папка с тестовыми картинками
            test_label_dir=config.TEST_LABEL_DIR, # Папка с ответами к ним
            detections_save_dir=config.DETECTIONS_DIR # Папка для сохранения результатов детекции
        )
    else:
        # Если детектор не загрузился (модели нет, файл битый) - сообщаем и выходим
        print("Ошибка: Не удалось загрузить детектор. Оценка невозможна.")

    overall_end_time = time.time() # Общее время окончания
    print(f"--- Скрипт детекции и оценки завершен за {overall_end_time - overall_start_time:.2f} сек ---")