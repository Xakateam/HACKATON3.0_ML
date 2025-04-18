# person_detector_hog_svm/src/data_preparation.py

import os      # Для работы с файлами и путями
import cv2     # OpenCV для работы с изображениями (чтение, ресайз, сохранение)
import numpy as np # NumPy - пока не используется напрямую, но часто нужен с OpenCV
from tqdm import tqdm # Прогресс-бар для циклов, чтобы было видно, что процесс идет
import random  # Для случайного выбора мест для негативных сэмплов
import sys     # Для манипуляций с путями импорта (sys.path)

# Опять трюк с путем, чтобы импортировать наши модули config и utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Добавляем корень проекта в список мест, где Python ищет модули
if project_root not in sys.path: # Проверяем, чтобы не добавить путь дважды
    sys.path.append(project_root)

# Импортируем наш конфиг и утилиты
from src import config # Здесь все настройки: пути, размеры, пороги
from src import utils  # Здесь вспомогательные функции: загрузка аннотаций, расчет IoU и т.д.

def extract_samples(image_dir, label_dir, positive_output_dir, negative_output_dir,
                    window_size, neg_samples_per_image, neg_iou_threshold, person_class_id):
    """
    Главная рабочая лошадка этого модуля. Проходит по всем картинкам в `image_dir`,
    читает аннотации из `label_dir` и нарезает два типа сэмплов:
    1. Позитивные: Вырезанные изображения людей (класс `person_class_id`), приведенные к `window_size`.
    2. Негативные: Случайные куски фона того же `window_size`, которые НЕ пересекаются
                   с реальными людьми (проверка по `neg_iou_threshold`).
    Все нарезанные картинки сохраняются в `positive_output_dir` и `negative_output_dir`.

    Args:
        image_dir (str): Папка с исходными изображениями (например, обучающий набор).
        label_dir (str): Папка с YOLO-аннотациями (.txt файлы) к этим изображениям.
        positive_output_dir (str): Куда сохранять позитивные примеры (людей).
        negative_output_dir (str): Куда сохранять негативные примеры (фон).
        window_size (tuple): Целевой размер (ширина, высота) для ВСЕХ сэмплов. Важно для HOG!
        neg_samples_per_image (int): Сколько негативных примеров пытаться вырезать с каждой картинки.
        neg_iou_threshold (float): Максимальное допустимое пересечение (IoU) негативного сэмпла
                                   с любым реальным объектом-человеком.
        person_class_id (int): ID класса "человек" в файлах аннотаций.
    """
    print("Начинаем нарезку позитивных и негативных сэмплов...")
    positive_count = 0 # Счетчик сохраненных позитивов
    negative_count = 0 # Счетчик сохраненных негативов
    win_w, win_h = window_size # Распакуем размер окна для удобства

    # Получаем список всех файлов-картинок в исходной папке
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Найдено {len(image_files)} изображений для обработки.")

    # Идем по каждой картинке с прогресс-баром
    for filename in tqdm(image_files, desc="Обработка изображений"):
        # Собираем полные пути к картинке и файлу аннотации
        image_path = os.path.join(image_dir, filename)
        # Имя аннотации должно совпадать с именем картинки, но с расширением .txt
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

        # Читаем картинку
        image = cv2.imread(image_path)
        # Если картинка не прочиталась (битая, неподдерживаемый формат) - пропускаем
        if image is None:
            print(f"Предупреждение: Не удалось прочитать {image_path}. Пропускаем.")
            continue
        # Запоминаем реальные размеры картинки
        img_h, img_w = image.shape[:2]

        # Загружаем аннотации (список словарей) для этой картинки из .txt файла
        annotations = utils.load_annotations(label_path)

        # --- Извлечение позитивных сэмплов ---
        # Собираем сюда реальные рамки людей (в пикселях), чтобы потом проверять пересечение с негативами
        ground_truth_boxes = []
        for ann in annotations:
            # Нас интересуют только объекты класса "человек"
            if ann['class_id'] == person_class_id:
                # Конвертируем нормализованные координаты YOLO (0-1) в пиксельные (xmin, ymin, xmax, ymax)
                xmin, ymin, xmax, ymax = utils.norm_to_pixel(
                    ann['xc'], ann['yc'], ann['w'], ann['h'], img_w, img_h
                )

                # Проверяем, что рамка имеет ненулевую площадь (защита от ошибок в аннотациях)
                if xmax > xmin and ymax > ymin:
                    # Вырезаем кусок картинки по координатам рамки
                    positive_sample = image[ymin:ymax, xmin:xmax]
                    # Приводим вырезанный кусок к нашему стандартному размеру окна
                    # INTER_AREA - хороший выбор для уменьшения изображений
                    resized_positive = cv2.resize(positive_sample, window_size, interpolation=cv2.INTER_AREA)
                    # Формируем имя файла для сохранения
                    save_path = os.path.join(positive_output_dir, f"{os.path.splitext(filename)[0]}_pos_{positive_count}.png")
                    # Сохраняем позитивный сэмпл
                    cv2.imwrite(save_path, resized_positive)
                    positive_count += 1 # Увеличиваем счетчик
                    # Добавляем пиксельные координаты этой рамки в список для дальнейшей проверки IoU
                    ground_truth_boxes.append([xmin, ymin, xmax, ymax])

        # --- Извлечение негативных сэмплов ---
        # Пытаемся нарезать `neg_samples_per_image` штук с текущей картинки
        neg_extracted_count = 0 # Сколько уже нарезали с ЭТОЙ картинки
        # Чтобы не зациклиться, если подходящий фон найти сложно, ограничиваем число попыток
        max_attempts = neg_samples_per_image * 25 # Например, в 25 раз больше, чем нужно найти

        for _ in range(max_attempts):
            # Если уже нарезали достаточно негативов с этой картинки - выходим из цикла попыток
            if neg_extracted_count >= neg_samples_per_image:
                break

            # Если картинка слишком маленькая, меньше нашего окна, то негативы не нарежем
            if img_w <= win_w or img_h <= win_h:
                break # Переходим к следующей картинке

            # Генерируем случайные координаты левого верхнего угла (nx, ny)
            # для окна размером win_w x win_h внутри картинки
            nx = random.randint(0, img_w - win_w - 1) # От 0 до (ширина_картинки - ширина_окна - 1)
            ny = random.randint(0, img_h - win_h - 1) # От 0 до (высота_картинки - высота_окна - 1)
            # Координаты правого нижнего угла
            nx_max = nx + win_w
            ny_max = ny + win_h
            # Сохраняем рамку негативного кандидата
            neg_box = [nx, ny, nx_max, ny_max]

            # Теперь самая важная проверка: не пересекается ли наш случайный кусок фона
            # слишком сильно с реальными людьми на картинке?
            max_iou = 0.0 # Будем искать максимальное пересечение
            for gt_box in ground_truth_boxes: # Проходим по всем рамкам людей
                # Считаем IoU (Intersection over Union) между случайной рамкой и рамкой человека
                iou = utils.calculate_iou(neg_box, gt_box)
                # Обновляем максимум, если текущее пересечение больше
                max_iou = max(max_iou, iou)

            # Если максимальное пересечение МЕНЬШЕ нашего порога (т.е. почти не пересекается)
            if max_iou < neg_iou_threshold:
                # Ура, это хороший негативный сэмпл! Вырезаем его.
                negative_sample = image[ny:ny_max, nx:nx_max]
                # На всякий случай проверим, что размер вырезанного куска точно совпадает с нужным
                # (хотя при правильных расчетах выше это должно быть так)
                if negative_sample.shape[1] == win_w and negative_sample.shape[0] == win_h:
                    # Формируем имя файла
                    save_path = os.path.join(negative_output_dir, f"{os.path.splitext(filename)[0]}_neg_{negative_count}.png")
                    # Сохраняем
                    cv2.imwrite(save_path, negative_sample)
                    negative_count += 1 # Увеличиваем общий счетчик негативов
                    neg_extracted_count += 1 # Увеличиваем счетчик негативов для текущей картинки
                # else: # Можно добавить отладочный print, если размеры вдруг не совпали
                    # print(f"Warning: Extracted negative sample size mismatch for {filename}. Expected {window_size}, got {negative_sample.shape[:2][::-1]}")


    # --- Финальный отчет ---
    print("-" * 30)
    print(f"Нарезка сэмплов завершена.")
    print(f"Всего позитивных сэмплов: {positive_count}")
    print(f"Всего негативных сэмплов: {negative_count}")
    print(f"Позитивные сэмплы сохранены в: {positive_output_dir}")
    print(f"Негативные сэмплы сохранены в: {negative_output_dir}")
    print("-" * 30)

# Этот блок выполняется, только если запустить скрипт напрямую
# (python src/data_preparation.py)
# Используется для тестирования или как основной способ запуска этого этапа.
if __name__ == "__main__":
    print("Запускаем data_preparation.py как основной скрипт...")
    # Вызываем нашу главную функцию со всеми параметрами из конфига
    extract_samples(
        image_dir=config.TRAIN_IMAGE_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        positive_output_dir=config.POSITIVE_SAMPLES_DIR,
        negative_output_dir=config.NEGATIVE_SAMPLES_DIR,
        window_size=config.WINDOW_SIZE,
        neg_samples_per_image=config.NEG_SAMPLES_PER_IMAGE,
        neg_iou_threshold=config.NEG_IOU_THRESHOLD,
        person_class_id=config.PERSON_CLASS_ID
    )
    print("Работа data_preparation.py завершена.")