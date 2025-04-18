# person_detector_hog_svm/src/utils.py

import numpy as np # NumPy для математических операций и работы с массивами (особенно в NMS)
import cv2         # OpenCV для ресайза изображений
import os          # Для проверки существования файлов аннотаций

# --- Функции для работы с аннотациями YOLO ---

def load_annotations(label_path):
    """
    Читает файл аннотаций в формате YOLO (.txt) и возвращает список словарей.
    Каждая строка YOLO: class_id xc yc w h (нормализованные координаты 0-1).

    Args:
        label_path (str): Путь к файлу .txt с аннотациями.

    Returns:
        list: Список словарей, где каждый словарь представляет один объект
              {'class_id': int, 'xc': float, 'yc': float, 'w': float, 'h': float}.
              Если файл не найден или пуст, возвращает пустой список.
    """
    annotations = [] # Сюда будем складывать разобранные аннотации

    # Проверяем, существует ли файл, прежде чем пытаться его открыть
    if not os.path.exists(label_path):
        # Если файла нет (например, для негативного изображения без объектов),
        # просто возвращаем пустой список - это нормальная ситуация.
        return annotations

    try:
        # Открываем файл для чтения ('r')
        with open(label_path, 'r') as f:
            # Читаем файл построчно
            for line in f:
                # Убираем лишние пробелы по краям строки и разбиваем по пробелам
                parts = line.strip().split()
                # Ожидаем 5 частей: class_id, xc, yc, w, h
                if len(parts) == 5:
                    try:
                        # Конвертируем значения в нужные типы
                        class_id = int(parts[0]) # ID класса - целое число
                        # Координаты и размеры - числа с плавающей точкой
                        xc, yc, w, h = map(float, parts[1:])
                        # Добавляем словарь с данными об объекте в наш список
                        annotations.append({'class_id': class_id, 'xc': xc, 'yc': yc, 'w': w, 'h': h})
                    except ValueError:
                        # Если не удалось сконвертировать что-то в число - пропускаем строку
                        print(f"Предупреждение: Неверный формат строки в файле {label_path}: '{line.strip()}'. Пропускаем.")
                        continue
    except Exception as e:
        # Если произошла ошибка при чтении файла
        print(f"Ошибка при чтении файла аннотаций {label_path}: {e}")
        # Вернем то, что успели собрать (может быть пустым списком)
        return annotations

    # Возвращаем список найденных и разобранных аннотаций
    return annotations

def norm_to_pixel(xc, yc, w, h, img_width, img_height):
    """
    Конвертирует нормализованные координаты YOLO (центр x, центр y, ширина, высота от 0 до 1)
    в абсолютные пиксельные координаты рамки (xmin, ymin, xmax, ymax).

    Args:
        xc (float): Нормализованная координата X центра рамки.
        yc (float): Нормализованная координата Y центра рамки.
        w (float): Нормализованная ширина рамки.
        h (float): Нормализованная высота рамки.
        img_width (int): Ширина изображения в пикселях.
        img_height (int): Высота изображения в пикселях.

    Returns:
        tuple: Кортеж с пиксельными координатами (xmin, ymin, xmax, ymax).
               Координаты ограничены границами изображения.
    """
    # Переводим нормализованные координаты в пиксели
    _xc_pix = xc * img_width   # Пиксельная координата X центра
    _yc_pix = yc * img_height  # Пиксельная координата Y центра
    _w_pix = w * img_width     # Ширина в пикселях
    _h_pix = h * img_height    # Высота в пикселях

    # Вычисляем координаты углов (xmin, ymin) и (xmax, ymax)
    xmin = int(_xc_pix - _w_pix / 2)
    ymin = int(_yc_pix - _h_pix / 2)
    xmax = int(_xc_pix + _w_pix / 2)
    ymax = int(_yc_pix + _h_pix / 2)

    # --- Важная деталь: Ограничение рамки границами изображения ---
    # Иногда из-за округлений или особенностей аннотаций рамка может вылезти за пределы картинки.
    # Обрезаем ее, чтобы избежать ошибок при вырезании сэмплов или других операциях.
    xmin = max(0, xmin)               # xmin не может быть меньше 0
    ymin = max(0, ymin)               # ymin не может быть меньше 0
    xmax = min(img_width - 1, xmax)   # xmax не может быть больше ширины - 1
    ymax = min(img_height - 1, ymax)  # ymax не может быть больше высоты - 1

    return xmin, ymin, xmax, ymax

# --- Функции для работы с рамками (Bounding Boxes) ---

def calculate_iou(boxA, boxB):
    """
    Вычисляет метрику Intersection over Union (IoU) для двух прямоугольных рамок.
    IoU показывает, насколько сильно рамки пересекаются (от 0 до 1).
    1 - полное совпадение, 0 - нет пересечения.

    Args:
        boxA (list or tuple): Координаты первой рамки [xmin, ymin, xmax, ymax].
        boxB (list or tuple): Координаты второй рамки [xmin, ymin, xmax, ymax].

    Returns:
        float: Значение IoU (от 0.0 до 1.0).
               Возвращает 0.0, если нет пересечения или одна из рамок имеет нулевую площадь.
    """
    # --- Находим координаты прямоугольника пересечения ---
    # Левый верхний угол пересечения: берем максимальные из левых верхних углов A и B
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # Правый нижний угол пересечения: берем минимальные из правых нижних углов A и B
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # --- Вычисляем площадь пересечения ---
    # Ширина пересечения = xB - xA
    # Высота пересечения = yB - yA
    # Если xB < xA или yB < yA, значит, пересечения нет. max(0, ...) учтет это.
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # --- Вычисляем площади обеих исходных рамок ---
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # --- Вычисляем IoU по формуле: ПлощадьПересечения / ПлощадьОбъединения ---
    # ПлощадьОбъединения = ПлощадьA + ПлощадьB - ПлощадьПересечения
    denominator = float(boxAArea + boxBArea - interArea)

    # Избегаем деления на ноль, если объединение имеет нулевую площадь
    # (например, если обе рамки - точки или линии)
    if denominator == 0:
        return 0.0

    iou = interArea / denominator
    return iou


def non_max_suppression(boxes, scores, threshold):
    """
    Реализация алгоритма Non-Maximum Suppression (NMS).
    Убирает лишние, сильно перекрывающиеся рамки, оставляя только
    самые "уверенные" из каждой группы перекрывающихся.

    Эта функция написана "с нуля" и может быть альтернативой `cv2.dnn.NMSBoxes`
    или `imutils.object_detection.non_max_suppression`.

    Args:
        boxes (list or np.array): Список рамок в формате [[xmin, ymin, xmax, ymax], ...].
        scores (list or np.array): Список соответствующих уверенностей (scores) для каждой рамки.
        threshold (float): Порог IoU. Рамки с IoU > этого порога будут считаться
                           относящимися к одному объекту и подавляться.

    Returns:
        list: Список индексов тех рамок из исходного списка `boxes`,
              которые нужно оставить после NMS.
    """
    # Если рамок нет, то и делать нечего
    if len(boxes) == 0:
        return []

    # Если это не numpy массивы, конвертируем для удобства векторных операций
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    # --- Подготовка данных ---
    # Получаем координаты всех рамок для векторных вычислений
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Вычисляем площади всех рамок
    area = (x2 - x1) * (y2 - y1)

    # --- Основной цикл NMS ---
    # Сортируем индексы рамок по убыванию их уверенности (scores)
    # `argsort` дает индексы от меньшего к большему, `[::-1]` переворачивает
    idxs = np.argsort(scores)[::-1]

    pick = [] # Сюда будем собирать индексы рамок, которые мы решили оставить

    # Пока еще есть индексы для рассмотрения...
    while len(idxs) > 0:
        # Берем индекс последней рамки в текущем списке `idxs`.
        # Так как список отсортирован по убыванию score, это будет рамка
        # с НАИВЫСШЕЙ уверенностью среди оставшихся.
        last = len(idxs) - 1
        i = idxs[last]
        # Добавляем индекс этой "лучшей" рамки в наш итоговый список `pick`.
        pick.append(i)

        # --- Находим пересечение текущей лучшей рамки `i` со ВСЕМИ остальными ---
        # `idxs[:last]` - это все индексы в списке, кроме последнего (который мы уже выбрали)
        # Находим координаты пересечения (как в функции calculate_iou, но векторно)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Считаем ширину и высоту пересечения
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Считаем IoU между рамкой `i` и всеми остальными
        interArea = w * h
        unionArea = area[idxs[:last]] + area[i] - interArea
        # Избегаем деления на ноль
        overlap = np.divide(interArea, unionArea, out=np.zeros_like(interArea, dtype=float), where=unionArea!=0)

        # --- Удаляем "плохие" рамки ---
        # Нам нужно удалить из `idxs`:
        # 1. Индекс `last` (рамку `i`, которую мы уже выбрали).
        # 2. Все индексы `idxs[:last]`, для которых `overlap > threshold`.
        # `np.where(overlap > threshold)[0]` находит индексы (внутри среза `idxs[:last]`), где IoU слишком большой.
        # `np.concatenate` объединяет индекс `last` и индексы "плохих" рамок.
        # `np.delete` удаляет все эти индексы из `idxs`.
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    # Цикл завершился, в `pick` остались индексы только нужных рамок.
    # Возвращаем их (это будут индексы относительно ИСХОДНОГО списка `boxes`).
    return pick

# --- Вспомогательная функция для изображений ---

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Изменяет размер изображения до заданной ширины ИЛИ высоты,
    сохраняя при этом исходные пропорции.

    Args:
        image (np.ndarray): Входное изображение.
        width (int, optional): Желаемая ширина. Если указана, высота подгоняется.
        height (int, optional): Желаемая высота. Если указана, ширина подгоняется.
                                Если указаны и width, и height, приоритет у width.
        inter (int, optional): Метод интерполяции OpenCV (cv2.INTER_AREA, cv2.INTER_LINEAR и т.д.).
                               INTER_AREA хорош для уменьшения.

    Returns:
        np.ndarray: Измененное изображение.
                    Если width и height не указаны, возвращает исходное изображение.
    """
    dim = None # Здесь будут новые размеры (width, height)
    (h, w) = image.shape[:2] # Получаем текущие высоту и ширину

    # Если не заданы ни ширина, ни высота - ничего не делаем
    if width is None and height is None:
        return image

    # Если задана только высота
    if width is None:
        # Считаем пропорцию изменения высоты
        r = height / float(h)
        # Вычисляем новую ширину, сохраняя пропорцию, и используем заданную высоту
        dim = (int(w * r), height)
    # Если задана ширина (или и ширина, и высота - приоритет у ширины)
    else:
        # Считаем пропорцию изменения ширины
        r = width / float(w)
        # Используем заданную ширину и вычисляем новую высоту, сохраняя пропорцию
        dim = (width, int(h * r))

    # Изменяем размер изображения с вычисленными размерами и выбранным методом интерполяции
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# --- __init__.py для пакета src ---
# В файле person_detector_hog_svm/src/__init__.py можно оставить просто комментарий:
# # -*- coding: utf-8 -*-
# # Этот файл может быть пустым.
# # Он нужен, чтобы Python распознавал папку 'src' как пакет (package),
# # что позволяет использовать относительные импорты вида 'from . import module'.