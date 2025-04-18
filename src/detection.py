# person_detector_hog_svm/src/detection.py

import cv2                     # OpenCV для работы с изображениями и HOG
import numpy as np             # NumPy для массивов (веса, рамки)
import joblib                  # Для загрузки сохраненной модели SVM
import json                    # Для загрузки параметров HOG из JSON
import os                      # Работа с файловой системой (пути, проверка существования)
import sys                     # Для добавления пути к 'src'

# Снова наш трюк с путем, чтобы импорты из 'src' работали
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем наш конфиг и утилиты
from src import config # Все настройки (пути, параметры детекции)
from src import utils  # Вспомогательные функции (вдруг понадобятся)

def load_detector(model_path, hog_params_path):
    """
    Загружает обученную SVM модель и параметры HOG, а затем
    создает и настраивает HOGDescriptor для детекции.

    Args:
        model_path (str): Путь к файлу модели SVM (.joblib).
        hog_params_path (str): Путь к файлу с параметрами HOG (.json).

    Returns:
        cv2.HOGDescriptor: Готовый к использованию HOG детектор,
                           в который уже "вшит" наш обученный SVM.
        None: Если что-то пошло не так при загрузке.
    """
    print("Загружаем обученную модель SVM и параметры HOG...")
    # Проверяем, на месте ли файлы модели и параметров
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели SVM не найден по пути: {model_path}")
        return None
    if not os.path.exists(hog_params_path):
        print(f"Ошибка: Файл параметров HOG не найден по пути: {hog_params_path}")
        return None

    try:
        # Загружаем модель SVM (сохраненную через joblib)
        print(f"Загрузка SVM модели из {model_path}...")
        model = joblib.load(model_path)
        print("SVM модель загружена.")

        # Загружаем параметры HOG из JSON файла
        print(f"Загрузка параметров HOG из {hog_params_path}...")
        with open(hog_params_path, 'r') as f:
            hog_params_loaded = json.load(f)
        print("Параметры HOG загружены.")

        # Важно! В JSON размеры могли сохраниться как списки,
        # а HOGDescriptor ожидает кортежи (tuple). Конвертируем обратно.
        win_size = tuple(hog_params_loaded['winSize'])
        block_size = tuple(hog_params_loaded['blockSize'])
        block_stride = tuple(hog_params_loaded['blockStride'])
        cell_size = tuple(hog_params_loaded['cellSize'])

        # Создаем HOG дескриптор с ЗАГРУЖЕННЫМИ параметрами.
        # Крайне важно использовать те же параметры, с которыми считались признаки для обучения!
        print("Инициализация HOG дескриптора...")
        hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size,
            hog_params_loaded['nbins'], hog_params_loaded['derivAperture'],
            hog_params_loaded['winSigma'], hog_params_loaded['histogramNormType'],
            hog_params_loaded['L2HysThreshold'], hog_params_loaded['gammaCorrection'],
            hog_params_loaded['nlevels'], hog_params_loaded['signedGradients']
        )
        print("HOG дескриптор создан.")

        # --- Главная магия HOG+SVM ---
        # Получаем веса (коэффициенты) и свободный член из обученной модели SVM
        # и "вшиваем" их прямо в HOG дескриптор. Теперь HOG знает, как классифицировать.
        # Формат особый: сначала все веса подряд, потом свободный член.
        print("Установка SVM детектора в HOG...")
        svm_detector_weights = model.coef_.ravel() # Веса признаков (делаем плоским массивом)
        svm_intercept = model.intercept_         # Свободный член (смещение)
        # Объединяем их в один массив
        svm_detector = np.append(svm_detector_weights, svm_intercept)
        # Устанавливаем этот объединенный вектор в HOG
        hog.setSVMDetector(svm_detector)
        print("SVM детектор успешно установлен в HOG.")

        # Возвращаем полностью настроенный HOG-объект
        return hog

    except Exception as e:
        # Если на любом этапе загрузки или настройки произошла ошибка
        print(f"Критическая ошибка при загрузке компонентов детектора: {e}")
        import traceback
        traceback.print_exc() # Показываем детальную информацию об ошибке
        return None


def detect_people(image, hog_detector, win_stride=(8, 8), padding=(16, 16),
                  scale=1.05, hit_threshold=0, use_nms=True, nms_threshold=0.2):
    """
    Основная функция детекции. Применяет HOG+SVM детектор к изображению.

    Args:
        image (np.ndarray): Входное изображение (желательно цветное, BGR).
        hog_detector (cv2.HOGDescriptor): Загруженный и настроенный HOG детектор.
        win_stride (tuple): Шаг скользящего окна при поиске.
        padding (tuple): Отступы вокруг рамки при вычислении HOG.
        scale (float): Коэффициент масштабирования для пирамиды изображений.
        hit_threshold (float): Порог уверенности SVM. Детекции с весом >= этого значения учитываются.
        use_nms (bool): Применять ли Non-Maximum Suppression для удаления дубликатов.
        nms_threshold (float): Порог IoU для NMS.

    Returns:
        tuple:
            - list: Финальный список рамок [[x, y, w, h], ...] обнаруженных людей.
            - int: Количество найденных людей (длина списка рамок).
            - np.ndarray: Копия входного изображения с нарисованными рамками и счетчиком.
    """
    # Базовые проверки входных данных
    if image is None:
        print("Ошибка: Входное изображение - None.")
        return [], 0, None # Возвращаем пустые результаты
    if hog_detector is None:
        print("Ошибка: HOG детектор не инициализирован (None).")
        # Вернем исходную картинку, т.к. обработать не можем
        return [], 0, image

    # Создаем копию изображения, чтобы рисовать на ней, не портя оригинал
    output_image = image.copy()
    # Инициализируем списки для сырых результатов от HOG
    rects = []    # Список рамок (x, y, w, h)
    weights = []  # Список весов (уверенность SVM) для каждой рамки

    # --- Запуск HOG детекции ---
    try:
        # Самый главный вызов: ищем объекты на разных масштабах
        # HOGDescriptor.detectMultiScale сама строит пирамиду и двигает окно
        rects, weights = hog_detector.detectMultiScale(
            image,
            hitThreshold=hit_threshold,  # Используем наш порог уверенности
            winStride=win_stride,        # Шаг окна
            padding=padding,             # Отступы
            scale=scale,                 # Масштаб пирамиды
            useMeanshiftGrouping=False   # Группировку оставим для NMS, здесь выключим
        )
        # Если детекция прошла успешно, rects будет списком найденных рамок,
        # а weights - списком соответствующих им "очков" от SVM.
    except Exception as e:
        # Если внутри detectMultiScale что-то сломалось
        print(f"Ошибка во время выполнения HOG detectMultiScale: {e}")
        # В этом случае rects и weights останутся пустыми (как инициализировали)
        # Обработка пустых списков ниже корректно вернет 0 детекций.
        pass # Просто продолжаем, ошибка уже выведена

    # ---- Подавление Не-Максимумов (Non-Maximum Suppression - NMS) ----
    # Цель: Убрать лишние, сильно перекрывающиеся рамки для одного и того же объекта.

    final_boxes_xywh = [] # Сюда будем складывать финальные рамки после NMS

    # Обрабатываем только если HOG вообще что-то нашел
    if len(rects) > 0:
        # Преобразуем веса в numpy массив нужного типа (float32) и формы (плоский)
        scores = np.array(weights).flatten().astype(np.float32)

        # Важная проверка на случай, если detectMultiScale вернула разное
        # количество рамок и весов (очень редкая, но возможная ошибка)
        if len(rects) != len(scores):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Несовпадение числа рамок ({len(rects)}) и весов ({len(scores)}) после HOG. Пропускаем NMS.")
            # В таком аварийном случае просто берем все рамки как есть
            final_boxes_xywh = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

        # Если количество совпадает и NMS включен (use_nms=True)...
        elif use_nms:
            # Используем встроенный NMS из OpenCV (cv2.dnn.NMSBoxes)
            # Он чуть капризнее к формату входных данных.

            # Порог уверенности для NMS. Можно поставить низкий, т.к. основной отсев был по hit_threshold.
            # Но не 0, чтобы отсечь совсем неуверенные срабатывания, если hit_threshold < 0.
            nms_score_threshold = 0.01

            # NMSBoxes ожидает список списков рамок [[x, y, w, h], ...]
            boxes_for_nms = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in rects]

            # Отладочные принты (можно раскомментировать, если что-то идет не так с NMS)
            # print(f"Debug: Boxes for NMS = {boxes_for_nms}")
            # print(f"Debug: Scores for NMS = {scores}")
            # print(f"Debug: nms_score_threshold = {nms_score_threshold}")
            # print(f"Debug: nms_threshold (IoU) = {nms_threshold}")

            try:
                # Запускаем NMS! Он возвращает индексы тех рамок, которые нужно оставить.
                indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, nms_score_threshold, nms_threshold)

                # Проверяем, остались ли какие-то рамки после NMS
                if len(indices) > 0:
                    # Иногда NMS возвращает индексы в виде [[0], [2], ...], делаем их плоским списком [0, 2, ...]
                    if indices.ndim > 1:
                        indices = indices.flatten()

                    # Отладочные принты
                    # print(f"Debug: Indices after NMS = {indices}")
                    # print(f"Debug: Max index = {np.max(indices) if len(indices) > 0 else 'N/A'}")
                    # print(f"Debug: Number of boxes before NMS = {len(boxes_for_nms)}")

                    # Дополнительная проверка: не вернул ли NMS некорректный индекс?
                    max_index = np.max(indices)
                    if max_index >= len(boxes_for_nms):
                        print(f"ОШИБКА: NMS вернул некорректный индекс ({max_index}), превышающий число рамок ({len(boxes_for_nms)}). Используем рамки до NMS.")
                        final_boxes_xywh = boxes_for_nms # Берем все рамки до NMS
                    else:
                        # Все хорошо, выбираем только те рамки, индексы которых вернул NMS
                        final_boxes_xywh = [boxes_for_nms[i] for i in indices]

            except Exception as nms_error:
                # Если сам вызов NMS вызвал ошибку
                print(f"Ошибка при выполнении cv2.dnn.NMSBoxes: {nms_error}")
                print("Пропускаем NMS из-за ошибки. Используем все рамки до NMS.")
                # Берем все исходные рамки HOG
                final_boxes_xywh = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in rects]

        # Если NMS выключен (use_nms=False)
        else:
             # Просто берем все рамки, найденные HOG
             final_boxes_xywh = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in rects]

    # Если HOG ничего не нашел (len(rects) == 0), то final_boxes_xywh останется пустым []

    # ---- Рисуем результаты ----
    # Проходим по финальному списку рамок
    for (x, y, w, h) in final_boxes_xywh:
        # Проверяем, что ширина и высота положительные (защита от аномальных рамок)
        if w > 0 and h > 0:
            # Рисуем зеленую рамку толщиной 2 пикселя
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Добавляем текст с количеством найденных людей
    count = len(final_boxes_xywh)
    label = f"People Count: {count}" # Формируем строку
    # Наносим текст на картинку (координаты (10, 30), шрифт, масштаб, цвет, толщина)
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Возвращаем список финальных рамок, их количество и картинку с результатами
    return final_boxes_xywh, count, output_image


# ---- Блок для самостоятельного запуска и тестирования ----
# Выполняется, только если запустить файл напрямую: python src/detection.py
if __name__ == "__main__":
    print("Запускаем detection.py как основной скрипт для теста...")

    # Загружаем наш детектор
    detector = load_detector(config.SVM_MODEL_PATH, config.HOG_PARAMS_JSON)

    # Если детектор загрузился успешно...
    if detector:
        # Пытаемся найти тестовое изображение для примера
        test_image_name = None
        test_image_path = None
        # Проверяем, существует ли папка с тестовыми изображениями из конфига
        if os.path.isdir(config.TEST_IMAGE_DIR):
            try:
                # Ищем все файлы картинок в этой папке
                image_files = [f for f in os.listdir(config.TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Если нашли хотя бы одну...
                if image_files:
                    # Берем первую попавшуюся
                    test_image_name = image_files[0]
                    test_image_path = os.path.join(config.TEST_IMAGE_DIR, test_image_name)
                    print(f"Информация: Используем первое найденное тестовое изображение: {test_image_path}")
                else:
                    print(f"Предупреждение: В папке {config.TEST_IMAGE_DIR} не найдено тестовых изображений.")
            except Exception as e:
                 # Если ошибка при чтении содержимого папки
                 print(f"Ошибка при поиске тестового изображения в {config.TEST_IMAGE_DIR}: {e}")
        else:
            # Если самой папки не существует
            print(f"Ошибка: Папка для тестовых изображений не найдена: {config.TEST_IMAGE_DIR}")


        # Если удалось найти путь к тестовой картинке и такой файл существует...
        if test_image_path and os.path.exists(test_image_path):
            # Читаем изображение
            image = cv2.imread(test_image_path)

            # Если изображение успешно прочитано...
            if image is not None:
                print(f"Запускаем детекцию на изображении: {test_image_path}...")

                # Вызываем нашу основную функцию детекции с параметрами из конфига
                final_boxes, count, output_img = detect_people(
                    image,
                    detector,
                    win_stride=tuple(config.HOG_PARAMS['blockStride']), # Шаг окна берем из параметров HOG
                    padding=(16, 16),                    # Отступы (можно тоже в конфиг)
                    scale=config.DETECTION_PYRAMID_SCALE,# Масштаб пирамиды из конфига
                    hit_threshold=config.DETECTION_THRESHOLD, # Порог уверенности из конфига
                    use_nms=True,                        # Включаем NMS
                    nms_threshold=config.NMS_THRESHOLD   # Порог NMS из конфига
                )

                print(f"Детекция завершена. Найдено людей: {count}.")

                # Пытаемся показать результат на экране
                try:
                    cv2.imshow("Detections", output_img)
                    print("Нажмите любую клавишу в окне с изображением, чтобы закрыть...")
                    cv2.waitKey(0) # Ждем нажатия любой клавиши
                    cv2.destroyAllWindows() # Закрываем все окна OpenCV
                except cv2.error as e:
                    # Если не можем показать окно (например, нет графической оболочки)
                    print(f"Предупреждение: Не удалось показать изображение (GUI недоступен или другая ошибка): {e}")
                    print("Вместо этого просто сохраним результат в файл.")

                # Сохраняем изображение с результатами в папку 'output/detections'
                save_path = os.path.join(config.DETECTIONS_DIR, f"detected_{test_image_name}")
                try:
                    cv2.imwrite(save_path, output_img)
                    print(f"Результат детекции сохранен в: {save_path}")
                except Exception as e:
                    print(f"Ошибка при сохранении результата детекции: {e}")

            else: # Если cv2.imread вернул None
                print(f"Ошибка: Не удалось прочитать тестовое изображение: {test_image_path}")
        else:
            # Если путь не найден или файл по пути не существует
            if test_image_path and not os.path.exists(test_image_path):
                 print(f"Ошибка: Тестовый файл изображения не существует по пути: {test_image_path}")
            # Если test_image_path вообще не был найден, сообщение об ошибке было выведено выше

    else: # Если detector is None
        print("Критическая ошибка: Детектор не был загружен. Запуск детекции невозможен.")

    print("Завершение работы тестового блока detection.py.")