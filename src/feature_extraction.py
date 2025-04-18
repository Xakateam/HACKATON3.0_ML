# person_detector_hog_svm/src/feature_extraction.py

import os      # Работа с путями и файлами
import cv2     # OpenCV для чтения картинок и вычисления HOG
import numpy as np # NumPy для работы с массивами признаков и меток
from tqdm import tqdm # Прогресс-бар для длинных циклов
import json    # Для сохранения параметров HOG в удобном формате
# import joblib # Закомментировали, т.к. joblib здесь не используется (он для моделей SVM)
import sys     # Для настройки путей импорта

# Снова добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем наш центральный конфиг
from src import config

def compute_hog_features_for_samples(sample_dir, hog_descriptor, window_size):
    """
    Проходит по всем картинкам (сэмплам) в указанной папке,
    вычисляет для каждой HOG-признаки и возвращает их списком.

    Args:
        sample_dir (str): Папка, где лежат картинки-сэмплы
                          (например, папка с позитивными или негативными примерами).
        hog_descriptor (cv2.HOGDescriptor): Уже инициализированный HOG-дескриптор
                                           с нужными параметрами.
        window_size (tuple): Ожидаемый размер (ширина, высота) картинок в папке.
                             Нужен для проверки и возможного ресайза.

    Returns:
        list: Список, где каждый элемент - это numpy-массив (вектор) HOG-признаков
              для одной картинки. Если папка пуста или не найдена, вернет пустой список.
    """
    features = [] # Сюда будем собирать векторы признаков

    # Проверяем, существует ли папка с сэмплами
    if not os.path.isdir(sample_dir):
        print(f"Предупреждение: Папка с сэмплами не найдена: {sample_dir}")
        return features # Возвращаем пустой список

    # Получаем список файлов картинок в папке
    filenames = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Если картинок нет, то и делать нечего
    if not filenames:
        print(f"Предупреждение: В папке {sample_dir} не найдено файлов изображений.")
        return features # Возвращаем пустой список

    # Получаем имя папки для вывода в прогресс-баре (чтобы было понятно, что обрабатывается)
    dir_name = os.path.basename(sample_dir)
    print(f"Вычисляем HOG-признаки для сэмплов в папке: {dir_name}")

    # Идем по всем найденным файлам картинок
    for filename in tqdm(filenames, desc=f"Обработка {dir_name}"):
        # Полный путь к файлу
        img_path = os.path.join(sample_dir, filename)
        # Читаем картинку СРАЗУ в оттенках серого, так как HOG работает именно с ними
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Если картинка не прочиталась
        if img is None:
            print(f"Предупреждение: Не удалось прочитать изображение {img_path}. Пропускаем.")
            continue # Переходим к следующему файлу

        # Дополнительная проверка и ресайз (на случай, если в папке оказались картинки не того размера)
        # Сравниваем реальные размеры (высота, ширина) с ожидаемыми (ширина, высота)
        if img.shape[1] != window_size[0] or img.shape[0] != window_size[1]:
            print(f"Предупреждение: Изображение {filename} имеет размер {img.shape[1]}x{img.shape[0]}, "
                  f"ожидался {window_size[0]}x{window_size[1]}. Приводим к нужному размеру.")
            # Используем INTER_AREA для уменьшения (или увеличения, но лучше чтобы все было одного размера изначально)
            img = cv2.resize(img, window_size, interpolation=cv2.INTER_AREA)

        # --- Вычисление HOG ---
        # `compute` возвращает вектор признаков для данного изображения
        hog_features = hog_descriptor.compute(img)

        # Иногда, для очень маленьких или странных картинок, compute может вернуть None. Проверяем.
        if hog_features is not None:
            # HOG признаки возвращаются как столбец, нам нужен плоский вектор.
            # Добавляем его в наш список признаков.
            features.append(hog_features.flatten())
        else:
            # Если HOG не посчитался - сообщаем об этом
            print(f"Предупреждение: Не удалось вычислить HOG для {img_path}. Пропускаем.")

    # Возвращаем список всех вычисленных векторов признаков
    return features

def save_hog_params(params, filepath):
    """
    Сохраняет словарь с параметрами HOG в JSON-файл.
    Это нужно, чтобы при детекции использовать ТОЧНО ТЕ ЖЕ параметры HOG,
    с которыми были вычислены признаки для обучения SVM.

    Args:
        params (dict): Словарь с параметрами HOG (из config.HOG_PARAMS).
        filepath (str): Путь к файлу JSON, куда сохранить параметры.
    """
    # JSON не умеет сохранять кортежи (tuple), только списки (list).
    # Поэтому создаем копию словаря и конвертируем все значения-кортежи в списки.
    params_serializable = params.copy()
    for key, value in params_serializable.items():
        if isinstance(value, tuple): # Проверяем, является ли значение кортежем
            params_serializable[key] = list(value) # Преобразуем в список
    try:
        # Открываем файл для записи ('w' - write)
        with open(filepath, 'w') as f:
            # Используем json.dump для записи словаря в файл
            # indent=4 делает файл читаемым (с отступами)
            json.dump(params_serializable, f, indent=4)
        print(f"Параметры HOG успешно сохранены в: {filepath}")
    except Exception as e:
        # Если при сохранении возникла ошибка (нет прав, нет места и т.д.)
        print(f"Ошибка при сохранении параметров HOG в {filepath}: {e}")


# ---- Блок для самостоятельного запуска ----
# Выполняется, если запустить файл напрямую: python src/feature_extraction.py
if __name__ == "__main__":
    print("Запускаем feature_extraction.py как основной скрипт...")

    # Инициализируем HOG дескриптор с параметрами из нашего конфига
    # Важно сделать это один раз здесь, а не создавать заново для каждой картинки.
    print("Инициализация HOG дескриптора с параметрами из config.py...")
    try:
        hog = cv2.HOGDescriptor(
            config.HOG_PARAMS['winSize'],
            config.HOG_PARAMS['blockSize'],
            config.HOG_PARAMS['blockStride'],
            config.HOG_PARAMS['cellSize'],
            config.HOG_PARAMS['nbins'],
            config.HOG_PARAMS['derivAperture'],
            config.HOG_PARAMS['winSigma'],
            config.HOG_PARAMS['histogramNormType'],
            config.HOG_PARAMS['L2HysThreshold'],
            config.HOG_PARAMS['gammaCorrection'],
            config.HOG_PARAMS['nlevels'],
            config.HOG_PARAMS['signedGradients']
        )
        print("HOG дескриптор успешно инициализирован.")
    except Exception as e:
        print(f"Критическая ошибка при инициализации HOG дескриптора: {e}")
        exit() # Выходим, если HOG не создался

    # Вычисляем признаки для позитивных сэмплов (людей)
    positive_features = compute_hog_features_for_samples(
        config.POSITIVE_SAMPLES_DIR, hog, config.WINDOW_SIZE # Передаем HOG и размер окна
    )

    # Вычисляем признаки для негативных сэмплов (фона)
    negative_features = compute_hog_features_for_samples(
        config.NEGATIVE_SAMPLES_DIR, hog, config.WINDOW_SIZE
    )

    # Проверяем, удалось ли извлечь хоть какие-то признаки. Если нет - дальше работать нет смысла.
    if not positive_features or not negative_features:
        print("Ошибка: Не удалось извлечь признаки для позитивных или негативных сэмплов.")
        print("Убедитесь, что скрипт 1_prepare_data.py отработал корректно и создал непустые папки с сэмплами.")
        exit() # Прерываем выполнение

    # --- Подготовка данных для обучения ---
    # Создаем метки: 1 для позитивных (люди), 0 для негативных (фон)
    positive_labels = np.ones(len(positive_features), dtype=np.int32) # Явно указываем тип int32
    negative_labels = np.zeros(len(negative_features), dtype=np.int32) # Явно указываем тип int32

    # Объединяем все признаки в одну большую матрицу (каждая строка - вектор HOG)
    # `vstack` ставит позитивные и негативные признаки друг под другом.
    # Важно привести к типу float32 - этого часто требуют алгоритмы машинного обучения.
    all_features = np.vstack((positive_features, negative_features)).astype(np.float32)
    # Объединяем метки в один вектор, соответствующий строкам матрицы признаков
    all_labels = np.concatenate((positive_labels, negative_labels))

    # Выводим статистику по собранным данным
    print("-" * 30)
    print(f"Всего позитивных признаков: {len(positive_features)}")
    print(f"Всего негативных признаков: {len(negative_features)}")
    print(f"Длина вектора признаков HOG: {all_features.shape[1]}") # Количество фичей в одном векторе
    print(f"Общее количество сэмплов для обучения: {all_features.shape[0]}")
    print(f"Размер итоговой матрицы признаков: {all_features.shape}")
    print(f"Размер итогового вектора меток: {all_labels.shape}")
    print("-" * 30)

    # --- Сохранение результатов ---
    # Сохраняем собранные признаки и метки в бинарные файлы NumPy (.npy)
    print("Сохраняем признаки и метки в .npy файлы...")
    try:
        np.save(config.FEATURES_PATH, all_features) # Путь берем из конфига
        np.save(config.LABELS_PATH, all_labels)     # Путь берем из конфига
        print(f"Признаки сохранены в: {config.FEATURES_PATH}")
        print(f"Метки сохранены в: {config.LABELS_PATH}")
    except Exception as e:
        # Если ошибка при сохранении
        print(f"Ошибка при сохранении файлов признаков/меток: {e}")
        exit() # Выходим

    # Сохраняем параметры HOG, использованные на этом шаге, в JSON
    # Это нужно, чтобы потом при детекции использовать те же самые параметры
    save_hog_params(config.HOG_PARAMS, config.HOG_PARAMS_JSON)

    print("Извлечение и сохранение признаков HOG успешно завершено.")