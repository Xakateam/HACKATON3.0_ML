# person_detector_hog_svm/scripts/2_train_model.py

import sys
import os
import time

# Опять же, обеспечиваем видимость модулей из папки 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Импортируем наши модули для вычисления признаков (HOG) и обучения (SVM)
# Заметьте, мы не будем запускать их __main__ блоки, а будем вызывать конкретные функции.
# Это более правильный подход, чем просто импортировать и надеяться, что оно само заработает.
from src import feature_extraction
from src import training
from src import config # Загружаем конфиг с путями и параметрами

# ---- Разбили логику на функции для ясности ----
# Раньше это могло быть прямо в __main__, но так чище.

def run_feature_extraction():
    """ Готовит HOG-признаки для всех позитивных и негативных сэмплов. """
    print("\n--- Запускаем извлечение HOG-признаков ---")
    start_time = time.time() # Засекаем время начала
    try:
        # Создаем HOG-дескриптор с параметрами из нашего конфига.
        # Важно, чтобы эти параметры (особенно winSize) совпадали с теми,
        # что использовались при подготовке данных и будут использоваться при детекции.
        print("Инициализируем HOG дескриптор...")
        hog = feature_extraction.cv2.HOGDescriptor(
            config.HOG_PARAMS['winSize'],        # Размер окна (должен быть как у сэмплов)
            config.HOG_PARAMS['blockSize'],      # Размер блока для нормализации
            config.HOG_PARAMS['blockStride'],    # На сколько блок смещается
            config.HOG_PARAMS['cellSize'],       # Размер ячейки гистограммы градиентов
            config.HOG_PARAMS['nbins'],          # Сколько "корзин" в гистограмме
            # --- остальные параметры HOG ---
            config.HOG_PARAMS['derivAperture'],
            config.HOG_PARAMS['winSigma'],
            config.HOG_PARAMS['histogramNormType'],
            config.HOG_PARAMS['L2HysThreshold'],
            config.HOG_PARAMS['gammaCorrection'],
            config.HOG_PARAMS['nlevels'],         # Для detectMultiScale, не так важно для compute
            config.HOG_PARAMS['signedGradients'] # Использовать ли направление градиента
        )
        print("HOG дескриптор готов.")

        # Считаем HOG-фичи для всех картинок в папках с позитивными...
        positive_features = feature_extraction.compute_hog_features_for_samples(
            config.POSITIVE_SAMPLES_DIR, hog, config.WINDOW_SIZE # Передаем сам дескриптор и размер окна
        )
        # ...и негативными сэмплами.
        negative_features = feature_extraction.compute_hog_features_for_samples(
            config.NEGATIVE_SAMPLES_DIR, hog, config.WINDOW_SIZE
        )

        # Важная проверка: если фичи не извлеклись (папки пустые, картинки битые), то дальше нет смысла.
        if not positive_features or not negative_features:
            print("Ошибка: Не удалось извлечь HOG-признаки. Проверьте папки с сэмплами.")
            return False # Сигнализируем об ошибке

        # Готовим данные для обучения SVM:
        # Позитивам ставим метку 1, негативам - 0.
        positive_labels = feature_extraction.np.ones(len(positive_features))
        negative_labels = feature_extraction.np.zeros(len(negative_features))

        # Склеиваем все фичи в один большой numpy-массив (матрицу признаков)
        # и все метки в один вектор. Важно указать тип float32 для фич!
        all_features = feature_extraction.np.vstack((positive_features, negative_features)).astype(feature_extraction.np.float32)
        all_labels = feature_extraction.np.concatenate((positive_labels, negative_labels)).astype(feature_extraction.np.int32)

        print("-" * 30)
        print(f"Всего признаков: {all_features.shape[0]}, Длина вектора признаков: {all_features.shape[1]}")
        print("-" * 30)

        # Сохраняем результат в файлы .npy (удобный формат numpy)
        print("Сохраняем признаки и метки в файлы...")
        feature_extraction.np.save(config.FEATURES_PATH, all_features) # Путь из конфига
        feature_extraction.np.save(config.LABELS_PATH, all_labels)     # Путь из конфига
        # Также сохраним параметры HOG, с которыми мы считали фичи. Это важно для детекции!
        feature_extraction.save_hog_params(config.HOG_PARAMS, config.HOG_PARAMS_JSON)
        print("Признаки, метки и параметры HOG сохранены.")

    except Exception as e:
        # Ловим любую другую неожиданную ошибку
        print(f"Произошла ошибка при извлечении признаков: {e}")
        import traceback
        traceback.print_exc() # Печатаем полный стектрейс для отладки
        return False # Сигнализируем об ошибке
    end_time = time.time() # Засекаем время окончания
    print(f"--- Извлечение признаков завершено за {end_time - start_time:.2f} сек ---")
    return True # Все прошло успешно


def run_training():
    """ Обучает SVM модель на подготовленных HOG-признаках. """
    print("\n--- Запускаем обучение SVM модели ---")
    start_time = time.time()
    try:
        # Вызываем функцию обучения из модуля training
        training.train_svm_model(
            features_path=config.FEATURES_PATH,   # Путь к сохраненным фичам
            labels_path=config.LABELS_PATH,     # Путь к сохраненным меткам
            model_save_path=config.SVM_MODEL_PATH, # Куда сохранить обученную модель
            svm_c=config.SVM_C,                 # Параметр регуляризации SVM (из конфига)
            svm_dual=config.SVM_DUAL            # Параметр dual для LinearSVC (из конфига)
        )
    except Exception as e:
        print(f"Произошла ошибка во время обучения SVM: {e}")
        import traceback
        traceback.print_exc()
        return False
    end_time = time.time()
    print(f"--- Обучение SVM завершено за {end_time - start_time:.2f} сек ---")
    return True

# ---- Основной блок скрипта ----
if __name__ == "__main__":
    print("--- Запускаем пайплайн обучения модели ---")
    overall_start_time = time.time() # Общее время старта

    # Шаг 1: Извлекаем признаки HOG
    # Функция вернет True, если все ОК, и False при ошибке.
    if run_feature_extraction():
        # Шаг 2: Обучаем SVM (только если признаки успешно готовы)
        if run_training():
            print("\nПайплайн обучения успешно завершен!")
        else:
            print("\nОшибка: Пайплайн обучения прерван на шаге тренировки SVM.")
    else:
        print("\nОшибка: Пайплайн обучения прерван на шаге извлечения признаков.")

    overall_end_time = time.time() # Общее время окончания
    print(f"--- Общее время выполнения пайплайна: {overall_end_time - overall_start_time:.2f} сек ---")