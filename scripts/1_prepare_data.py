# person_detector_hog_svm/scripts/1_prepare_data.py

import sys
import os

# Хитрость, чтобы Python мог найти наши модули в папке 'src'
# Нужно подняться на два уровня от текущего скрипта (scripts -> корень проекта)
# и добавить корень проекта в пути поиска модулей.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) # Теперь можно делать 'from src import ...'

# Подтягиваем нашу логику нарезки данных и настройки проекта
from src import data_preparation
from src import config # Заодно убедимся, что все пути в конфиге создались

# Этот блок выполнится, только если мы запускаем этот скрипт напрямую
# (а не импортируем его куда-то еще)
if __name__ == "__main__":
    print("--- Запускаем подготовку данных (нарезка сэмплов) ---")

    # Вызываем главную функцию нарезки сэмплов.
    # Все параметры берем из нашего центрального конфига, чтобы было удобно менять.
    data_preparation.extract_samples(
        image_dir=config.TRAIN_IMAGE_DIR,             # Откуда брать обучающие картинки
        label_dir=config.TRAIN_LABEL_DIR,             # Где лежат YOLO-аннотации к ним
        positive_output_dir=config.POSITIVE_SAMPLES_DIR, # Куда складывать вырезанных людей (позитивы)
        negative_output_dir=config.NEGATIVE_SAMPLES_DIR, # Куда складывать куски фона (негативы)
        window_size=config.WINDOW_SIZE,                 # К какому размеру приводить все сэмплы (важно для HOG!)
        neg_samples_per_image=config.NEG_SAMPLES_PER_IMAGE, # Сколько негативов резать с каждой картинки (для разнообразия)
        neg_iou_threshold=config.NEG_IOU_THRESHOLD,     # Макс. пересечение негатива с реальным человеком (чтобы фон был фоном)
        person_class_id=config.PERSON_CLASS_ID          # Какой ID в YOLO-файлах считать за человека (обычно 0)
    )

    print("--- Подготовка данных завершена ---")