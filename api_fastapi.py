# person_detector_hog_svm/api_fastapi.py

import os
import cv2
import numpy as np
import traceback
import io

# --- FastAPI ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# --- Импорт логики вашего проекта ---
from src import detection
from src import config

# --- Инициализация FastAPI ---
# Добавляем метаданные для автодокументации
app = FastAPI(
    title="Person Detector API",
    description="API для детекции людей на изображении с использованием HOG+SVM.",
    version="1.0.0"
)

# --- Глобальные переменные ---
# Загружаем детектор ОДИН РАЗ при старте API
print("Loading HOG detector for API...")
try:
    hog_detector = detection.load_detector(config.SVM_MODEL_PATH, config.HOG_PARAMS_JSON)
    if hog_detector is None:
        # Выбрасываем исключение, если модель критически не загрузилась
        raise RuntimeError("Could not load the HOG detector model.")
    print("Detector loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR during detector loading: {e}")
    print(traceback.format_exc())
    # Устанавливаем в None, чтобы API возвращал ошибку, если старт все же произойдет
    hog_detector = None


# --- Базовая модель ответа (не обязательно, но улучшает документацию) ---
# from pydantic import BaseModel
# class DetectionResponse(BaseModel):
#     success: bool
#     error: str | None = None
#     person_count: int


# --- API Endpoint ---
@app.post("/predict",
          # response_model=DetectionResponse, # Можно использовать Pydantic модель для ответа
          summary="Обнаружить людей на изображении",
          tags=["Detection"])
async def predict(image: UploadFile = File(..., description="Изображение для обработки (формат jpg, png и т.д.)")):
    """
    Принимает файл изображения и возвращает количество обнаруженных людей.

    - **image**: Файл изображения для загрузки.
    """
    # Проверяем, загрузился ли детектор при старте
    if hog_detector is None:
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail="Detector model is not loaded or failed to load."
        )

    # Проверяем тип файла (опционально, но полезно)
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, # Bad Request
            detail=f"Invalid file type: {image.content_type}. Please upload an image."
        )

    try:
        # Читаем байты изображения из UploadFile
        contents = await image.read()
        # Декодируем байты в изображение OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
             raise HTTPException(
                status_code=400, # Bad Request
                detail="Could not decode image file. Ensure it's a valid image format supported by OpenCV."
            )

        # --- Выполняем детекцию ---
        # Используем параметры из конфига
        final_boxes, count, output_img = detection.detect_people(
            img_cv,
            hog_detector,
            win_stride=(8, 8),
            padding=(16, 16),
            scale=config.DETECTION_PYRAMID_SCALE,
            hit_threshold=config.DETECTION_THRESHOLD,
            use_nms=True,
            nms_threshold=config.NMS_THRESHOLD
        )

        # --- Возвращаем результат ---
        # FastAPI автоматически конвертирует dict в JSONResponse
        return {
            "success": True,
            "error": None,
            "person_count": count
            # Можно добавить bounding boxes при необходимости:
            # "boxes": [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in final_boxes]
        }

    except HTTPException as e:
        # Перебрасываем HTTPException, которые мы сами сгенерировали
        raise e
    except Exception as e:
        print("Error during prediction:")
        print(traceback.format_exc()) # Печатаем полный стектрейс в консоль сервера
        # Для других ошибок возвращаем 500
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An internal error occurred during processing: {str(e)}"
        )
    finally:
        # Закрываем файл, если он был открыт (FastAPI обычно делает это сам, но для ясности)
        await image.close()


# --- Дополнительный endpoint для проверки статуса ---
@app.get("/health",
         summary="Проверить статус API",
         tags=["Health"])
async def health_check():
    """Возвращает статус загрузки модели."""
    if hog_detector is not None:
        return {"status": "OK", "message": "Detector is loaded."}
    else:
        # Используем JSONResponse для установки кастомного статус-кода
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={"status": "Error", "message": "Detector failed to load."}
        )


# --- Запуск (только для информации, т.к. запускать будем через uvicorn) ---
if __name__ == "__main__":
    print("To run the FastAPI application, use the command:")
    print("uvicorn api_fastapi:app --reload --host 0.0.0.0 --port 8000")
    # import uvicorn
    # uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True) # Можно и так, но стандартно через команду