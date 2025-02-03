from fastapi import APIRouter, UploadFile, File
from app.tasks.celery_tasks import process_image
from fastapi.responses import FileResponse
from celery.result import AsyncResult
import os

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Ensure temp directory exists
    os.makedirs("temp_uploads", exist_ok=True)

    # Save uploaded file
    file_path = f"temp_uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Start Celery task
    task = process_image.delay(file.filename)
    return {"task_id": task.id}

@router.get("/task/{task_id}")
def get_task_result(task_id: str):
    task_result = AsyncResult(task_id)

    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "processing"}
    elif task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(task_result.result)}
    elif task_result.state == "SUCCESS":
        result = task_result.result  # This contains the processed image path
        return {"task_id": task_id, "status": "completed", "processed_image": result["processed_image"]}
    else:
        return {"task_id": task_id, "status": task_result.state}