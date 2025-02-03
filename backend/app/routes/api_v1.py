# backend/app/routes/api_v1.py
from fastapi import APIRouter, UploadFile, File
from app.tasks.celery_tasks import process_image

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open(f"temp_uploads/{file.filename}", "wb") as f:
        f.write(await file.read())
    
    # Trigger Celery task
    task = process_image.delay(file.filename)
    return {"task_id": task.id}