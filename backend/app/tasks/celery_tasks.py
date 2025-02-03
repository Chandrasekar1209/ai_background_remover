# backend/app/tasks/celery_tasks.py
from celery import Celery
import torch
from ai_model.models.unet_model import UNet
import cv2

celery = Celery(__name__, broker="redis://redis:6379/0")

@celery.task(bind=True)
def process_image(self, filename):
    try:
        # Load AI model
        model = UNet()
        model.load_state_dict(torch.load("ai_model/weights/model_weights.pth"))
        print("Model loaded.")

        # Process image
        img = cv2.imread(f"temp_uploads/{filename}")
        mask = model(torch.tensor(img).permute(2, 0, 1).unsqueeze(0))

        # Save result
        output_path = f"static/processed_images/{filename}"
        cv2.imwrite(output_path, mask)

        return {"status": "completed", "processed_image": output_path}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
