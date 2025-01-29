from fastapi import FastAPI
from app.routes.api_v1 import router as api_router

# Initialize the FastAPI app
app = FastAPI()

# Include the API router for versioned routes
app.include_router(api_router, prefix="/api/v1")
