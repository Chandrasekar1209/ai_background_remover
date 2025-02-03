from fastapi import FastAPI
from app.routes.api_v1 import router as api_router
from fastapi.middleware.cors import CORSMiddleware  

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 👈 Allows all origins (Change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # 👈 Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # 👈 Allows all headers
)

# Include the API router for versioned routes
app.include_router(api_router, prefix="/api/v1")
