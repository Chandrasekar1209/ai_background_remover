FROM python:3.9-slim
WORKDIR /app

# Copy requirements.txt first to leverage Docker's caching mechanism
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else from the backend folder
COPY ./ .   
# This will copy everything from the build context to the Docker image

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
