FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code, including campus_model.pth and the database folder
COPY . .

ENV PORT=8080

# Updated to main:app to match your main.py filename
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]