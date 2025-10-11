FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Prophet, shap, etc.
RUN apt-get update && apt-get install -y build-essential libatlas-base-dev

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
