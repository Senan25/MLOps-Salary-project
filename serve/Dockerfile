FROM python:3.9-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    libffi-dev \
    build-essential

WORKDIR /app

COPY . /app

# Install Python packages
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
