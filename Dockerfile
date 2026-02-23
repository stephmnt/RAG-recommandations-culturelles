FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY src /app/src
COPY configs /app/configs
COPY mistral /app/mistral
COPY config.yaml /app/config.yaml
COPY pytest.ini /app/pytest.ini

RUN mkdir -p /app/data /app/artifacts /app/logs /app/reports

EXPOSE 8000

CMD ["python", "app.py", "run-api", "--host", "0.0.0.0", "--port", "8000"]
