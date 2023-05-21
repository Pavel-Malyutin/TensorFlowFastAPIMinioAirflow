FROM tiangolo/uwsgi-nginx:python3.9

WORKDIR /app

COPY ./app ./app

COPY requirements.txt /app/requirements.txt
COPY .env /app/.env

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENV PYTHONUNBUFFERED 1

CMD ["gunicorn", "--bind", ":8080", "app.main:app", "--worker-class", "uvicorn.workers.UvicornH11Worker", "--timeout", "300", "--max-requests", "100", "--backlog", "2048", "--workers", "2", "--threads", "4", "--log-level", "debug"]