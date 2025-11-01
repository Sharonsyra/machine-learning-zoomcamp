FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY hw05/pyproject.toml .
COPY app.py .

RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
