FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --default-timeout=100 uvicorn fastapi numpy onnx onnxruntime pandas joblib scikit-learn

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]