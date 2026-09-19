# GPU use: run with NVIDIA Container Toolkit and install a CUDA-compatible PyTorch build.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-gui.txt requirements-models.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-gui.txt -r requirements-models.txt
COPY . .
EXPOSE 7860
CMD ["python", "studio.py", "--host", "0.0.0.0", "--port", "7860"]
