# # 使用官方 Python 映像
# FROM python:3.12-slim

# 使用 NVIDIA CUDA Runtime (可用 GPU)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 安裝 Python 與 ffmpeg
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# 設定工作目錄
WORKDIR /app

# 先複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案內容進容器
COPY . .

# 預設執行命令
CMD ["python3", "main.py"]