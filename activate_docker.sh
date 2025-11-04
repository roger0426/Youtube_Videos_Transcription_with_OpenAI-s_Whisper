#!/bin/bash

# 定義資料夾路徑
FOLDER_o="output"
FOLDER_d="downloads"

# 確保資料夾存在，若不存在就建立
mkdir -p "$FOLDER_o"
mkdir -p "$FOLDER_d"

CONTAINER_NAME="yt_transcript"
IMAGE_NAME="yt-transcript-app"

# 判斷 Dockerfile 或 requirements.txt 是否修改
DOCKERFILE_HASH=$(cat Dockerfile requirements.txt main.py | sha256sum | awk '{print $1}')

# 記錄上次 build hash
HASH_FILE=".docker_build_hash"

# 是否需要 rebuild
REBUILD=false

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "⚠️ Docker image '${IMAGE_NAME}' 不存在，需要重新 build。"
    REBUILD=true
fi

if [ ! -f "$HASH_FILE" ]; then
    REBUILD=true
else
    OLD_HASH=$(cat $HASH_FILE)
    if [ "$DOCKERFILE_HASH" != "$OLD_HASH" ]; then
        REBUILD=true
    fi
fi

if [ "$REBUILD" = true ]; then
    echo "🚀 Rebuilding Docker image..."

    # 若有舊 container，先刪掉
    if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
        echo "🧹 Removing old container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
    fi

    docker build -t $IMAGE_NAME .
    echo "$DOCKERFILE_HASH" > "$HASH_FILE"
    echo "✅ Docker image rebuilt and hash updated."
else
    echo "✅ Docker image is up-to-date."
fi

# 檢查容器是否存在
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "🔹 Container exists."
    RUNNING=$(docker ps -q -f name=^${CONTAINER_NAME}$)
    if [ "$RUNNING" ]; then
        docker logs -f $CONTAINER_NAME
    else
        docker start -ai $CONTAINER_NAME
    fi
else
    echo "🚀 Creating and running new container..."
    docker run -it --name $CONTAINER_NAME \
        --gpus all \
        --env-file .env \
        -v $(pwd)/downloads:/app/downloads \
        -v $(pwd)/output:/app/output \
        $IMAGE_NAME
fi