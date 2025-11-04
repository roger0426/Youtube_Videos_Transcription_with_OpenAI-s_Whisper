# 定義資料夾路徑
$FOLDER_o = "output"
$FOLDER_d = "downloads"

# 確保資料夾存在，若不存在就建立
if (-not (Test-Path $FOLDER_o)) {
    New-Item -ItemType Directory -Path $FOLDER_o | Out-Null
}
if (-not (Test-Path $FOLDER_d)) {
    New-Item -ItemType Directory -Path $FOLDER_d | Out-Null
}

$CONTAINER_NAME = "yt_transcript"
$IMAGE_NAME = "yt-transcript-app"

# 計算 Dockerfile, requirements.txt 和 main.py 的 hash
$filesToHash = "Dockerfile","requirements.txt","main.py"
$hashInput = ""
foreach ($file in $filesToHash) {
    if (Test-Path $file) {
        $hashInput += Get-Content $file -Raw
    }
}
$DOCKERFILE_HASH = [System.BitConverter]::ToString((New-Object Security.Cryptography.SHA256Managed).ComputeHash([System.Text.Encoding]::UTF8.GetBytes($hashInput))).Replace("-", "").ToLower()

# 記錄上次 build hash
$HASH_FILE = ".docker_build_hash"

# 是否需要 rebuild
$REBUILD = $false

# 判斷 Docker image 是否存在
try {
    docker image inspect $IMAGE_NAME | Out-Null
} catch {
    Write-Host "⚠️ Docker image '$IMAGE_NAME' 不存在，需要重新 build。"
    $REBUILD = $true
}

# 判斷 hash 是否不同
if (-not (Test-Path $HASH_FILE)) {
    $REBUILD = $true
} else {
    $OLD_HASH = Get-Content $HASH_FILE -Raw
    if ($DOCKERFILE_HASH -ne $OLD_HASH) {
        $REBUILD = $true
    }
}

if ($REBUILD) {
    Write-Host "🚀 Rebuilding Docker image..."

    # 若有舊 container，先刪掉
    $existingContainer = docker ps -aq -f "name=^$CONTAINER_NAME$"
    if ($existingContainer) {
        Write-Host "🧹 Removing old container..."
        docker rm -f $CONTAINER_NAME | Out-Null
    }

    docker build -t $IMAGE_NAME .
    $DOCKERFILE_HASH | Out-File $HASH_FILE -Encoding ascii
    Write-Host "✅ Docker image rebuilt and hash updated."
} else {
    Write-Host "✅ Docker image is up-to-date."
}

# 檢查容器是否存在
$containerExists = docker ps -aq -f "name=^$CONTAINER_NAME$"
if ($containerExists) {
    Write-Host "🔹 Container exists."
    $running = docker ps -q -f "name=^$CONTAINER_NAME$"
    if ($running) {
        docker logs -f $CONTAINER_NAME
    } else {
        docker start -ai $CONTAINER_NAME
    }
} else {
    Write-Host "🚀 Creating and running new container..."
    docker run -it --name $CONTAINER_NAME `
        --gpus all `
        --env-file .env `
        -v "${PWD}\downloads:/app/downloads" `
        -v "${PWD}\output:/app/output" `
        $IMAGE_NAME
}