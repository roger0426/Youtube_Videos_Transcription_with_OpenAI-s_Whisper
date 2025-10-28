# 安裝必要套件 (若尚未安裝，請先在終端執行)
# pip install yt-dlp openai-whisper openai pydub psutil

from string import punctuation
import yt_dlp
import whisper
import openai
from pydub import AudioSegment
import os
import time
import re
import psutil
import json
import getpass
import zhconv
from dotenv import load_dotenv

# === 載入環境變數 ===
load_dotenv()

# === 清理檔案名稱的函數 ===
def clean_filename(filename):
    """清理檔案名稱，移除或替換不安全的字符"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = filename.strip('.')
    if len(filename) > 100:
        filename = filename[:100]
    return filename

# === 系統資源檢查 ===
def check_system_resources():
    """檢查系統資源，返回建議的模型大小"""
    try:
        # 檢查可用記憶體 (GB)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        # 檢查 GPU 記憶體 (如果可用)
        gpu_memory = 0
        gpu_type = "none"
        try:
            import torch
            if torch.backends.mps.is_available():
                # Mac GPU (M1/M2/M3)
                gpu_type = "mps"
                # Mac GPU 記憶體通常與系統記憶體共享，估算可用記憶體
                gpu_memory = available_memory * 0.5  # 假設一半可用給 GPU
            elif torch.cuda.is_available():
                # NVIDIA GPU
                gpu_type = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        print(f"系統記憶體: {available_memory:.1f} GB 可用")
        if gpu_memory > 0:
            print(f"GPU 類型: {gpu_type.upper()}")
            print(f"GPU 記憶體: {gpu_memory:.1f} GB")
        
        # 根據資源建議模型
        if gpu_memory >= 8:
            return "large-v3"
        elif gpu_memory >= 4 or available_memory >= 8:
            return "medium"
        elif available_memory >= 4:
            return "small"
        else:
            return "base"
            
    except Exception as e:
        print(f"檢查系統資源時發生錯誤: {e}")
        return "base"

# === API Key 管理 ===
def load_api_key():
    """從環境變數載入 API Key"""
    return os.getenv('OPENAI_API_KEY')

def save_api_key(api_key):
    """將 API Key 儲存到 .env 檔案"""
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(f'OPENAI_API_KEY={api_key}\n')
        print("✅ API Key 已儲存到 .env 檔案")
        return True
    except Exception as e:
        print(f"儲存 API Key 時發生錯誤: {e}")
        return False

def get_api_key():
    """取得 API Key，提供多種安全方式"""
    # 嘗試從 .env 檔案載入
    api_key = load_api_key()
    if api_key:
        print("✅ 從 .env 檔案讀取 API Key 成功")
        return api_key
    
    print("\n🔑 OpenAI API Key 設定")
    print("請選擇設定方式:")
    print("1. 手動輸入 (推薦)")
    print("2. 稍後設定（退出）")
    
    choice = input("請選擇 (1-2): ").strip()
    
    if choice == "1":
        print("\n請輸入你的 OpenAI API Key:")
        print("💡 提示: 輸入時不會顯示內容，這是正常的安全機制")
        api_key = getpass.getpass("API Key: ").strip()
        
        if api_key and len(api_key) > 20:  # 基本驗證
            print("是否要儲存 API Key 到 .env 檔案以便下次自動讀入？\n這會將 API Key 儲存在本機、本路徑下，請勿分享給他人。\n也建議您使用 .env 檔案來儲存 API Key，而不是手動輸入。\n強烈建議定期清理api key以防洩漏產生使用費用")
            save_choice = input("是否存至.env檔案？(y/n): ").strip().lower()
            if save_choice == 'y':
                save_api_key(api_key)
            return api_key
        else:
            print("❌ API Key 格式不正確")
            return None
    
    return None

# === 本地 Whisper 轉錄 ===
def transcribe_local(audio_path, model_size="base", language='zh'):
    """使用本地 Whisper 模型轉錄"""
    try:
        print(f"載入本地 Whisper 模型: {model_size}")
        
        # 檢查是否有 GPU 可用 (Mac M1/M2/M3)
        device = "cpu"
        use_gpu = False
        try:
            import torch
            if torch.backends.mps.is_available():
                # Mac GPU 有稀疏張量限制，直接使用 CPU 避免錯誤
                print("✅ 檢測到 Mac GPU (MPS)，但由於稀疏張量限制，可能會使用 CPU 以確保穩定性")
                device = "mps"
                use_gpu = True
            elif torch.cuda.is_available():
                device = "cuda"
                use_gpu = True
                print("✅ 檢測到 CUDA GPU，將使用 GPU 加速")
            else:
                print("⚠️  未檢測到 GPU，使用 CPU")
        except ImportError:
            print("⚠️  PyTorch 未安裝，使用 CPU")
        
        # 檢查是否為打包環境
        import sys
        is_packaged = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
        
        if is_packaged:
            print("📦 檢測到打包環境，使用特殊處理")
            # 在打包環境中，需要手動設定 Whisper 資源路徑
            try:
                import os
                # 設定 Whisper 資源目錄
                model_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
                os.makedirs(model_dir, exist_ok=True)
                # 設定環境變數
                os.environ['WHISPER_CACHE_DIR'] = model_dir
                print(f"設定 Whisper 快取目錄: {model_dir}")
            except Exception as e:
                print(f"⚠️  設定 Whisper 資源路徑時發生錯誤: {e}")
        
        # 嘗試載入模型並處理各種錯誤
        try:
            model = whisper.load_model(model_size, device=device)
            print(f"開始轉錄 (語言: {language}, 設備: {device})...")
        except Exception as model_error:
            error_msg = str(model_error).lower()
            
            if "sparse" in error_msg or "mps" in error_msg:
                print("❌ Mac GPU 遇到稀疏張量限制，降級到 CPU")
                device = "cpu"
                model = whisper.load_model(model_size, device=device)
                use_gpu = False
                print(f"開始轉錄 (語言: {language}, 設備: {device})...")
            elif "mel_filters" in error_msg or "assets" in error_msg:
                print("❌ Whisper 資源檔案缺失，嘗試重新下載...")
                # 嘗試清理快取並重新下載
                try:
                    import shutil
                    import os
                    cache_dir = os.path.expanduser("~/.cache/whisper")
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                        print("清理 Whisper 快取完成")
                except Exception as cleanup_error:
                    print(f"清理快取時發生錯誤: {cleanup_error}")
                
                # 重新載入模型
                model = whisper.load_model(model_size, device=device)
                print(f"開始轉錄 (語言: {language}, 設備: {device})...")
            else:
                raise model_error
        
        # 根據設備設定轉錄參數
        transcribe_kwargs = {
            'language': language, 
            'verbose': True, 
            'word_timestamps': True
        }
        
        # 只在支援的 GPU 上啟用 FP16
        if use_gpu and device == "cuda":
            transcribe_kwargs['fp16'] = True  # 只有 CUDA 完全支援 FP16
        elif use_gpu and device == "mps":
            # Mac GPU 不使用 FP16 避免稀疏張量問題
            print("⚠️  Mac GPU 不使用 FP16 以避免稀疏張量錯誤")
            
        result = model.transcribe(audio_path, **transcribe_kwargs)
        return result
    except Exception as e:
        print(f"本地轉錄時發生錯誤: {e}")
        return None

# === API Whisper 轉錄 ===
def transcribe_api(audio_file_path, client=None, language=None):
    """使用 OpenAI Whisper API 轉錄"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            if client is None:
                client = openai.OpenAI()
            response = client.audio.transcriptions.create(
                model="whisper-1",
                language=language,
                file=audio_file
            )
            return response.text
    except Exception as e:
        print(f"API 轉錄時發生錯誤: {e}")
        return None

# === GPT 標點符號處理 ===
def add_punctuation_with_gpt(text_chunk, client):
    """使用 GPT 為文字片段添加標點符號"""
    try:
        max_retry = 3
        for attempt in range(max_retry):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一個專業的文字編輯助手。請為以下沒有標點符號的文字添加適當的標點符號，包括句號、逗號、問號、驚嘆號等。保持原文的語意和結構，只添加標點符號。"
                    },
                    {
                        "role": "user", 
                        "content": f"請為以下文字添加標點符號：\n{text_chunk}"
                    }
                ],
                max_tokens=len(text_chunk)+500,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            len_orig = len(text_chunk)
            len_result = len(result)
            if len_orig == 0:
                break
            diff = abs(len_result - len_orig) / len_orig
            if diff <= 0.2 or attempt == max_retry - 1:
                print(f"✅ 標點符號處理完成！")
                print(f"處理前文字長度: {len_orig} 字元")
                print(f"處理後文字長度: {len(result)} 字元")
                return result
            # 否則重試
    except Exception as e:
        print(f"GPT 標點符號處理時發生錯誤: {e}")
        return text_chunk  # 如果失敗，返回原文

# === 批量標點符號處理 ===
def process_text_with_punctuation(text, client, text_len=1000):
    """將文字分割成片段，批量處理標點符號"""
    print(f"\n📝 開始標點符號處理...")
    print(f"原始文字長度: {len(text)} 字元")
    
    # 簡單按字元數分割，避免複雜的詞語分割邏輯
    chunks = []
    for i in range(0, len(text), text_len):
        chunk = text[i:i + text_len]
        chunks.append(chunk)
    
    print(f"分割成 {len(chunks)} 個片段")
    
    processed_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"--- 處理片段 {i}/{len(chunks)} ({len(chunk)} 字元)...")
        
        punctuated_chunk = add_punctuation_with_gpt(chunk, client)
        processed_chunks.append(punctuated_chunk)
        
        # 避免 API 限制
        if i < len(chunks):
            time.sleep(0.5)
    
    # 合併所有處理過的片段
    final_text = "".join(processed_chunks)
    print(f"✅ 標點符號處理完成！")
    print(f"處理後文字長度: {len(final_text)} 字元")
    
    return final_text

# === 清空預存模型 ===
def clear_whisper_models():
    """清空目前已存預存模型"""
    try:
        import shutil
        import os
        
        # Whisper 模型快取目錄
        cache_dir = os.path.expanduser("~/.cache/whisper")
        
        if not os.path.exists(cache_dir):
            print("✅ 沒有找到 Whisper 模型快取目錄")
            print("可能原因：")
            print("1. 尚未下載過任何 Whisper 模型")
            print("2. 模型儲存在其他位置")
            return True
        
        # 檢查目錄內容
        files = os.listdir(cache_dir)
        if not files:
            print("✅ Whisper 快取目錄為空，無需清理")
            return True
        
        print(f"📁 找到 Whisper 快取目錄: {cache_dir}")
        print(f"📊 目錄中包含 {len(files)} 個檔案/資料夾")
        
        # 顯示將要刪除的內容
        print("\n將要刪除的內容:")
        for file in files:
            file_path = os.path.join(cache_dir, file)
            if os.path.isdir(file_path):
                print(f"  📁 {file}/")
            else:
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                print(f"  📄 {file} ({size_mb:.1f} MB)")
        
        # 確認刪除
        print(f"\n⚠️  這將刪除所有 Whisper 模型檔案，釋放磁碟空間")
        print("下次使用時需要重新下載模型")
        confirm = input("確定要清空所有預存模型嗎？(y/n): ").strip().lower()
        
        if confirm == 'y':
            # 刪除整個快取目錄
            shutil.rmtree(cache_dir)
            print("✅ Whisper 模型快取已清空")
            print("💡 下次使用時會自動重新下載所需的模型")
            return True
        else:
            print("❌ 取消清空操作")
            return False
            
    except Exception as e:
        print(f"❌ 清空模型時發生錯誤: {e}")
        return False

# === 智能模型選擇 ===
def smart_model_selection(audio_path, language = None):
    """智能選擇最佳模型"""
    print("\n🤖 智能模型選擇模式")
    
    # 檢查音訊長度
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_minutes = len(audio) / (1000 * 60)
        print(f"音訊長度: {duration_minutes:.1f} 分鐘")
    except:
        duration_minutes = 0
    
    
    model_order = ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]
    
    recommended = check_system_resources()
    
    print(f"系統建議模型: {recommended}")
    print("將從建議模型開始嘗試，如果失敗會自動降級...")
    
    for model_size in model_order:
        print(f"\n嘗試使用 {model_size} 模型...")
        try:
            result = transcribe_local(audio_path, model_size, language)
            if result:
                print(f"✅ {model_size} 模型轉錄成功！")
                return result, model_size
            else:
                print(f"❌ {model_size} 模型轉錄失敗")
        except Exception as e:
            print(f"❌ {model_size} 模型發生錯誤: {e}")
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("檢測到記憶體不足，嘗試較小模型...")
                continue
            else:
                print("其他錯誤，嘗試較小模型...")
                continue
    
    print("❌ 所有本地模型都失敗了")
    return None, None

# === 取得影片資訊 ===
def get_video_info(youtube_url):
    """取得 YouTube 影片資訊而不下載"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"正在取得影片資訊: {youtube_url}")
            info = ydl.extract_info(youtube_url, download=False)
            title = info.get('title', 'unknown')
            duration = info.get('duration', 0)
            uploader = info.get('uploader', 'unknown')
            
            return {
                'title': title,
                'duration': duration,
                'uploader': uploader,
                'url': youtube_url
            }
    except Exception as e:
        print(f"取得影片資訊時發生錯誤: {e}")
        return None

# === 處理本地音訊文件 ===
def process_local_audio(file_path):
    """處理本地音訊文件，轉換為適合轉錄的格式"""
    try:
        # 檢查文件是否存在
        if not os.path.isfile(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None, None
        
        # 獲取文件信息
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 檢查是否為音訊文件
        audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        if file_ext in audio_extensions:
            print(f"✅ 檢測到音訊文件: {file_path}")
            return file_path, file_name
        elif file_ext in video_extensions:
            print(f"📹 檢測到影片文件: {file_path}")
            print("將直接使用此文件進行轉錄...")
            return file_path, file_name
        else:
            print(f"⚠️  未知文件格式: {file_ext}")
            print("嘗試直接使用此文件...")
            return file_path, file_name
            
    except Exception as e:
        print(f"處理本地文件時發生錯誤: {e}")
        return None, None

# === 下載 YouTube 音訊 ===
def download_audio(youtube_url, audio_path="downloads"):
    """下載 YouTube 音訊檔案"""
    try:
        if not os.path.exists(audio_path):
            os.makedirs(audio_path)
        
        # 先取得影片資訊以獲取標題
        ydl_info_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            title = info.get('title', 'unknown')
            clean_title = clean_filename(title)
        
        # 檢查是否為 HLS 串流
        is_hls = '.m3u8' in youtube_url.lower()
        
        if is_hls:
            # HLS 串流使用不同的設定
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(audio_path, f'{clean_title}.%(ext)s'),
                'noplaylist': True,
                'ignoreerrors': True,
                'no_warnings': True,
            }
        else:
            # 一般影片使用音訊提取
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(audio_path, f'{clean_title}.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'mp3',
                'noplaylist': True,
                'ignoreerrors': True,
                'no_warnings': True,
            }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"正在下載音訊: {youtube_url}")
            print(f"原始標題: {title}")
            print(f"清理後標題: {clean_title}")
            ydl.download([youtube_url])
            
            # 尋找下載的檔案 - 支援多種可能的檔案名稱
            downloaded_files = []
            import time
            current_time = time.time()
            
            for file in os.listdir(audio_path):
                file_path = os.path.join(audio_path, file)
                # 檢查是否為音訊/影片檔案
                if file.endswith(('.mp3', '.m4a', '.webm', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.ts')):
                    # 檢查檔案修改時間（最近下載的）
                    file_time = os.path.getmtime(file_path)
                    # 如果檔案是在最近5分鐘內創建的，認為是剛下載的
                    if current_time - file_time < 300:  # 5分鐘
                        downloaded_files.append((file_path, file_time, file))
            
            if downloaded_files:
                # 選擇最新的檔案
                latest_file = max(downloaded_files, key=lambda x: x[1])
                file_path, _, file_name = latest_file
                print(f"音訊下載完成: {file_name}")
                return file_path, clean_title
            
            # 如果找不到最近下載的檔案，嘗試尋找任何音訊/影片檔案
            print("未找到最近下載的檔案，搜尋所有音訊/影片檔案...")
            all_media_files = []
            for file in os.listdir(audio_path):
                if file.endswith(('.mp3', '.m4a', '.webm', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.ts')):
                    file_path = os.path.join(audio_path, file)
                    file_time = os.path.getmtime(file_path)
                    all_media_files.append((file_path, file_time, file))
            
            if all_media_files:
                # 選擇最新的檔案
                latest_file = max(all_media_files, key=lambda x: x[1])
                file_path, _, file_name = latest_file
                print(f"找到媒體檔案: {file_name}")
                return file_path, clean_title
            
            raise Exception("找不到下載的音訊檔案")
        
    except Exception as e:
        print(f"下載音訊時發生錯誤: {e}")
        return None, None

# === 處理不同類型的影片來源 ===
def process_video_source(source):
    """處理不同類型的影片來源"""
    source = source.strip()
    
    # 檢查是否為 blob URL
    if source.startswith('blob:'):
        print("\n⚠️  檢測到 Blob URL")
        print("Blob URL 無法直接下載，因為它是瀏覽器內部的臨時引用。")
        print("\n請嘗試以下方法：")
        print("1. 在瀏覽器中按 F12 打開開發者工具")
        print("2. 切換到 Network 標籤")
        print("3. 重新載入頁面或播放影片")
        print("4. 尋找實際的影片文件 URL（.mp4, .m3u8, .ts 等）")
        print("5. 複製該 URL 並重新輸入")
        return None, None, "blob_url"
    
    # 檢查是否為本地文件
    elif os.path.isfile(source):
        print(f"\n📁 檢測到本地文件: {source}")
        return source, os.path.splitext(os.path.basename(source))[0], "local_file"
    
    # 檢查是否為 YouTube URL
    elif 'youtube.com' in source or 'youtu.be' in source:
        return source, None, "youtube"
    
    # 檢查是否為其他線上影片 URL
    elif source.startswith(('http://', 'https://')):
        print(f"\n🌐 檢測到線上影片 URL: {source}")
        print("嘗試使用 yt-dlp 下載...")
        return source, None, "online_video"
    
    else:
        print(f"\n❌ 無法識別的來源格式: {source}")
        return None, None, "unknown"

# === 為特定路徑的逐字稿加上標點符號 ===
def add_punctuation_to_file(file_path):
    """為指定路徑的逐字稿文件加上標點符號"""
    try:
        # 檢查文件是否存在
        if not os.path.isfile(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 讀取文件內容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print("❌ 文件為空")
            return False
        
        print(f"📄 讀取文件: {file_path}")
        print(f"📊 文件大小: {len(content)} 字元")
        
        # 檢查是否已經有標點符號
        has_punctuation = any(p in content for p in '。，！？；：')
        if has_punctuation:
            print("⚠️  文件似乎已經包含標點符號")
            overwrite = input("是否要重新處理？(y/n): ").strip().lower()
            if overwrite != 'y':
                print("❌ 取消處理")
                return False
        
        # 計算預估費用
        text_length = len(content)
        estimated_tokens = text_length * 1.3  # 粗略估算：中文字符約1.3個token
        estimated_cost = (estimated_tokens / 1000) * 0.00015  # GPT-4o-mini 價格：$0.00015/1K tokens
        
        print(f"\n💰 費用預估:")
        print(f"   文字長度: {text_length} 字元")
        print(f"   預估 tokens: {estimated_tokens:.0f}")
        print(f"   預估費用: 約 ${estimated_cost:.4f} 美元 (約 {estimated_cost * 30:.2f} 台幣)")
        
        # 確認是否繼續處理
        print(f"\n是否要繼續進行標點符號處理？")
        print("1. 是，繼續處理")
        print("2. 否，取消處理")
        
        confirm = input("請選擇 (1-2): ").strip()
        if confirm != "1":
            print("❌ 取消處理")
            return False
        
        # 取得 API Key
        api_key = get_api_key()
        if not api_key:
            print("❌ 無法取得 API Key，無法進行標點符號處理")
            return False
        
        # 備份原文件
        backup_path = file_path + ".backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 已備份原文件至: {backup_path}")
        
        # 處理標點符號
        client = openai.OpenAI(api_key=api_key)
        punctuated_text = process_text_with_punctuation(content, client)
        
        # 儲存處理後的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(punctuated_text)
        
        print(f"✅ 標點符號處理完成！")
        print(f"📄 已更新文件: {file_path}")
        print(f"📊 處理後長度: {len(punctuated_text)} 字元")
        print(f"💾 原文件備份: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 處理文件時發生錯誤: {e}")
        return False

def save_file(used_method, output_filename, text):
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"\n✅ 標點符號處理完成！")
    print(f"使用方法: {used_method} + GPT標點符號處理")
    print(f"輸出檔案: {output_filename}")
    print(f"處理後文字長度: {len(text)} 字元")

# === 主程式 ===
if __name__ == "__main__":
    print("🎵 智能語音轉文字工具")
    print("=" * 50)
    
    # 功能選擇
    print("請選擇功能:")
    print("1. 轉錄影片 (支援 YouTube、本地文件、線上影片)")
    print("2. 為特定的逐字稿加上標點符號")
    print("3. 清空目前已存的 AI 模型")
    print("4. 退出程式")
    choice = input("請選擇 (1-4): ").strip()
    
    if choice == "2":
        # 為特定路徑的逐字稿加上標點符號
        print("\n📝 為逐字稿加上標點符號功能")
        print("=" * 50)
        
        # 取得文件路徑
        print("請輸入要處理的逐字稿文件路徑:")
        print("支援格式: .txt 文件")
        print("範例: /path/to/transcript.txt 或 output/逐字稿.txt")
        
        file_path = input("文件路徑: ").strip()
        if not file_path:
            print("❌ 未提供文件路徑")
            exit(1)
        
        # 處理文件
        success = add_punctuation_to_file(file_path)
        if success:
            print("\n✅ 處理完成！")
        else:
            print("\n❌ 處理失敗")
        
        exit(0)
        
    elif choice == "3":
        # 清空預存模型
        print("\n🗑️  清空預存 AI 模型功能")
        print("=" * 50)
        clear_whisper_models()
        exit(0)
    elif choice == "4":
        print("程式結束")
        exit(0)
    elif choice != "1":
        print("無效選擇，程式結束")
        exit(1)
    
    # 取得影片來源
    print("\n請輸入影片來源:")
    print("支援格式:")
    print("- YouTube URL (如: https://www.youtube.com/watch?v=...)")
    print("- 本地文件路徑 (如: /path/to/video.mp4)")
    print("- 線上影片 URL (如: https://example.com/video.mp4)")
    print("- 注意: Blob URL 無法直接處理，需要找到實際的影片 URL")
    
    video_source = input("影片來源: ").strip()
    if not video_source:
        print("未提供來源，程式結束")
        exit(1)
    
    # 處理不同類型的影片來源
    processed_source, video_title, source_type = process_video_source(video_source)
    
    if source_type == "blob_url":
        print("\n請按照上述說明找到實際的影片 URL 後重新執行程式")
        exit(1)
    elif source_type == "unknown":
        print("無法處理此來源格式")
        exit(1)
    
    # 根據來源類型處理影片資訊
    video_info = None
    audio_file = None
    
    if source_type == "local_file":
        # 本地文件處理
        print(f"\n📁 處理本地文件: {processed_source}")
        audio_file, file_title = process_local_audio(processed_source)
        if not audio_file:
            print("❌ 無法處理本地文件")
            exit(1)
        video_title = file_title or "local_video"
        video_info = {'title': video_title, 'duration': 0, 'uploader': 'local', 'url': processed_source}
        
    elif source_type in ["youtube", "online_video"]:
        # 線上影片處理
        print("\n📋 正在取得影片資訊...")
        video_info = get_video_info(processed_source)
        
        if not video_info:
            print("❌ 無法取得影片資訊，請檢查網址是否正確")
            exit(1)
    
    # 顯示影片資訊
    if video_info:
        print("\n" + "="*60)
        print("📺 影片資訊確認")
        print("="*60)
        print(f"標題: {video_info['title']}")
        print(f"上傳者: {video_info['uploader']}")
        if video_info['duration'] > 0:
            duration_minutes = video_info['duration'] / 60
            print(f"長度: {duration_minutes:.1f} 分鐘")
        print(f"來源: {video_info['url']}")
        print("="*60)
    
    # 讓使用者確認
    print("\n請確認是否要轉錄此影片？")
    print("1. 是，繼續轉錄")
    print("2. 否，重新輸入網址")
    print("3. 退出程式")
    
    confirm = input("請選擇 (1-3): ").strip()
    
    if confirm == "2":
        # 重新輸入來源
        video_source = input("\n請重新輸入影片來源: ").strip()
        if not video_source:
            print("未提供來源，程式結束")
            exit(1)
        
        # 重新處理影片來源
        processed_source, video_title, source_type = process_video_source(video_source)
        
        if source_type == "blob_url":
            print("\n請按照上述說明找到實際的影片 URL 後重新執行程式")
            exit(1)
        elif source_type == "unknown":
            print("無法處理此來源格式")
            exit(1)
        
        # 重新取得影片資訊
        if source_type == "local_file":
            audio_file, file_title = process_local_audio(processed_source)
            if not audio_file:
                print("❌ 無法處理本地文件")
                exit(1)
            video_title = file_title or "local_video"
            video_info = {'title': video_title, 'duration': 0, 'uploader': 'local', 'url': processed_source}
        else:
            print("\n📋 正在取得影片資訊...")
            video_info = get_video_info(processed_source)
            
            if not video_info:
                print("❌ 無法取得影片資訊，請檢查網址是否正確")
                exit(1)
        
        # 再次顯示影片資訊
        print("\n" + "="*60)
        print("📺 影片資訊確認")
        print("="*60)
        print(f"標題: {video_info['title']}")
        print(f"上傳者: {video_info['uploader']}")
        if video_info['duration'] > 0:
            duration_minutes = video_info['duration'] / 60
            print(f"長度: {duration_minutes:.1f} 分鐘")
        print(f"來源: {video_info['url']}")
        print("="*60)
        
        # 再次確認
        confirm = input("\n確認轉錄此影片？(y/n): ").strip().lower()
        if confirm != 'y':
            print("程式結束")
            exit(1)
            
    elif confirm == "3":
        print("程式結束")
        exit(1)
    elif confirm != "1":
        print("無效選擇，程式結束")
        exit(1)
    
    # 選擇轉錄語言
    print("\n請選擇轉錄語言:")
    print("1. 中文")
    print("2. 英文")
    print("3. 多語言混雜")
    language = input("請選擇 (1-3): ").strip()
    
    if language == '1':
        language = 'zh'
    elif language == '2':
        language = 'en'
    else:
        language = None
    
    # 選擇轉錄方式
    print("\n請選擇轉錄方式:")
    print("1. 智能模式 (由大至小測試本地可用的模型)")
    print("2. 本地模式 (僅使用本地模型)")
    if video_info['duration'] > 0:
        print(f"3. API模式 (僅使用OpenAI API，預估處理價格：{round(0.2*video_info['duration']/60, 2)}元)")
    else:
        print("3. API模式 (僅使用OpenAI API，預估處理價格：未知)")
    
    mode = input("請選擇 (1-3): ").strip()
    
    # 處理音訊文件
    if source_type != "local_file":
        # 下載音訊
        print("\n開始下載音訊...")
        max_retries = 3
        
        for attempt in range(max_retries):
            print(f"嘗試第 {attempt + 1} 次下載...")
            result = download_audio(processed_source)
            if result and result[0]:
                audio_file, video_title = result
                break
            if attempt < max_retries - 1:
                print("等待 3 秒後重試...")
                time.sleep(3)
        
        if not audio_file or not video_title:
            print("❌ 下載失敗，請檢查網址是否正確")
            exit(1)
        
        print(f"✅ 音訊下載成功: {video_title}")
    else:
        # 本地文件，直接使用
        video_title = video_info['title']
        print(f"✅ 使用本地文件: {video_title}")
    
    # 根據選擇的模式進行轉錄
    transcript = None
    used_method = ""
    
    if mode == "1":  # 智能模式
        print("\n🤖 智能模式啟動")
        # 先嘗試本地
        transcript, model_used = smart_model_selection(audio_file, language)
        if transcript:
            used_method = f"本地模型 ({model_used})"
        else:
            print("\n本地轉錄失敗，嘗試使用 API...")
            api_key = get_api_key()
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                transcript_text = transcribe_api(audio_file, client, language)
                if transcript_text:
                    transcript = {"text": transcript_text}
                    used_method = "OpenAI API"
                else:
                    print("❌ API 轉錄也失敗了")
            else:
                print("❌ 無法取得 API Key，轉錄失敗")
    
    elif mode == "2":  # 本地模式
        print("\n💻 本地模式")
        model_size = input("請選擇模型大小 (large-v3/large-v2/large/medium/small/base/tiny), 越大品質越好 但也需要更多的運行資源及時間, 預設\"base\": ").strip() or "base"
        transcript = transcribe_local(audio_file, model_size, language)
        if transcript:
            used_method = f"本地模型 ({model_size})"
    
    elif mode == "3":  # API模式
        print("\n🌐 API模式")
        api_key = get_api_key()
        if api_key:
            print("正在使用 API 轉錄...")
            client = openai.OpenAI(api_key=api_key)
            transcript_text = transcribe_api(audio_file, client, language)
            if transcript_text:
                transcript = {"text": transcript_text}
                used_method = "OpenAI API"
        else:
            print("❌ 無法取得 API Key")
    
    # 儲存結果
    if transcript:
        output_path = "output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_filename = os.path.join(output_path, f"{video_title}.txt")
        traditional_text = zhconv.convert(transcript["text"], 'zh-hant')
        punctuation_choice = ''
        if mode != "3": # not using OpenIA API
            # 詢問是否要添加標點符號
            print(f"\n📝 轉錄完成！原始文字長度: {len(traditional_text)} 字元")
            print("\n是否要使用 GPT 為文字添加標點符號？")
            # 計算標點符號處理的預估成本
            text_length = len(traditional_text)
            estimated_tokens = text_length * 1.3  # 粗略估算：中文字符約1.3個token
            estimated_cost = (estimated_tokens / 1000) * 0.00015  # GPT-4o-mini 價格：$0.00015/1K tokens
            
            print("1. 是，添加標點符號 (需要 OpenAI API Key)")
            print(f"   預估成本: 約 ${estimated_cost:.4f} 美元 (約 {estimated_cost * 30:.2f} 台幣)")
            print(f"   文字長度: {text_length} 字元，預估 {estimated_tokens:.0f} tokens")
            print("2. 否，直接儲存原始文字")
        
            punctuation_choice = input("請選擇 (1-2): ").strip()
        
            if punctuation_choice == "1":
                # 需要 API Key 進行標點符號處理
                api_key = get_api_key()
                if api_key:
                    
                    with open(os.path.join(output_path, f"{video_title}_unpunctuated.txt"), "w", encoding="utf-8") as f:
                        f.write(traditional_text)
                    
                    client = openai.OpenAI(api_key=api_key)
                    punctuated_text = process_text_with_punctuation(traditional_text, client)
                    
                    # 儲存帶標點符號的版本
                    save_file(used_method, output_filename, punctuated_text)
                else:
                    print("❌ 無法取得 API Key，儲存原始文字")
                    save_file(used_method, os.path.join(output_path, f"{video_title}_unpunctuated.txt"), traditional_text)
        else:
            save_file(used_method, output_filename, traditional_text)
    else:
        print("\n❌ 轉錄失敗")
