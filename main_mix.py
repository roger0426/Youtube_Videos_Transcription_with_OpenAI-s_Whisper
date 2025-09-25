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
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        print(f"系統記憶體: {available_memory:.1f} GB 可用")
        if gpu_memory > 0:
            print(f"GPU 記憶體: {gpu_memory:.1f} GB")
        
        # 根據資源建議模型
        if gpu_memory >= 8:
            return "large"
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
            save_choice = input("是否要儲存 API Key 到 .env 檔案以便下次使用？(y/n): ").strip().lower()
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
        model = whisper.load_model(model_size)
        print(f"開始轉錄 (語言: {language})...")
        result = model.transcribe(audio_path, language=language, verbose=True, word_timestamps=True)
        return result
    except Exception as e:
        print(f"本地轉錄時發生錯誤: {e}")
        return None

# === API Whisper 轉錄 ===
def transcribe_api(audio_file_path, client=None):
    """使用 OpenAI Whisper API 轉錄"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            if client is None:
                client = openai.OpenAI()
            response = client.audio.transcriptions.create(
                model="whisper-1",
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "你是一個專業的文字編輯助手。請為以下沒有標點符號的文字添加適當的標點符號，包括句號、逗號、問號、驚嘆號等。保持原文的語意和結構，只添加標點符號。"
                },
                {
                    "role": "user", 
                    "content": f"請為以下文字添加標點符號：\n\n{text_chunk}"
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 標點符號處理時發生錯誤: {e}")
        return text_chunk  # 如果失敗，返回原文

# === 批量標點符號處理 ===
def process_text_with_punctuation(text, client, chunk_size=100):
    """將文字分割成片段，批量處理標點符號"""
    print(f"\n📝 開始標點符號處理...")
    print(f"原始文字長度: {len(text)} 字元")
    
    # 按空格分割文字
    words = text.split(" ")
    if len(text) > 100 and len(words) < 5: # 文字未被分段
        chunk_size = 2000
    
    print(f"分割成 {len(words)} 個句子")
    
    # 計算需要處理的片段數量
    total_chunks = (len(words) + chunk_size - 1) // chunk_size
    print(f"將分成 {total_chunks} 個片段處理，每片段 {chunk_size} 個句子")
    
    processed_chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        print(f"處理片段 {chunk_num}/{total_chunks} ({len(chunk_words)} 個句子)...")
        
        # 使用 GPT 添加標點符號
        punctuated_chunk = add_punctuation_with_gpt(chunk_text, client)
        processed_chunks.append(punctuated_chunk)
        
        # 添加小延遲避免 API 限制
        time.sleep(0.5)
    
    # 合併所有處理過的片段
    final_text = " ".join(processed_chunks)
    print(f"✅ 標點符號處理完成！")
    print(f"處理後文字長度: {len(final_text)} 字元")
    
    return final_text

# === 智能模型選擇 ===
def smart_model_selection(audio_path):
    """智能選擇最佳模型"""
    print("\n🤖 智能模型選擇模式")
    
    # 檢查音訊長度
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_minutes = len(audio) / (1000 * 60)
        print(f"音訊長度: {duration_minutes:.1f} 分鐘")
    except:
        duration_minutes = 0
    
    # 建議的模型順序（從大到小）
    model_order = ["large", "medium"] # "small", "base", "tiny"
    recommended = check_system_resources()
    
    print(f"系統建議模型: {recommended}")
    print("將從最大模型開始嘗試，如果失敗會自動降級...")
    
    for model_size in model_order:
        print(f"\n嘗試使用 {model_size} 模型...")
        try:
            result = transcribe_local(audio_path, model_size)
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
        
        # 使用清理後的標題作為檔案名稱
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
            
            # 尋找下載的檔案
            for file in os.listdir(audio_path):
                if file.startswith(clean_title) and file.endswith(('.mp3', '.m4a', '.webm', '.ogg')):
                    file_path = os.path.join(audio_path, file)
                    print(f"音訊下載完成: {file}")
                    return file_path, clean_title
            
            raise Exception("找不到下載的音訊檔案")
        
    except Exception as e:
        print(f"下載音訊時發生錯誤: {e}")
        return None, None

# === 主程式 ===
if __name__ == "__main__":
    print("🎵 智能語音轉文字工具")
    print("=" * 50)
    
    # 取得 YouTube 網址
    youtube_url = input("要轉錄的YouTube影片網址: ").strip()
    if not youtube_url:
        print("未提供網址，程式結束")
        exit(1)
    
    # 先取得影片資訊讓使用者確認
    print("\n📋 正在取得影片資訊...")
    video_info = get_video_info(youtube_url)
    
    if not video_info:
        print("❌ 無法取得影片資訊，請檢查網址是否正確")
        exit(1)
    
    # 顯示影片資訊
    print("\n" + "="*60)
    print("📺 影片資訊確認")
    print("="*60)
    print(f"標題: {video_info['title']}")
    print(f"上傳者: {video_info['uploader']}")
    if video_info['duration'] > 0:
        duration_minutes = video_info['duration'] / 60
        print(f"長度: {duration_minutes:.1f} 分鐘")
    print(f"網址: {video_info['url']}")
    print("="*60)
    
    # 讓使用者確認
    print("\n請確認是否要轉錄此影片？")
    print("1. 是，繼續轉錄")
    print("2. 否，重新輸入網址")
    print("3. 退出程式")
    
    confirm = input("請選擇 (1-3): ").strip()
    
    if confirm == "2":
        # 重新輸入網址
        youtube_url = input("\n請重新輸入YouTube影片網址: ").strip()
        if not youtube_url:
            print("未提供網址，程式結束")
            exit(1)
        
        # 重新取得影片資訊
        print("\n📋 正在取得影片資訊...")
        video_info = get_video_info(youtube_url)
        
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
        print(f"網址: {video_info['url']}")
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
    
    # 選擇轉錄方式
    print("\n請選擇轉錄方式:")
    print("1. 智能模式 (優先本地，失敗時使用API)")
    print("2. 本地模式 (僅使用本地模型)")
    print(f"3. API模式 (僅使用OpenAI API，預估處理價格：{round(0.2*duration_minutes, 2)}元)")
    
    mode = input("請選擇 (1-3): ").strip()
    
    # 下載音訊
    print("\n開始下載音訊...")
    max_retries = 3
    audio_file = None
    video_title = None
    
    for attempt in range(max_retries):
        print(f"嘗試第 {attempt + 1} 次下載...")
        result = download_audio(youtube_url)
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
    
    # 根據選擇的模式進行轉錄
    transcript = None
    used_method = ""
    
    if mode == "1":  # 智能模式
        print("\n🤖 智能模式啟動")
        # 先嘗試本地
        transcript, model_used = smart_model_selection(audio_file)
        if transcript:
            used_method = f"本地模型 ({model_used})"
        else:
            print("\n本地轉錄失敗，嘗試使用 API...")
            api_key = get_api_key()
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                transcript_text = transcribe_api(audio_file, client)
                if transcript_text:
                    transcript = {"text": transcript_text}
                    used_method = "OpenAI API"
                else:
                    print("❌ API 轉錄也失敗了")
            else:
                print("❌ 無法取得 API Key，轉錄失敗")
    
    elif mode == "2":  # 本地模式
        print("\n💻 本地模式")
        model_size = input("請選擇模型大小 (large/medium/small/base/tiny), 越大品質越好: ").strip() or "base"
        transcript = transcribe_local(audio_file, model_size)
        if transcript:
            used_method = f"本地模型 ({model_size})"
    
    elif mode == "3":  # API模式
        print("\n🌐 API模式")
        api_key = get_api_key()
        if api_key:
            print("正在使用 API 轉錄...")
            client = openai.OpenAI(api_key=api_key)
            transcript_text = transcribe_api(audio_file, client)
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
        # 詢問是否要添加標點符號
        print(f"\n📝 轉錄完成！原始文字長度: {len(traditional_text)} 字元")
        print("\n是否要使用 GPT 為文字添加標點符號？")
        print("1. 是，添加標點符號 (需要 OpenAI API Key)")
        print("2. 否，直接儲存原始文字")
        
        punctuation_choice = input("請選擇 (1-2): ").strip()
        
        if punctuation_choice == "1":
            # 需要 API Key 進行標點符號處理
            api_key = get_api_key()
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                punctuated_text = process_text_with_punctuation(traditional_text, client)
                
                # 儲存帶標點符號的版本
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(punctuated_text)
                
                print(f"\n✅ 標點符號處理完成！")
                print(f"使用方法: {used_method} + GPT標點符號處理")
                print(f"輸出檔案: {output_filename}")
                print(f"處理後文字長度: {len(punctuated_text)} 字元")
            else:
                print("❌ 無法取得 API Key，儲存原始文字")
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(traditional_text)
                print(f"✅ 轉錄完成！")
                print(f"使用方法: {used_method}")
                print(f"輸出檔案: {output_filename}")
                print(f"逐字稿長度: {len(traditional_text)} 字元")
        else:
            # 直接儲存原始文字
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(traditional_text)
            
            print(f"\n✅ 轉錄完成！")
            print(f"使用方法: {used_method}")
            print(f"輸出檔案: {output_filename}")
            print(f"逐字稿長度: {len(traditional_text)} 字元")
    else:
        print("\n❌ 轉錄失敗")
