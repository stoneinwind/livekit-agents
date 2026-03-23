from typing import Annotated # 用于类型标注
from vision.moondream_ollama import MoondreamClient
from vision.screen_capture import WindowCapture
import asyncio, time, os
import logging
import time
from dotenv import load_dotenv

load_dotenv(".env")

# Initialize with API key -- moondream only works under WSL
# model = md.vl(api_key=os.getenv("MOONDREAM_API_KEY"))

logging.basicConfig(level=logging.DEBUG)  # 启用默认的logging机制（默认level=WARNING）

if __name__ == "__main__":
    # result = asyncio.run(analyze_screen("Describe what is on the screen"))
    # print(result)
    cap = WindowCapture("照片查看", calc_diff_whole=False)
    img = cap.capture_once()
    if img is not None:
        cap.save_frame_img(img)
        print("cap once OK: Image saved as test-screenshot.jpg")
    else:
        print("cap once NOK: 截图失败，请检查窗口是否存在")

    read_input = input("Press Enter to continue...")

    mc=MoondreamClient()
    try:
        for id, img in enumerate(cap.capture_loop(), start=1):
            if img is not None:
                filename = f"test-screenshot-{id}.jpg"  # ✅ f-string
                cap.save_frame_img(img, filename)
                print(f"cap loop OK: Image saved as {filename}")      # ✅ f-string
                # call Moondream API to analyze the image
                analysis_result = mc.describe(img)  
                print(f"Moondream caption result against {filename}: ", analysis_result) 
                time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")

