from typing import Annotated # 用于类型标注
from vision.moondream_ollama import MoondreamClient
from vision.screen_capture import ScreenCapture, WindowCapture
import asyncio, time

import logging
logging.basicConfig(level=logging.DEBUG)  # 启用默认的logging机制（默认level=WARNING）

async def analyze_screen(
    user_query: Annotated[str, "The specific question about the screen content"] = "Describe what is on the screen"
):
    print(f"DEBUG: Starting screen analysis for: {user_query}")

    try:
        sc=ScreenCapture()
        # 1. 截图 (返回的是raw data）
        screenshot_data = sc.capture_once()
    
        # 2. Moondream 识别
        mc=MoondreamClient()
        analysis_result = mc.describe(screenshot_data)            
        return f"Screen analysis result: {analysis_result}"
    except Exception as e:
        return f"Failed to analyze screen: {str(e)}"


if __name__ == "__main__":
    # result = asyncio.run(analyze_screen("Describe what is on the screen"))
    # print(result)
    cap = WindowCapture("网易", calc_diff_whole=False)
    img = cap.capture_once()
    if img is not None:
        cap.save_frame_img(img)
        print("cap once OK: Image saved as test-screenshot.jpg")
    else:
        print("cap once NOK: 截图失败，请检查窗口是否存在")

    for id, img in enumerate(cap.capture_loop(), start=1):
        if img is not None:
            filename = f"test-screenshot-{id}.jpg"  # ✅ f-string
            cap.save_frame_img(img, filename)
            print(f"cap loop OK: Image saved as {filename}")      # ✅ f-string
