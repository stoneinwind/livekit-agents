import base64
import requests
#import cv2 # 需要安装opencv-python包，用于图像处理
from PIL import Image # 自带模块，简单处理即可
import io
import json
import asyncio

class MoondreamClient:
    def __init__(self, model="moondream:1.8b"):
        self.url = "http://localhost:11434/api/chat"
        self.model = model

    def _encode_image_cv(self, frame):
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode()    

    def _encode_image_pil(self, frame):
        # frame 是 numpy 数组 (来自ScreenCapture)
        # 转换色彩通道: mss 抓取的是 BGRA，PIL 需要 RGB
        # [..., :3] 取前三个通道，[::-1] 将 BGR 转为 RGB
        rgb_frame = frame[:, :, :3][:, :, ::-1]
        img = Image.fromarray(rgb_frame)
        print(f"DEBUG image size: {img.size}, mode: {img.mode}")  # 确认图像正常
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)  # 明确指定quality
        # Debug: 保存到本地
        img.save("debug_screenshot.jpg", format="JPEG", quality=85)
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8").replace('\n','')        # with open("debug_screenshot.jpg", "rb") as f:
        #     encoded = base64.b64encode(f.read()).decode("utf-8")
        print(f"DEBUG base64 length: {len(encoded)}")  # 太短说明图像有问题
        with open("debug_b64.txt", "w") as f:
            f.write(encoded)
        return encoded

    async def describe_async(self, frame):
        """Describe the given frame asynchronously.
        Args:
            frame: The frame to describe.
        Returns:
            A string containing the description of the frame.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.describe, frame)

    def describe(self, frame):
        try:
            image_b64 = self._encode_image_pil(frame)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe the image concisely and correctly. Skip the letters or numbers or logos on the image.",
                        "images": [image_b64],
                    }
                ],
                "stream": False
            }

            r = requests.post(self.url, 
                headers={"Content-Type": "application/json"}, 
                #data=json.dumps(payload),
                json=payload,
                timeout=10)
            return r.json()["message"]["content"]
        except Exception as e:
            return "Error analyzing the image"
        