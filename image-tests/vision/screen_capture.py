import mss # An ultra fast cross-platform multiple screenshots module in pure python using ctypes.
import numpy as np
import hashlib
import time
import asyncio
import logging
import pygetwindow as gw
import mss
from PIL import Image

logger = logging.getLogger(__name__) # module's logger
class ScreenCapture:
    """ 屏幕捕获类
    后台按照interval截屏并计算是否与上一帧的差异达到阈值，如果达到则返回当前帧 """
    def __init__(self, interval=5.0, diff_threshold=0.1):
        self.interval = interval
        self.prev_hash = None
        self.prev_frame = None
        self.diff_threshold = diff_threshold

    def _hash_frame(self, frame: np.ndarray):
        ''' 简化版图片数据差分判定 diff（hash），够用且快
        问题是过于严苛，一点点的细微差异都会造成diff判定 '''
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _calc_diff_whole(self, frame: np.ndarray) -> float:
        """
        计算当前帧与上一帧的平均像素差异比例。
        返回值为 0.0 (完全相同) 到 1.0 (完全不同) 之间的浮点数。
        """
        if self.prev_frame is None or self.prev_frame.shape != frame.shape:
            return 1.0  # 第一帧默认差异为 100%；如果分辨率变了或窗口尺寸变了，也视为完全不同（否则numpy运算报错）            
        # 1. 计算绝对差值 (Absolute Difference)
        # 注意：使用 np.abs 前先转为 float 或 int，防止 uint8 溢出导致 0-1=255
        # diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))     
        # 如果分辨率过高，图片数据过大，可以考虑采样：每隔 4 个像素取一个点进行对比，速度提升显著且不影响精度判断
        diff = np.abs(frame[::4, ::4].astype(np.int16) - self.prev_frame[::4, ::4].astype(np.int16))   
        # 2. 计算平均差异分值
        # 255 是 8位图像（RGB/BGR）的最大像素值
        score = np.mean(diff) / 255.0
        return score

    def _calc_diff_ratio(self, frame: np.ndarray) -> float:
        """
        计算当前帧与上一帧的平均像素差异比例。
        返回值为 0.0 (完全相同) 到 1.0 (完全不同) 之间的浮点数。
        """
        if self.prev_frame is None or self.prev_frame.shape != frame.shape:
            return 1.0  # 第一帧默认差异为 100%；如果分辨率变了或窗口尺寸变了，也视为完全不同（否则numpy运算报错）            
        diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))    
        # 每个像素取所有通道的最大差值（任意通道变化就算）
        diff_per_pixel = diff.max(axis=2)  # shape: (H, W)        
        # 超过阈值的像素才算"真正变化了"
        changed = np.sum(diff_per_pixel > 15)  # 15 可调，过滤噪点
        total = diff_per_pixel.size        
        score = changed / total
        return score

    def capture_loop(self):
        with mss.ss() as sct: # ✅ 整个 loop 在同一线程，创建一次即可
            while True:
                monitor = sct.monitors[1] # 0是所有屏幕，1是默认的第一块屏幕，以此类推
                # grab的结果是个screenshot对象，包含left, top, width, height, image属性（默认访问raw，即BGRA原始像素数组；rgb则是去掉Alpha通道后的字节数组）
                img = np.array(sct.grab(monitor)) # 截取屏幕，得到的是纯numpy数组
                h = self._hash_frame(img)
                # 改进：更精细的差异判定 -- 调用差异对比逻辑
                # diff_score = self._calc_diff(img)
                # if diff_score >= self.diff_threshold:
                #     self.prev_frame = img
                #     yield img

                if self.prev_hash != h:
                    self.prev_hash = h
                    yield img
                time.sleep(self.interval)
    
    def capture_once(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1] # 0是所有屏幕，1是默认的第一块屏幕，以此类推
            img = np.array(sct.grab(monitor)) # 截取屏幕，得到的是纯numpy数组
        return img

    async def capture_once_async(self):
        """capture screen once asynchronously.
        Returns:
            A capture frame.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture_once)

    @staticmethod
    def save_frame_img(frame, filename="test-screenshot.jpg", quality=85):        # frame 是 numpy 数组 
        ''' 工具方法，保存帧为图片文件
        args:
            frame: numpy 数组 (来自capture方法)
            filename: 保存的文件名，默认为 "test-screenshot.jpg
            quality: 保存图片的质量，默认为 85
        '''
        # 转换色彩通道: mss 抓取的是 BGRA，PIL 需要 RGB
        # [..., :3] 取前三个通道，[::-1] 将 BGR 转为 RGB
        rgb_frame = frame[:, :, :3][:, :, ::-1]
        img = Image.fromarray(rgb_frame)
        logger.debug(f"DEBUG image size: {img.size}, mode: {img.mode}")  # 确认图像正常
        img.save(filename, format="JPEG", quality=quality)

class WindowCapture(ScreenCapture):
    """ 窗口捕获类
    捕获指定窗口的屏幕 """
    def __init__(self, window_name, interval=2.0, calc_diff_whole=True):
        super().__init__(interval)
        self.window_name = window_name
        self._cached_rect = None      # 缓存窗口位置
        self.calc_diff_whole = calc_diff_whole # 是否用全图差异判定（否则就是比率判定）

    def _get_window_rect(self, force_refresh=False) -> dict | None:
        # 使用缓存，除非强制刷新
        if self._cached_rect and not force_refresh:
            return self._cached_rect

        if not self.window_name:
            return None

        wins = gw.getWindowsWithTitle(self.window_name)
        if not wins:
            raise ValueError(f"未找到窗口: {self.window_name}")

        w = wins[0]
        if w.width <= 0 or w.height <= 0:
            logger.warning("窗口已最小化或不可见")
            return None
        if w.left < 0 or w.top < 0:
            logger.warning("窗口位置可能在屏幕外，请调整窗口位置")
            return None

        self._cached_rect = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
        return self._cached_rect

    def capture_once(self) -> np.ndarray | None:
        rect = self._get_window_rect(force_refresh=True)  # capture_once 总是刷新位置
        if rect is None:
            return None
        with mss.mss() as sct:
            return np.array(sct.grab(rect))

    def capture_loop(self):
        prev_hash = None
        refresh_counter = 0

        with mss.mss() as sct:
            while True:
                try:
                    # 每 10次循环刷新一次窗口位置（应对窗口被移动的情况）
                    refresh_counter += 1
                    #rect = self._get_window_rect(force_refresh=(refresh_counter % 10 == 0))
                    rect = self._get_window_rect(force_refresh=True) # 强制刷新窗口位置
                    if rect is None:
                        logger.warning("窗口位置可能在屏幕外，请调整窗口位置")
                        time.sleep(self.interval)
                        continue
                    img = np.array(sct.grab(rect)) # 截取屏幕，得到的是纯numpy数组
                    # h = self._hash_frame(img) # 稍微变动一点点就认为更新了
                    # if self.prev_hash != h:
                    #     self.prev_hash = h
                    #     yield img
                    # 改进：更精细的差异判定 -- 调用差异对比逻辑
                    diff_score = self._calc_diff_whole(img) if self.calc_diff_whole else self._calc_diff_ratio(img)
                    if diff_score >= self.diff_threshold:
                        self.prev_frame = img
                        yield img
                    else:
                        logger.debug(f"窗口无变化或变化不够明显，不返回帧: {diff_score}")
                except ValueError as e:
                    # 窗口消失了（用户关闭）
                    logger.warning(f"窗口丢失，停止捕获: {e}")
                    continue
                except Exception as e:
                    logger.error(f"截图出错: {e}")

                time.sleep(self.interval)