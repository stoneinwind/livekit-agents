import numpy as np
import asyncio
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit import rtc
from piper import PiperVoice, SynthesisConfig
import unicodedata, re
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """更彻底的文本清洗，防止 Piper 崩溃或乱码"""
    # 1. 统一 NFKC 规范化（全角→半角，兼容字符分解）
    text = unicodedata.normalize("NFKC", text)    
    # 2. 替换常见智能引号、破折号
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")   # en dash
            .replace("—", "-")   # em dash
            .replace("―", "-")
    )    
    # 3. 移除控制字符（除了 \n \t \r）
    text = "".join(c for c in text if c == "\n" or c == "\t" or c == "\r" or unicodedata.category(c)[0] != "C")    
    # 4. 替换或移除 emoji / 非常见符号（Piper 通常不支持或表现差）
    #    这里保守策略：替换成空格，你也可以换成 "" 或其他提示文本
    text = re.sub(r'[^\w\s.,!?;:\-\'"()\[\]@#$%^&*+=<>/\\|~`]', ' ', text)    
    # 5. 连续空白压缩 + 首尾去空
    text = re.sub(r'\s+', ' ', text).strip()    
    return text if text else " "  # 防止空文本导致崩溃

# 使用python库piper-tts（而非standalone exec）
class PiperTTSPlugin(tts.TTS):
    def __init__(self, model, speed=1.0, volume=1.0, noise_scale=0.667, noise_w=0.8, use_cuda=False):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=22050, num_channels=1)
        self.model_path = model
        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.use_cuda = use_cuda
        self._voice = None
        self._load_voice()

    def _load_voice(self):
        # according to the docs if you enable cuda you need onnxruntime-gpu package, read the docs
        # if no GPU onnx version: onnxruntime\capi\onnxruntime_inference_collection.py:
        #   Specified provider 'CUDAExecutionProvider' is not in available provider names.
        #   Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
        # with onnx GPU runtime installed,:
        #   ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self._voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
        
    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        cleaned_input = clean_text(text)
        if not cleaned_input.strip():
            raise ValueError("Input text is empty after cleaning")
        return PiperApiStream(self, cleaned_input, conn_options)

class PiperApiStream(tts.ChunkedStream):
    def __init__(self, plugin, text, conn_options):
        super().__init__(tts=plugin, input_text=text, conn_options=conn_options)
        self.plugin = plugin

    async def _run(self, output_emitter:tts.AudioEmitter):
        try:     
            req_id = f"pipertts-{id(self)}"
            output_emitter.initialize(                
                request_id=req_id,
                sample_rate=self.plugin._sample_rate,
                num_channels=1,
                mime_type="audio/pcm"                     # or "audio/wav" — pcm is safer here
                # seg_id=seg_id # 如果是非streaming模式，默认会置空segment_id（参initialize方法）再启动start_segment
            )

            config = SynthesisConfig(
                volume=self.plugin.volume,
                length_scale=self.plugin.speed,
                noise_scale=self.plugin.noise_scale,
                noise_w_scale=self.plugin.noise_w,
                normalize_audio=True # True will depress abnormal volumes
            )            
            loop = asyncio.get_running_loop() # instead of event loop

            import queue
            q = queue.Queue(maxsize=10) # backpressure - if maxsize reached, block producer
            STOP = object()
            def producer():
                try:
                    for chunk in self.plugin._voice.synthesize(
                        self.input_text,
                        syn_config=config
                        ):
                        q.put(chunk.audio_int16_bytes)
                except Exception as e:
                    logger.error(f"Piper synthesize failed: {e}")
                    q.put(e)  # 把异常也作为数据传给消费者
                finally:
                    q.put(STOP)
            loop.run_in_executor(None, producer)
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is STOP:
                    break
                if isinstance(item, Exception):
                    # 这里可以替换成 LiveKit metrics 上报
                    logger.error(f"Piper synthesize error reported: {item}")
                    raise item  # 关键异常向上抛，让上层感知失败
                # 输出音频数据chunked stream
                output_emitter.push(item)            
        except Exception as e:
            print(f"Piper execeptions: {e}")
            _silence_data = np.zeros(22050, dtype=np.int16).tobytes()
            output_emitter.push(_silence_data)
