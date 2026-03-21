from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero # 各种rovider也用plugin方式集成
from datetime import datetime
import os
# Load environment variablesload_dotenv(".env")
load_dotenv(".env")

# validate llm
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("NV_OPENAI_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "say hi"}],
    max_tokens=50,
)
print(response.choices[0].message.content)

class Assistant(Agent):
    """Basic voice assistant with Airbnb booking capabilities."""

    def __init__(self): # 这里设定system prompt
        super().__init__(
        instructions="""You are a helpful and friendly voice assistant. Keep your responses concise and natural, 
        as if having a conversation. Make sure your responses are limited to less than 300 tokens each.
        """)

async def entrypoint(ctx: agents.JobContext): # assistant的入口函数（第一次调用）
    """Entry point for the agent."""
    # Configure the- voice pipeline with essential
    # session = AgentSession(
    # stt=deepgram.STT(model="nova-2"),
    # llm=openai.LLM(model=os.getenv("LLM_CHOICE","gpt-4.1-mini")),
    # tts=openai.TTS(voice="echo"),
    # vad=silero.VAD.load()
    # )
    from custom.stt.whisper import WhisperSTT
    stt=WhisperSTT(
            #model="u:\\home\\stoneinwind\\huggingface\\models\\deepdml\\faster-whisper-large-v3-turbo-ct2\\",
            model="u:\\home\\stoneinwind\\huggingface\\models\\systran-faster-whisper-small\\",
            language="en", # zh/ja/.etc also supports auto
            device="cuda",
            compute_type="int8_float16", # accroding to ctranslate, it is best on GPU
            model_cache_directory=False,
            #zh_lang=True,
            #init_prompt="以下是普通话的内容，请使用简体中文，并正确添加标点。" # 给zh专用
        )

    #from custom.tts.PiperTTSPluginLocal import PiperTTSPlugin # stand-alone piper exec
    # 英语
    # tts=PiperTTSPlugin("executables/piper/piper.exe", "models/piper/en_US-joe-medium.onnx", 1, 22500) # speed and samplerate
    # 中文
    #tts=PiperTTSPlugin("executables/piper/piper.exe", "models/piper/zh_CN-huayan-medium.onnx", 1, 22500) # speed and samplerate
    #from custom.tts.PiperTTSPlugin import PiperTTSPlugin # piper-tts libarary
    from custom.tts.PiperTTSStreamPlugin import PiperTTSPlugin # piper-tts stream libarary
    #tts=PiperTTSPlugin("models/piper/zh_CN-huayan-medium.onnx", use_cuda=True) # speed and samplerate
    tts=PiperTTSPlugin("models/piper/en_US-joe-medium.onnx", use_cuda=False) # speed and samplerate

    llm = openai.LLM( # 注意：这是livekit对OpenAI的包装类（不是原生，所以参数有差异）
        model="openai/gpt-oss-20b",
        base_url="https://integrate.api.nvidia.com/v1", # NV版本
        api_key=os.getenv("NV_OPENAI_KEY"),
        # 核心调参组合（复制粘贴用）
        temperature=0.2,               # 0.0 ~ 0.2，越低越确定性、越不啰嗦（推荐 0.2）
        top_p=0.9,                     # 或 1.0，不设也行
        max_completion_tokens=250,     # 强烈建议设！限制最大输出长度，天然防长篇（150~350 区间最佳）
        # presence_penalty=0.6,          # 轻微抑制新话题/展开 - nv 不支持
        # frequency_penalty=0.4,         # 减少词重复啰嗦 - nv不支持
        # 如果类支持 stop，可以加：
        # stop=["\n\n", "。", "！"],   # 强制在某些标点早停（视语言）        
        # 其他可选（如果类支持且 NVIDIA 接受）
        # timeout=30.0,
        # max_retries=2,
        reasoning_effort="low" # OpenAI支持None，但N的GPT模型只有low/medium/high
    )

    # build PP session
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load() # silero自带了VAD
    )

    # start this session
    await session.start(
        room=ctx.room, # talk history will be stored here
        agent=Assistant())

    # inital greetings
    await session.generate_reply( # LK的妙处就是对话过程中随时可以这样用代码硬插一段对话
        instructions="welcome the user warmly and ask him/her how you can help"
        #instructions="欢迎用户上线 问问你能帮什么忙"
        )

if __name__ == "__main__":
    import asyncio
    #asyncio.run())
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        # 可以指定更多参数
        # api_key=os.getenv("LIVEKIT_API_KEY"),
        # api_secret=os.getenv("LIVEKIT_API_SECRET"),
        # ws_url=os.getenv("LIVEKIT_URL"))
        )
    )
