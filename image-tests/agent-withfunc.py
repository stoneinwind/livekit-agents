from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import openai, silero # 各种rovider也用plugin方式集成
from datetime import datetime
import os
from typing import Annotated # 用于类型标注
from vision.moondream_ollama import MoondreamClient
from vision.screen_capture import ScreenCapture, WindowCapture

import sys
from pathlib import Path

# 自动把项目根加到 sys.path（livekit-agents 目录）
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from custom.stt.whisper import WhisperSTT
from custom.tts.PiperTTSStreamPlugin import PiperTTSPlugin # piper-tts stream libarary

import logging

logger = logging.getLogger(__name__)

load_dotenv(".env")

from openai import OpenAI
# # 纯测试NV的OpenAI LLM
# client = OpenAI(
#     api_key=os.getenv("NV_OPENAI_KEY"),
#     base_url="https://integrate.api.nvidia.com/v1",
# )
# response = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=[{"role": "user", "content": "say hi"}],
#     max_tokens=50,
# )
# print(response.choices[0].message.content)

global CaptureType # "screen" or "window"
global CaptureTarget # target_window_title like "Chrome"

class Assistant(Agent):
    """Basic voice assistant with screenshot and analysis capabilities."""

    def __init__(self, screen_tool: [ScreenCapture|WindowCapture], vision_tool: MoondreamClient): 
        super().__init__(
        # instructions="""You are a helpful and friendly voice assistant. Keep your responses concise and natural, 
        # as if having a conversation. Make sure your responses are limited to less than 100 tokens each.
        # You can see the user's screen if they ask. Use the 'analyze_screen' tool to understand visual context. 
        # Keep responses concise and to the point.
        # """
        instructions="You are a helpful and friendly voice assistant. Keep your responses concise and natural. Call the tool if necessary"
        )
        self.capturetool = screen_tool
        self.moondream = vision_tool

    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, any]:
        """Look up weather information for a given location.        
        Args:
            location: The location to look up weather information for.
        """
        logger.debug(f"DEBUG: Starting to query weather for: {location}")
        # Generate speech to inform the user
        # self.session.generate_reply(instructions=f"Trying to query weather for {location}. This may take a moment.") # 会触发错误
        self.session.say("Trying to query weather for {location}. This may take a moment.")        
        await context.wait_for_playout()  
        await asyncio.sleep(3)
        logger.debug("Tool: sleep on purpose. OK ")
        return {"weather": "sunny", "temperature_f": 70}

    # --- 定义视觉识别工具 ---
    @function_tool()
    async def analyze_screen(
        self,
        context: RunContext, 
        #user_query: Annotated[str, "The specific question about the screen content"] = "Describe what is on the screen"
    ) -> dict[str, any]:
        """Captures the current screen and returns the analysis result.
        """
        logger.debug(f"DEBUG: Starting screen analysis")
        # Generate speech to inform the user
        # self.session.say("Trying to capture and analyze screen. This may take a moment.")    
        # Wait for speech to complete using context.wait_for_playout()
        # Do NOT await the speech handle directly in tool calls
        # await context.wait_for_playout()        

        try:
            # 1. 截图 (返回的是raw data）
            screenshot_data = await self.capturetool.capture_once_async()
            logger.debug("Tool: Screen captured once OK and shape is : " + str(screenshot_data.shape))

            # 2. 提示等待下
            self.session.say("Trying to capture and analyze screen. This may take a moment.")        
            await context.wait_for_playout()  
            
            # 3. Moondream 识别
            # 提示：如果识别耗时较长，可以在这里先返回一个“请稍等，我正在看”的占位
            logger.debug("Tool: Starting screen analysis...waiting")
            #analysis_result = await self.moondream.describe_async(screenshot_data)       
            analysis_result = await self.moondream.describe_async(screenshot_data)       
            logger.debug("Tool: Screen analysis OK and result is : {analysis_result}")   
            return {"Screen description": analysis_result}  
        except Exception as e:
            logger.error(f"Failed to describe screen: {str(e)}")
            return None

async def entrypoint(ctx: agents.JobContext): # assistant的入口函数（第一次调用）
    """Entry point for the agent."""    
    stt=WhisperSTT(
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
    #tts=PiperTTSPlugin("models/piper/zh_CN-huayan-medium.onnx", use_cuda=True) # speed and samplerate
    tts=PiperTTSPlugin(os.path.join(root, "models/piper/en_US-joe-medium.onnx"), use_cuda=False) # speed and samplerate

    llm = openai.LLM( # 注意：这是livekit对OpenAI的包装类（不是原生，所以参数有差异）
        model="openai/gpt-oss-20b",
        base_url="https://integrate.api.nvidia.com/v1", # NV版本
        api_key=os.getenv("NV_OPENAI_KEY"),
        temperature=0.2,               # 0.0 ~ 0.2，越低越确定性、越不啰嗦（推荐 0.2）
        top_p=0.9,                     # 或 1.0，不设也行
        max_completion_tokens=100,     # 强烈建议设！限制最大输出长度，天然防长篇（150~350 区间最佳）
        reasoning_effort="low" # OpenAI支持None，但N的GPT模型只有low/medium/high
    )

    # build PP session
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load() # silero自带了VAD
    )

    assistant = Assistant(screen_tool=(ScreenCapture() if CaptureType == "screen" else WindowCapture(CaptureTarget)) , vision_tool=MoondreamClient())
    print(f"Using {CaptureType} capture tool for {CaptureTarget} and moondream vision tool")

    # start this session
    await session.start(
        room=ctx.room, # talk history will be stored here
        agent=assistant)

    # inital greetings
    await session.generate_reply( # LK的妙处就是对话过程中随时可以这样用代码硬插一段对话
        instructions="welcome the user warmly and ask him/her how you can help"
        #instructions="欢迎用户上线 问问你能帮什么忙"
        )

if __name__ == "__main__":
    import asyncio
    #asyncio.run())

    user_input = input("Which application would you like to capture? (Press Enter for Whole current screen,  or enter specific title name specific window)\n")
    if user_input == "" or user_input.lower() == "screen":
        CaptureType = "screen"
        print("CaptureType set to whole screen")
    else:
        CaptureType = "window"
        CaptureTarget = user_input.strip()
        print(f"CaptureType set to specific window: {CaptureTarget}")

    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        )
    )
