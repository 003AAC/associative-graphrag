"""
LLM引擎 - 离线/在线双模式
离线：llama_cpp 加载本地ChatGLM3
在线：DeepSeek API（OpenAI兼容接口）
支持运行时切换和API Key动态设置
"""
import json
import logging
from typing import Optional
from config import (
    RUN_MODE, ONLINE_API_KEY, ONLINE_BASE_URL, ONLINE_MODEL, ONLINE_PROVIDER,
    OFFLINE_MODEL_PATH, OFFLINE_N_CTX, OFFLINE_N_GPU_LAYERS
)

logger = logging.getLogger(__name__)


class LLMEngine:
    """统一LLM调用接口，支持离线/在线切换"""

    def __init__(self):
        self._api_key = ONLINE_API_KEY
        self.mode = self._determine_mode()
        self._offline_llm = None
        self._online_client = None
        logger.info(f"🧠 LLM引擎初始化，运行模式: {self.mode}")

    def _determine_mode(self) -> str:
        # 检查llama-cpp-python是否可用
        self._offline_available = self._check_offline_available()
        if RUN_MODE == "offline":
            if not self._offline_available:
                logger.warning("⚠️ 离线模式不可用（llama-cpp-python未安装或模型路径为空），切换到在线模式")
                return "online" if self._api_key else "offline"
            return "offline"
        elif RUN_MODE == "online":
            return "online"
        else:  # auto
            if self._api_key:
                return "online"
            elif self._offline_available:
                return "offline"
            else:
                logger.warning("⚠️ 无API Key且离线不可用，将尝试在线模式（需设置API Key）")
                return "online"

    def _check_offline_available(self) -> bool:
        """检查离线模式是否可用"""
        if not OFFLINE_MODEL_PATH:
            return False
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_offline(self):
        """懒加载本地模型"""
        if self._offline_llm is None:
            from llama_cpp import Llama
            logger.info("🔄 加载本地模型...")
            self._offline_llm = Llama(
                model_path=OFFLINE_MODEL_PATH,
                n_ctx=OFFLINE_N_CTX,
                n_gpu_layers=OFFLINE_N_GPU_LAYERS,
                verbose=False
            )
            logger.info("✅ 本地模型加载完成")
        return self._offline_llm

    def _get_online(self):
        """懒加载在线客户端"""
        if self._online_client is None:
            from openai import OpenAI
            self._online_client = OpenAI(
                api_key=self._api_key,
                base_url=ONLINE_BASE_URL
            )
        return self._online_client

    def set_api_key(self, key: str):
        """运行时设置API Key"""
        self._api_key = key
        self._online_client = None  # 重置客户端，下次调用时重新创建
        # 如果当前是离线模式且有key了，自动切换
        if self.mode == "offline" and key:
            self.mode = "online"
            logger.info("🔄 检测到API Key，自动切换到在线模式")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """统一生成接口"""
        if self.mode == "offline":
            return self._generate_offline(prompt, max_tokens, temperature)
        else:
            return self._generate_online(prompt, max_tokens, temperature)

    def _generate_offline(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """离线生成"""
        llm = self._get_offline()
        response = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return response["choices"][0]["text"].strip()

    def _generate_online(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """在线生成（DeepSeek/OpenAI兼容接口）"""
        if not self._api_key:
            raise RuntimeError("未配置API Key，无法调用在线LLM。请在config.py中设置ONLINE_API_KEY或通过Web界面输入。")
        client = self._get_online()
        response = client.chat.completions.create(
            model=ONLINE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def switch_mode(self, mode: str):
        """运行时切换模式"""
        if mode == "online" and not self._api_key:
            raise ValueError("切换到在线模式需要先设置API Key")
        if mode in ("offline", "online"):
            self.mode = mode
            logger.info(f"🔄 切换LLM模式: {mode}")
        else:
            raise ValueError(f"不支持的模式: {mode}")

    @property
    def current_mode(self) -> str:
        return self.mode

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)
